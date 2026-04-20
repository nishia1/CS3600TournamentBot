"""
George: Greedy agent with Bayesian rat tracking — no lookahead.

Each turn, the first applicable action in priority order is taken:
  1. CARPET  – roll the longest available carpet (≥2 cells). If the run is
               short enough to extend (< 7), the opponent isn't on the carpet,
               and turns remain, prime to extend instead of rolling.
               When search EV beats the carpet points, search wins.
               In the late game (≤4 turns left) always roll immediately.
  2. SEARCH  – search the highest-probability rat cell if EV exceeds both
               the best available prime value and 1.0.
  3. PRIME   – prime in the direction with the highest projected carpet value.
  4. WALK    – BFS toward the SPACE cell with the best priming potential;
               if the walk direction is also a valid prime, prime instead.
               If search EV > 1.0, search rather than walk aimlessly.
  5. FALLBACK – search if EV > 0, otherwise first legal move.

Opponent search results are used to update belief: a hit resets to the
stationary prior (rat respawned); a miss zeroes out only the searched cell.
"""

import json
import random
from collections import deque
from collections.abc import Callable

import jax.numpy as jnp
from typing import Tuple

from game import *
from game import enums, board as board_mod, move as move_mod
from game.enums import Cell, Direction, MoveType, BOARD_SIZE
from game.rat import (
    NOISE_PROBS as _NOISE_PROBS,
    DISTANCE_ERROR_PROBS as _DIST_PROBS,
    DISTANCE_ERROR_OFFSETS as _DIST_OFFSETS,
)

DEBUG = False

_CPT = [0, -1, 2, 4, 6, 10, 15, 21]

_DIR_DELTAS = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}

_REVERSE_DIR = {
    Direction.UP:    Direction.DOWN,
    Direction.DOWN:  Direction.UP,
    Direction.LEFT:  Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


def _count_primed_run(board, x, y, direction):
    """Count consecutive PRIMED cells starting from (x,y) in direction (exclusive of (x,y))."""
    dx, dy = _DIR_DELTAS[direction]
    primed = board._primed_mask
    count = 0
    cx, cy = x + dx, y + dy
    while 0 <= cx < 8 and 0 <= cy < 8:
        bit = (cy << 3) | cx
        if primed >> bit & 1:
            count += 1
            cx += dx
            cy += dy
        else:
            break
    return count


def _count_space_run(board, x, y, direction):
    """Count consecutive SPACE cells starting from (x,y) in direction (exclusive of (x,y))."""
    dx, dy = _DIR_DELTAS[direction]
    space = board._space_mask
    count = 0
    cx, cy = x + dx, y + dy
    while 0 <= cx < 8 and 0 <= cy < 8:
        bit = (cy << 3) | cx
        if space >> bit & 1:
            count += 1
            cx += dx
            cy += dy
        else:
            break
    return count


def _prime_direction_value(board, px, py, direction):
    """
    Score priming at (px, py) facing `direction`. Returns (value, can_prime).

    Counts the primed run that would result (cells behind + this cell + cells ahead),
    then projects the maximum potential run length by adding SPACE cells further ahead.
    Value = _CPT[potential_run] * (0.3 + 0.7 * completion_fraction), blending the
    full carpet payout with how close the run already is to completion.
    Returns (0.5, True) when the potential run is < 2 (only the +1 prime point).
    Returns (-999, False) when the prime is illegal or the current cell is not SPACE.
    """
    dx, dy = _DIR_DELTAS[direction]
    nx, ny = px + dx, py + dy

    if not (0 <= nx < 8 and 0 <= ny < 8):
        return -999, False
    m = move_mod.Move.prime(direction)
    if not board.is_valid_move(m):
        return -999, False

    pbit = (py << 3) | px
    if not (board._space_mask >> pbit & 1):
        return -999, False

    rev = _REVERSE_DIR[direction]
    primed_behind = _count_primed_run(board, px, py, rev)
    primed_ahead  = _count_primed_run(board, nx, ny, direction)
    far_x = nx + dx * primed_ahead
    far_y = ny + dy * primed_ahead
    space_ahead   = _count_space_run(board, far_x, far_y, direction)

    current_run   = primed_behind + 1 + primed_ahead
    potential_run = min(current_run + space_ahead, 7)

    if potential_run >= 2:
        carpet_val = _CPT[potential_run]
        completion_fraction = current_run / potential_run
        value = carpet_val * (0.3 + 0.7 * completion_fraction)
    else:
        value = 0.5  # just the +1 prime point, no profitable carpet ahead

    return value, True


def _prime_value_at(board, px, py, direction):
    """
    Like _prime_direction_value but skips board.is_valid_move — used for scoring
    hypothetical walk destinations where the worker isn't actually there yet.
    Returns a numeric score only (no can_prime flag).
    """
    dx, dy = _DIR_DELTAS[direction]
    nx, ny = px + dx, py + dy
    if not (0 <= nx < 8 and 0 <= ny < 8):
        return -999
    pbit = (py << 3) | px
    if not (board._space_mask >> pbit & 1):
        return -999
    nbit = (ny << 3) | nx
    if not ((board._space_mask | board._carpet_mask) >> nbit & 1):
        return -999

    rev = _REVERSE_DIR[direction]
    primed_behind = _count_primed_run(board, px, py, rev)
    primed_ahead  = _count_primed_run(board, nx, ny, direction)
    far_x = nx + dx * primed_ahead
    far_y = ny + dy * primed_ahead
    space_ahead   = _count_space_run(board, far_x, far_y, direction)

    current_run   = primed_behind + 1 + primed_ahead
    potential_run = min(current_run + space_ahead, 7)

    if potential_run >= 2:
        completion_fraction = current_run / potential_run
        return _CPT[potential_run] * (0.3 + 0.7 * completion_fraction)
    return 0.5



class PlayerAgent:

    def __init__(self, board: board_mod.Board, transition_matrix=None, time_left: Callable = None):
        N = BOARD_SIZE
        self.N = N

        T = jnp.asarray(transition_matrix, dtype=jnp.float32)
        self.TT = jnp.asarray(T.T)

        pi = jnp.ones(N * N, dtype=jnp.float32) / (N * N)
        for _ in range(2000):
            pi = self.TT @ pi
            pi = pi / pi.sum()
        self._stationary = pi

        self.belief = jnp.array(self._stationary)
        self._normalize()

        self.catches = 0
        self._first_turn = True
        self._search_log = []
        self._pending_conf = None

        self.GEORGE_QUOTES = [
            "If George had a nickel for every time he caught the rat, he'd have {catches} nickels",
            "Well played. Probably.",
            "George has left the chat",
            "Curious George",
            "George needs a better gaming chair",
            "smoooooth operatooorrr",
            "Objective-C: The Big Nerd Ranch Guide",
            "George will remember this",
            "Answered by Ansel on Edstem",
            "George dreams to be like Carrie some day",
            "The G in AGI stands for George",
            "George is seeking 2026 summer internships",
            "Banana Peel was here",
            "Join fencing, stab your friends!",
            "George's Institute of Technology",
            "Everything Not Saved Will Be Lost.",
            "MIT: The Georgia Tech of the North",
            "I think, therefore I am.",
            "AGI IS ONLY 2 YEARS AWAY!!!",
            "Still better than Siri",
            "THWG!",
            "Invest in Cloudman Capital",
            "Maybe George know's whats on the final?",
            "i'm tired...",
            "My battery is low and it's getting dark"
        ]

    def commentate(self):
        if DEBUG:
            summary = f"Caught the rat {self.catches} time{'s' if self.catches != 1 else ''}!"
            return summary + " LOG:" + json.dumps(self._search_log)
        else:
            return random.choice(self.GEORGE_QUOTES).format(catches=self.catches)

    def play(self, board: board_mod.Board, sensor_data: Tuple, time_left: Callable):
        noise, dist = sensor_data
        wx, wy = board.player_worker.get_location()
        turns_left = board.player_worker.turns_left

        # ── resolve last search ──
        caught = board.player_search[1] is True
        if self._pending_conf is not None:
            self._search_log.append([round(self._pending_conf, 6), caught])
            self._pending_conf = None
        if caught:
            self.catches += 1
            self._reset_belief()

        opp_loc, opp_result = board.opponent_search
        if opp_result is True:
            # Opponent caught the rat — it respawned, reset to prior
            self._reset_belief()
        elif opp_result is False and opp_loc is not None:
            # Opponent searched and missed — rat wasn't there, zero out that cell
            ox, oy = opp_loc
            idx = oy * self.N + ox
            self.belief = self.belief.at[idx].set(0.0)
            self._normalize()

        # ── belief update ──
        if not self._first_turn:
            self._transition()
            self._transition()
        self._first_turn = False
        self._update_noise(int(noise), board)
        self._update_distance(int(dist), wx, wy)

        # ── search EV ──
        best_idx = int(jnp.argmax(self.belief))
        best_p = float(self.belief[best_idx])
        best_ry, best_rx = divmod(best_idx, self.N)
        search_ev = best_p * enums.RAT_BONUS - (1 - best_p) * enums.RAT_PENALTY

        # ── precompute best prime direction (shared by search threshold + prime step) ──
        pbit = (wy << 3) | wx
        can_prime = bool(board._space_mask >> pbit & 1)
        best_prime_val, best_prime_dir = 0.0, None
        if can_prime:
            for d in Direction:
                v, ok = _prime_direction_value(board, wx, wy, d)
                if ok and v > best_prime_val:
                    best_prime_val, best_prime_dir = v, d

        # ── 1. CARPET if available (≥2 cells) — try to extend first ──
        best_carpet = None
        best_carpet_pts = 0
        for m in board.get_valid_moves():
            if m.move_type == MoveType.CARPET and m.roll_length >= 2:
                val = _CPT[min(m.roll_length, 7)]
                if val > best_carpet_pts:
                    best_carpet_pts = val
                    best_carpet = m

        if best_carpet is not None:
            # Late game: stop extending, cash in immediately
            late_game = turns_left <= 4

            can_extend = False
            if not late_game:
                ext_dir = _REVERSE_DIR[best_carpet.direction]
                ex, ey = board.opponent_worker.get_location()
                opp_near = False
                dx_c, dy_c = _DIR_DELTAS[best_carpet.direction]
                for i in range(1, best_carpet.roll_length + 1):
                    cx = wx + dx_c * i
                    cy = wy + dy_c * i
                    if abs(ex - cx) + abs(ey - cy) <= 2:
                        opp_near = True
                        break
                if (can_prime and best_carpet.roll_length < 7 and not opp_near):
                    m_ext = move_mod.Move.prime(ext_dir)
                    if board.is_valid_move(m_ext) and turns_left > 2:
                        can_extend = True

            if can_extend:
                new_len = best_carpet.roll_length + 1
                if DEBUG:
                    print(f"[George] EXTEND prime {ext_dir.name} "
                        f"run {best_carpet.roll_length}→{new_len}")
                return m_ext

            # Compare carpet against search (use raw points, not padded score)
            if search_ev > 0 and search_ev > best_carpet_pts:
                self._pending_conf = best_p
                if DEBUG:
                    print(f"[George] SEARCH over carpet ({best_rx},{best_ry}) "
                        f"P={best_p:.4f} ev={search_ev:.2f}")
                return move_mod.Move.search((best_rx, best_ry))
            if DEBUG:
                print(f"[George] CARPET {best_carpet.direction.name} "
                    f"x{best_carpet.roll_length} = {best_carpet_pts} pts")
            return best_carpet

        # ── 2. SEARCH if EV exceeds the best prime value ──
        if search_ev > max(best_prime_val, 1.0):
            self._pending_conf = best_p
            if DEBUG:
                print(f"[George] SEARCH ({best_rx},{best_ry}) P={best_p:.4f} ev={search_ev:.2f}")
            return move_mod.Move.search((best_rx, best_ry))

        # ── 3. PRIME in the best direction (precomputed above) ──
        if can_prime and best_prime_dir is not None:
            if DEBUG:
                print(f"[George] PRIME {best_prime_dir.name} val={best_prime_val:.2f}")
            return move_mod.Move.prime(best_prime_dir)

        # ── 4. WALK toward the best-positioned SPACE cell ──
        # If search_ev > 1.0, searching beats aimless walking
        if search_ev > 1.0:
            self._pending_conf = best_p
            if DEBUG:
                print(f"[George] SEARCH-WALK ({best_rx},{best_ry}) P={best_p:.4f} ev={search_ev:.2f}")
            return move_mod.Move.search((best_rx, best_ry))

        move = self._walk_toward_best_space(board, wx, wy)
        if move is not None:
            # If current cell is SPACE, try to prime in the walk direction (free +1)
            if can_prime:
                m_prime = move_mod.Move.prime(move.direction)
                if board.is_valid_move(m_prime):
                    if DEBUG:
                        print(f"[George] PRIME-WALK {move.direction.name}")
                    return m_prime
            if DEBUG:
                print(f"[George] WALK {move.direction.name}")
            return move

        # ── 5. Fallback ──
        if search_ev > 0:
            self._pending_conf = best_p
            return move_mod.Move.search((best_rx, best_ry))

        valid = board.get_valid_moves()
        return valid[0] if valid else move_mod.Move.search((best_rx, best_ry))

    def _walk_toward_best_space(self, board, wx, wy):
        """BFS over reachable cells; walk toward the SPACE cell with best prime potential."""
        space     = board._space_mask
        open_mask = space | board._carpet_mask
        first_dir = {(wx, wy): None}
        q = deque([(wx, wy)])
        best_score, best_fd = -999.0, None

        while q:
            x, y = q.popleft()
            for d in Direction:
                dx, dy = _DIR_DELTAS[d]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                if (nx, ny) in first_dir:
                    continue
                nbit = (ny << 3) | nx
                if not (open_mask >> nbit & 1):
                    continue
                fd = first_dir[(x, y)] if first_dir[(x, y)] is not None else d
                first_dir[(nx, ny)] = fd
                if space >> nbit & 1:
                    score = max(_prime_value_at(board, nx, ny, d2) for d2 in Direction)
                    if score > best_score:
                        best_score, best_fd = score, fd
                q.append((nx, ny))

        if best_fd is None:
            return None
        m = move_mod.Move.plain(best_fd)
        return m if board.is_valid_move(m) else None

    # ── Bayesian rat tracker ──────────────────────────────────────────────────

    def _normalize(self):
        s = self.belief.sum()
        if s < 1e-300:
            self.belief = jnp.array(self._stationary)
            s = self.belief.sum()
        self.belief = self.belief / s

    def _reset_belief(self):
        self.belief = jnp.array(self._stationary)
        self._normalize()

    def _transition(self):
        self.belief = self.TT @ self.belief
        self._normalize()

    def _update_noise(self, noise_idx: int, board: board_mod.Board):
        N = self.N
        likelihood = jnp.array([_NOISE_PROBS[int(board.get_cell((x, y)))][noise_idx]
                                 for y in range(N) for x in range(N)], dtype=jnp.float32)
        self.belief = self.belief * likelihood
        self._normalize()

    def _update_distance(self, obs_dist: int, wx: int, wy: int):
        N = self.N
        xs = jnp.arange(N)
        ys = jnp.arange(N)
        dist_flat = (jnp.abs(xs[None, :] - wx) + jnp.abs(ys[:, None] - wy)).ravel()
        offsets = obs_dist - dist_flat
        likelihood = jnp.zeros(N * N, dtype=jnp.float32)
        for i, off in enumerate(_DIST_OFFSETS):
            likelihood = likelihood.at[offsets == off].set(_DIST_PROBS[i])
        self.belief = self.belief * likelihood
        self._normalize()
