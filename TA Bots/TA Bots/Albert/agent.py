"""
Albert: Bayesian rat tracker combined with iterative-deepening alpha-beta minimax.

Each turn:
  1. Exact Bayesian belief update (two transitions + noise + distance).
  2. Run iterative-deepening alpha-beta minimax; when search EV > 0, the best
     rat search is evaluated as a chance node at the root alongside game moves.
"""

import math
import random
import time
from collections.abc import Callable
from typing import List, Tuple

import jax.numpy as jnp

from game import *
from game import board as board_mod, enums, move as move_mod
from game.enums import MoveType
from game.rat import (
    NOISE_PROBS as _NOISE_PROBS,
    DISTANCE_ERROR_PROBS as _DIST_PROBS,
    DISTANCE_ERROR_OFFSETS as _DIST_OFFSETS,
)

DEBUG = False

# ── Albert's minimax ──────────────────────────────────────────────────────────

# Move Ordering Heuristic:
# 1. Carpet moves, longer first (except for length 1)
# 2. Prime moves
# 3. Search moves
# 4. Carpet moves of length 1
def _order_key(m):
    if m.move_type == MoveType.CARPET:
        if m.roll_length == 1:
            return (3, 0)
        return (0, -m.roll_length)
    if m.move_type == MoveType.PRIME:
        return (1, 0)
    return (2, 0)

# Simple heuristic: player's points minus opponent's points.
def _heuristic(board: board_mod.Board) -> float:
    return board.player_worker.points - board.opponent_worker.points


# Standard alpha-beta minimax
def _minimax(
    board: board_mod.Board,
    depth: int,
    maximizing: bool,
    alpha: float = -math.inf,
    beta: float = math.inf,
    time_limit: float = None,
    leaf_count: List[int] = None,
    search_info=None,
) -> Tuple[float, object]:
    if time_limit and time.time() > time_limit:
        return None, None

    if depth == 0 or board.is_game_over():
        if not maximizing:
            board.reverse_perspective()
        if leaf_count is not None:
            leaf_count[0] += 1
        return _heuristic(board), None

    valid_moves = sorted(board.get_valid_moves(), key=_order_key)

    if maximizing:
        max_val = -math.inf
        best_move = None
        for mv in valid_moves:
            child = board.forecast_move(mv)
            if child:
                child.reverse_perspective()
                val, _ = _minimax(child, depth - 1, False, alpha, beta, time_limit, leaf_count)
                if val is None:
                    return None, None
                if val > max_val:
                    max_val, best_move = val, mv
                alpha = max(alpha, val)
                if beta <= alpha:
                    break

        # Evaluate the best search move as a chance node at the root, using the rat belief.
        # We do not evaluate search nodes below the root as chance nodes since the beliefs would be different.
        if search_info is not None:
            best_p, best_x, best_y = search_info

            # Note that for both simulations, we increase the depth by 1 to account for the fact
            # that the search move uses up a turn without changing the board state (except for points). T

            # Simulate a hit
            board_hit = board.get_copy()
            board_hit.player_worker.increment_points(enums.RAT_BONUS)
            board_hit.end_turn()
            board_hit.reverse_perspective()
            hit_val, _ = _minimax(board_hit, depth - 1, False,
                                   -math.inf, math.inf, time_limit, leaf_count)

            # Simulate a miss
            board_miss = board.get_copy()
            board_miss.player_worker.decrement_points(enums.RAT_PENALTY)
            board_miss.end_turn()
            board_miss.reverse_perspective()
            miss_val, _ = _minimax(board_miss, depth - 1, False,
                                    -math.inf, math.inf, time_limit, leaf_count)

            if hit_val is not None and miss_val is not None:
                search_val = best_p * hit_val + (1 - best_p) * miss_val
                if search_val > max_val:
                    max_val = search_val
                    best_move = move_mod.Move.search((best_x, best_y))

        return max_val, best_move
    else:
        # We do not need search EV evaluation for opponent moves since we only use it at the root.
        # so this code is much simpler
        min_val = math.inf
        for mv in valid_moves:
            child = board.forecast_move(mv)
            if child:
                child.reverse_perspective()
                val, _ = _minimax(child, depth - 1, True, alpha, beta, time_limit, leaf_count)
                if val is None:
                    return None, None
                if val < min_val:
                    min_val = val
                beta = min(beta, val)
                if beta <= alpha:
                    break
        return min_val, None


# ── Agent ─────────────────────────────────────────────────────────────────────

class PlayerAgent:

    def __init__(self, board: board_mod.Board, transition_matrix=None, time_left: Callable = None):
        N = enums.BOARD_SIZE
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

        self.evaluation_depths = []

    def commentate(self):
        if DEBUG:
            summary = f"Caught the rat {self.catches} time{'s' if self.catches != 1 else ''}!"
            return summary + " LOG:" + str(self._search_log)
        else:
            return str(round(sum(self.evaluation_depths) / len(self.evaluation_depths), 2)) if self.evaluation_depths else "0"

    def play(
        self,
        board: board_mod.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
    
        noise, dist = sensor_data
        wx, wy = board.player_worker.get_location()

        # ── resolve last search result ────────────────────────────────────────
        caught = board.player_search[1] is True
        if self._pending_conf is not None:
            self._search_log.append([round(self._pending_conf, 6), caught])
            self._pending_conf = None
        if caught:
            self.catches += 1
            self._reset()

        if board.opponent_search[1] is True:
            self._reset()

        # ── belief update ─────────────────────────────────────────────────────
        # Rat moves every turn before sampling → two transitions per play() call.
        if not self._first_turn:
            self._transition()
            self._transition()
        self._first_turn = False

        self._update_noise(int(noise), board)
        self._update_distance(int(dist), wx, wy)

        # ── rat belief stats ──────────────────────────────────────────────────
        best_idx = int(jnp.argmax(self.belief))
        best_y, best_x = divmod(best_idx, self.N)
        best_p = float(self.belief[best_idx])
        search_ev = best_p * enums.RAT_BONUS - (1 - best_p) * enums.RAT_PENALTY
        search_info = (best_p, best_x, best_y) if search_ev > 0 else None

        # ── iterative-deepening minimax (search evaluated as chance node at root) ──
        time_limit = time.time() + 4 # 4 seconds per turn
        best_move = None
        leaf_count = [0]
        evaluated_depth = 0

        # Iterative deepening until we run out of time.
        for depth in range(1, 31):
            _, mv = _minimax(board.get_copy(), depth, True,
                             time_limit=time_limit, leaf_count=leaf_count,
                             search_info=search_info)
            if mv is not None:
                best_move = mv
                evaluated_depth = depth
            else:
                break

        if best_move is None:
            valid_moves = board.get_valid_moves()
            best_move = random.choice(valid_moves) if valid_moves else None

        if best_move is not None and best_move.move_type == MoveType.SEARCH:
            self._pending_conf = best_p
            if DEBUG:
                print(f"[Albert] SEARCH ({best_x},{best_y}) P={best_p:.4f}")
        else:
            if DEBUG:
                print(f"[Albert] MINIMAX depth={evaluated_depth} leaves={leaf_count[0]}")
        self.evaluation_depths.append(evaluated_depth)
        return best_move

    # ── private (belief tracker) ──────────────────────────────────────────────

    def _normalize(self):
        s = self.belief.sum()
        if s < 1e-300:
            self.belief = jnp.array(self._stationary)
            s = self.belief.sum()
        self.belief = self.belief / s

    def _reset(self):
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
