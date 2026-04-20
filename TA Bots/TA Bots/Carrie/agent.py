"""
Carrie: Deep search via engine optimizations on top of Albert's minimax.

Identical heuristic and belief tracker to Albert.  The minimax engine adds:
  1. Transposition table  – exact / lower / upper bounds, keyed on board state.
     Replacement policy: depth-preferred (never overwrite a deeper entry).
  2. Killer move heuristic – 2 quiet killers stored per depth-from-root slot.
  3. History heuristic     – non-carpet moves that caused cutoffs are up-scored.
  4. TT-move-first ordering – best move from TT is searched first.
  5. Best-move stored for min nodes in TT (improves move ordering at all depths).
  6. Quiescence search     – long carpets (roll≥3) searched past depth=0.
  7. Null move pruning     – at depth ≥ 3, skip our turn; if result ≥ beta,
     prune without searching (R=2 reduction, disabled at root and in null moves).

The TT is created fresh each turn (avoids stale entries across very different
board states).  Killers and history are per-turn as well so ordering adapts to
the current position during iterative deepening.
"""

import json
import math
import random
import time
import jax.numpy as jnp

from collections import deque
from collections.abc import Callable
from typing import List, Optional, Tuple

from game import *
from game import enums, board as board_mod, move as move_mod
from game.enums import MoveType
from game.rat import (
    NOISE_PROBS as _NOISE_PROBS,
    DISTANCE_ERROR_PROBS as _DIST_PROBS,
    DISTANCE_ERROR_OFFSETS as _DIST_OFFSETS,
)

DEBUG = False

# ── heuristic  ───────────────────────────────────────────
_CPT = [0, -1, 2, 4, 6, 10, 15, 21]
_INF_DIST = 127
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _movement_dist(start_bit: int, space_mask: int, carpet_mask: int, primed_mask: int) -> list:
    """BFS distance (turns) from start to all reachable bits, accounting for primed jumps."""
    open_mask = space_mask | carpet_mask
    dist = [_INF_DIST] * 64
    dist[start_bit] = 0
    q = deque([start_bit])
    while q:
        u = q.popleft()
        d1 = dist[u] + 1
        ux = u & 7
        uy = u >> 3
        for dx, dy in _DIRS:
            nx, ny = ux + dx, uy + dy
            if nx < 0 or nx >= 8 or ny < 0 or ny >= 8:
                continue
            nbit = (ny << 3) | nx
            if open_mask >> nbit & 1:
                if dist[nbit] > d1:
                    dist[nbit] = d1
                    q.append(nbit)
            elif primed_mask >> nbit & 1:
                cx, cy = nx, ny
                while 0 <= cx < 8 and 0 <= cy < 8:
                    cb = (cy << 3) | cx
                    if not (primed_mask >> cb & 1):
                        break
                    if dist[cb] > d1:
                        dist[cb] = d1
                        q.append(cb)
                    cx += dx
                    cy += dy
    return dist

"""
The core concept of this heuristic is "potential". 
Each square has a potential based on the longest primed run from that square (or 1 if it's a non-primed traversable square).
The potential is then weighted by turns_left / (distance + 1) to get a score contribution for that square, the idea being that
a square that is closer and more accessible (higher potential) is more valuable, especially if we have many turns left to take advantage of it.
We sum up the potential contributions for all squares, and take the difference between our potential and the opponent's potential.
Like Albert, the current score difference is also used, but we also scale this by turns left so it is the same scale as potential (the maximum weight for potential is turns_left, ie when distance is 1)
"""
def _heuristic(board: board_mod.Board) -> float:
    w = board.player_worker
    e = board.opponent_worker
    score_diff = w.points - e.points
    p_turns = w.turns_left
    e_turns = e.turns_left

    primed      = board._primed_mask
    space       = board._space_mask
    carpet      = board._carpet_mask
    traversable = space | carpet | primed

    if not traversable:
        return float(score_diff * max(p_turns, 1))

    px, py = w.get_location()
    ex, ey = e.get_location()
    p_bit = py * 8 + px
    e_bit = ey * 8 + ex

    dp = _movement_dist(p_bit, space, carpet, primed)
    de = _movement_dist(e_bit, space, carpet, primed)

    R = [0] * 64; L = [0] * 64; D = [0] * 64; U = [0] * 64
    if primed:
        for y in range(8):
            b = y * 8
            for x in range(6, -1, -1):
                if primed >> (b + x + 1) & 1:
                    R[b + x] = 1 + R[b + x + 1]
            for x in range(1, 8):
                if primed >> (b + x - 1) & 1:
                    L[b + x] = 1 + L[b + x - 1]
        for x in range(8):
            for y in range(6, -1, -1):
                if primed >> ((y + 1) * 8 + x) & 1:
                    D[y * 8 + x] = 1 + D[(y + 1) * 8 + x]
            for y in range(1, 8):
                if primed >> ((y - 1) * 8 + x) & 1:
                    U[y * 8 + x] = 1 + U[(y - 1) * 8 + x]

    p_sum = e_sum = 0.0
    for y in range(8):
        b = y * 8
        for x in range(8):
            bit = b + x
            if not (traversable >> bit & 1):
                continue
            max_c = 0
            if primed:
                for run in (R[bit], L[bit], D[bit], U[bit]):
                    if run >= 2:
                        v = _CPT[run if run <= 7 else 7]
                        if v > max_c:
                            max_c = v
            if space >> bit & 1:
                if max_c < 1:
                    max_c = 1
            elif max_c <= 0:
                continue
            d_p = dp[bit]
            d_e = de[bit]
            if d_p < d_e:
                if d_p < _INF_DIST:
                    p_sum += max_c * p_turns / (d_p + 1)
            elif d_e < d_p:
                if d_e < _INF_DIST:
                    e_sum += max_c * e_turns / (d_e + 1)

    return score_diff * max(p_turns, 1) + (p_sum - e_sum) * 0.5


# ── transposition table ────────────────────────────────────────────────────────
_TT_EXACT = 0
_TT_LOWER = 1   # fail-high: stored value is a lower bound
_TT_UPPER = 2   # fail-low:  stored value is an upper bound

# ── quiescence search depth ────────────────────────────────────────────────────
_Q_DEPTH = 3    # extra plies beyond depth=0, searching long carpets only

# ── null move pruning ──────────────────────────────────────────────────────────
_NMP_R = 2      # depth reduction for null move search


def _tt_store(tt: dict, key: tuple, depth: int, val: float, flag: int, mid) -> None:
    """Store entry in TT with depth-preferred replacement.

    Never overwrites an existing entry that was searched to a greater depth.
    No size cap — Python dict size is bounded by the 1-second search budget.
    """
    existing = tt.get(key)
    if existing is not None and existing[0] > depth:
        return  # keep the deeper entry
    tt[key] = (depth, val, flag, mid)


def _board_key(board: board_mod.Board, maximizing: bool) -> tuple:
    """Compact, hashable key for the current board state and perspective."""
    px, py = board.player_worker.get_location()
    ex, ey = board.opponent_worker.get_location()
    return (
        board._primed_mask,
        board._carpet_mask,
        (py << 3) | px,
        (ey << 3) | ex,
        board.player_worker.turns_left,
        board.opponent_worker.turns_left,
        board.player_worker.points,
        board.opponent_worker.points,
        maximizing,
    )


# ── move representation for killers/history (Move has no __eq__) ──────────────
def _move_id(m) -> tuple:
    """Hashable identity of a move (ignores object identity)."""
    if m.move_type == MoveType.CARPET:
        return (MoveType.CARPET, m.direction, m.roll_length)
    if m.move_type == MoveType.SEARCH:
        return (MoveType.SEARCH, m.search_loc)
    return (m.move_type, m.direction)


# ── move ordering ──────────────────────────────────────────────────────────────
def _order_key(m, tt_move_id, killer_ids, history) -> tuple:
    """
    Sort priority (ascending — lower = searched first):
      0. TT best move
      1. Good carpets (length >= 2), longest first
      2. Killer quiet moves
      3. Prime (history-scored)
      4. Plain (history-scored)
      5. Bad carpet (length 1)
    """
    mid = _move_id(m)
    if mid == tt_move_id:
        return (-5, 0)
    if m.move_type == MoveType.CARPET:
        if m.roll_length >= 2:
            return (-3, -m.roll_length)
        return (4, 0)
    if mid in killer_ids:
        return (-2, 0)
    if m.move_type == MoveType.PRIME:
        return (0, -history.get(mid, 0))
    # PLAIN
    return (1, -history.get(mid, 0))


# ── quiescence search ─────────────────────────────────────────────────────────

def _quiescence(
    board: board_mod.Board,
    maximizing: bool,
    alpha: float,
    beta: float,
    qdepth: int,
    leaf_count: List[int],
) -> float:
    """
    Continue searching only long carpet moves (roll_length >= 3) after depth=0.

    Stand-pat: the current static evaluation is always an option (the node can
    choose to make no noisy move).  We prune with alpha-beta and stop when
    qdepth reaches 0 or no long carpets remain.
    """
    # Count once per entry from _minimax (outermost quiescence call)
    if qdepth == _Q_DEPTH:
        leaf_count[0] += 1

    # Stand-pat evaluation — heuristic expects player_worker to be the side to move
    if not maximizing:
        board.reverse_perspective()
    stand_pat = _heuristic(board)
    if not maximizing:
        board.reverse_perspective()

    if maximizing:
        if stand_pat >= beta:
            return stand_pat
        best = stand_pat
        alpha = max(alpha, stand_pat)
        if qdepth == 0:
            return best
        for m in board.get_valid_moves():
            if m.move_type != MoveType.CARPET or m.roll_length < 3:
                continue
            child = board.forecast_move(m)
            if child is None:
                continue
            child.reverse_perspective()
            val = _quiescence(child, False, alpha, beta, qdepth - 1, leaf_count)
            if val > best:
                best = val
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    else:  # minimizing
        if stand_pat <= alpha:
            return stand_pat
        best = stand_pat
        beta = min(beta, stand_pat)
        if qdepth == 0:
            return best
        for m in board.get_valid_moves():
            if m.move_type != MoveType.CARPET or m.roll_length < 3:
                continue
            child = board.forecast_move(m)
            if child is None:
                continue
            child.reverse_perspective()
            val = _quiescence(child, True, alpha, beta, qdepth - 1, leaf_count)
            if val < best:
                best = val
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


# ── minimax with TT + killers + history ───────────────────────────────────────
def _minimax(
    board: board_mod.Board,
    depth: int,
    maximizing: bool,
    alpha: float,
    beta: float,
    time_limit: float,
    leaf_count: List[int],
    tt: dict,
    killers: list,      # killers[depth_from_root] = [move_id, ...]
    history: dict,      # history[move_id] = score
    search_info=None,   # (best_p, best_x, best_y) at root only
    _dfr: int = 0,      # depth-from-root (for killer slot indexing)
    _null_move: bool = False,  # True when called from null move pruning
) -> Tuple[Optional[float], object]:
    """Alpha-beta with TT, killers, history, null move pruning, and expectiminimax search at root."""

    if time.time() > time_limit:
        return None, None

    # ── TT lookup ─────────────────────────────────────────────────────────────
    key = _board_key(board, maximizing)
    tt_move_id = None
    orig_alpha  = alpha

    if key in tt:
        tt_depth, tt_val, tt_flag, tt_mid = tt[key]
        tt_move_id = tt_mid
        if tt_depth >= depth:
            if tt_flag == _TT_EXACT:
                # Only return at root if we have an associated move stored
                if tt_mid is not None or depth == 0:
                    return tt_val, None   # caller uses best_move from ID loop
            elif tt_flag == _TT_LOWER:
                alpha = max(alpha, tt_val)
            elif tt_flag == _TT_UPPER:
                beta = min(beta, tt_val)
            if alpha >= beta:
                return tt_val, None

    # ── leaf: quiescence search ───────────────────────────────────────────────
    if depth == 0 or board.is_game_over():
        val = _quiescence(board, maximizing, alpha, beta, _Q_DEPTH, leaf_count)
        _tt_store(tt, key, 0, val, _TT_EXACT, None)
        return val, None

    # ── generate + order moves ────────────────────────────────────────────────
    killer_ids = set(killers[_dfr]) if _dfr < len(killers) else set()
    valid_moves = sorted(
        board.get_valid_moves(),
        key=lambda m: _order_key(m, tt_move_id, killer_ids, history),
    )

    if maximizing:
        best_val, best_move = -math.inf, None

        # ── null move pruning ──────────────────────────────────────────────
        # Skip our turn; if the opponent still can't beat beta, prune.
        # Disabled at root (search_info present), in null move chains, and
        # in the endgame (few turns left, where passing is meaningful).
        if (not _null_move and search_info is None
                and depth >= 3
                and board.player_worker.turns_left > 3):
            null_board = board.get_copy()
            null_board.end_turn()
            null_board.reverse_perspective()
            null_val, _ = _minimax(
                null_board, depth - _NMP_R - 1, False, alpha, beta,
                time_limit, leaf_count, tt, killers, history,
                _dfr=_dfr + 1, _null_move=True,
            )
            if null_val is not None and null_val >= beta:
                return null_val, None

        for m in valid_moves:
            child = board.forecast_move(m)
            if child is None:
                continue
            child.reverse_perspective()
            mid = _move_id(m)
            val, _ = _minimax(
                child, depth - 1, False, alpha, beta,
                time_limit, leaf_count, tt, killers, history,
                _dfr=_dfr + 1,
            )
            if val is None:
                return None, None
            if val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
            if beta <= alpha:
                # Beta cutoff — update killers and history for quiet moves
                if m.move_type != MoveType.CARPET and _dfr < len(killers):
                    kl = killers[_dfr]
                    if mid not in kl:
                        kl.insert(0, mid)
                        if len(kl) > 2:
                            kl.pop()
                    history[mid] = history.get(mid, 0) + depth * depth
                break

        # ── search chance node at root ─────────────────────────────────────
        # This is the same logic as Albert
        if search_info is not None:
            best_p, best_x, best_y = search_info

            board_hit = board.get_copy()
            board_hit.player_worker.increment_points(enums.RAT_BONUS)
            board_hit.end_turn()
            board_hit.reverse_perspective()
            hit_val, _ = _minimax(
                board_hit, depth - 1, False, -math.inf, math.inf,
                time_limit, leaf_count, tt, killers, history, _dfr=_dfr + 1,
            )

            board_miss = board.get_copy()
            board_miss.player_worker.decrement_points(enums.RAT_PENALTY)
            board_miss.end_turn()
            board_miss.reverse_perspective()
            miss_val, _ = _minimax(
                board_miss, depth - 1, False, -math.inf, math.inf,
                time_limit, leaf_count, tt, killers, history, _dfr=_dfr + 1,
            )

            if hit_val is not None and miss_val is not None:
                search_val = best_p * hit_val + (1 - best_p) * miss_val
                if search_val > best_val:
                    best_val = search_val
                    best_move = move_mod.Move.search((best_x, best_y))

        # ── store in TT ───────────────────────────────────────────────────
        if best_move is not None:
            if best_val <= orig_alpha:
                flag = _TT_UPPER
            elif best_val >= beta:
                flag = _TT_LOWER
            else:
                flag = _TT_EXACT
            _tt_store(tt, key, depth, best_val, flag, _move_id(best_move))

        return best_val, best_move

    else:  # minimizing
        best_val  = math.inf
        best_move = None

        for m in valid_moves:
            child = board.forecast_move(m)
            if child is None:
                continue
            child.reverse_perspective()
            mid = _move_id(m)
            val, _ = _minimax(
                child, depth - 1, True, alpha, beta,
                time_limit, leaf_count, tt, killers, history,
                _dfr=_dfr + 1,
            )
            if val is None:
                return None, None
            if val < best_val:
                best_val = val
                best_move = m
            beta = min(beta, val)
            if beta <= alpha:
                if m.move_type != MoveType.CARPET and _dfr < len(killers):
                    kl = killers[_dfr]
                    if mid not in kl:
                        kl.insert(0, mid)
                        if len(kl) > 2:
                            kl.pop()
                    history[mid] = history.get(mid, 0) + depth * depth
                break

        _tt_store(tt, key, depth, best_val,
                  _TT_UPPER if best_val <= alpha else _TT_EXACT,
                  _move_id(best_move) if best_move is not None else None)

        return best_val, None


# ── agent ──────────────────────────────────────────────────────────────────────
class PlayerAgent:

    def __init__(
        self,
        board: board_mod.Board,
        transition_matrix=None,
        time_left: Callable = None,
    ):
        N = enums.BOARD_SIZE
        self.N = N

        T = jnp.asarray(transition_matrix, dtype=jnp.float32)
        self.TT_matrix = jnp.asarray(T.T)

        pi = jnp.ones(N * N, dtype=jnp.float32) / (N * N)
        for _ in range(2000):
            pi = self.TT_matrix @ pi
            pi = pi / pi.sum()
        self._stationary = pi

        self.belief = jnp.array(self._stationary)
        self._normalize()

        self.catches = 0
        self._first_turn = True
        self._search_log = []
        self._pending_conf = None

        self.evaluation_depths = []

    # ── public ────────────────────────────────────────────────────────────────

    def commentate(self):
        if DEBUG:
            summary = f"Caught the rat {self.catches} time{'s' if self.catches != 1 else ''}!"
            return summary + " LOG:" + json.dumps(self._search_log)
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

        # ── resolve last search ───────────────────────────────────────────────
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
        if not self._first_turn:
            self._transition()
            self._transition()
        self._first_turn = False

        self._update_noise(int(noise), board)
        self._update_distance(int(dist), wx, wy)

        # ── search info ───────────────────────────────────────────────────────
        best_idx = int(jnp.argmax(self.belief))
        best_p   = float(self.belief[best_idx])
        best_y, best_x = divmod(best_idx, self.N)
        search_ev   = best_p * enums.RAT_BONUS - (1 - best_p) * enums.RAT_PENALTY
        search_info = (best_p, best_x, best_y) if search_ev > 0 else None

        # ── iterative deepening with TT + killers + history ──────────────────
        tt        = {}
        killers   = [[] for _ in range(40)]
        history   = {}
        time_limit  = time.time() + 4
        best_move   = None
        leaf_count  = [0]
        evaluated_depth = 0

        for depth in range(1, 31):
            val, m = _minimax(
                board.get_copy(), depth, True, -math.inf, math.inf,
                time_limit, leaf_count, tt, killers, history,
                search_info=search_info,
            )
            if val is None:
                break  # timed out — keep best move from last complete depth
            if m is not None:
                best_move = m
            evaluated_depth = depth

        if best_move is None:
            non_search = [mv for mv in board.get_valid_moves()
                          if mv.move_type != MoveType.SEARCH]
            best_move = random.choice(non_search) if non_search else None

        if best_move is not None and best_move.move_type == MoveType.SEARCH:
            self._pending_conf = best_p
            if DEBUG:
                print(f"[Carrie] SEARCH ({best_x},{best_y}) P={best_p:.4f}")
        else:
            tt_hits = sum(1 for v in tt.values() if v[0] > 0)
            if DEBUG:
                print(f"[Carrie] depth={evaluated_depth} leaves={leaf_count[0]} "
                    f"tt_size={len(tt)} tt_useful={tt_hits}")
        self.evaluation_depths.append(evaluated_depth)
        return best_move

    # ── private (belief tracker — identical to Carrie) ────────────────────────

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
        self.belief = self.TT_matrix @ self.belief
        self._normalize()

    def _update_noise(self, noise_idx: int, board: board_mod.Board):
        N = self.N
        likelihood = jnp.array(
            [_NOISE_PROBS[int(board.get_cell((x, y)))][noise_idx]
             for y in range(N) for x in range(N)],
            dtype=jnp.float32,
        )
        self.belief = self.belief * likelihood
        self._normalize()

    def _update_distance(self, obs_dist: int, wx: int, wy: int):
        N = self.N
        xs = jnp.arange(N)
        ys = jnp.arange(N)
        dist_flat = (jnp.abs(xs[None, :] - wx) + jnp.abs(ys[:, None] - wy)).ravel()
        offsets   = obs_dist - dist_flat
        likelihood = jnp.zeros(N * N, dtype=jnp.float32)
        for i, off in enumerate(_DIST_OFFSETS):
            likelihood = likelihood.at[offsets == off].set(_DIST_PROBS[i])
        self.belief = self.belief * likelihood
        self._normalize()
