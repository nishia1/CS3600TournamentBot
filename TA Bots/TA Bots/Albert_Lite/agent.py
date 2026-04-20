"""
Albert Lite: pure minimax — no rat searching.

Each turn:
  1. Exact Bayesian belief update (two transitions + noise + distance).
  2. Run iterative-deepening alpha-beta minimax.
"""

import math
import random
import time
from collections.abc import Callable
from typing import List, Tuple

import jax.numpy as jnp

from game import *
from game import board as board_mod, enums
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

# Standard alpha-beta minimax. Unlike Albert, there is no search info.
def _minimax(
    board: board_mod.Board,
    depth: int,
    maximizing: bool,
    alpha: float = -math.inf,
    beta: float = math.inf,
    time_limit: float = None,
    leaf_count: List[int] = None,
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

        return max_val, best_move
    else:
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
        self._first_turn = True
        self.evaluation_depths = []

    def commentate(self):
        return str(round(sum(self.evaluation_depths) / len(self.evaluation_depths), 2)) if self.evaluation_depths else "0"

    def play(
        self,
        board: board_mod.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):

        # ── iterative-deepening minimax ───────────────────────────────────────
        time_limit = time.time() + 4
        best_move = None
        leaf_count = [0]
        evaluated_depth = 0

        for depth in range(1, 31):
            _, mv = _minimax(board.get_copy(), depth, True,
                             time_limit=time_limit, leaf_count=leaf_count)
            if mv is not None:
                best_move = mv
                evaluated_depth = depth
            else:
                break

        if best_move is None:
            valid_moves = board.get_valid_moves()
            best_move = random.choice(valid_moves) if valid_moves else None

        if DEBUG:
            print(f"[Albert_Lite] MINIMAX depth={evaluated_depth} leaves={leaf_count[0]}")
        self.evaluation_depths.append(evaluated_depth)
        return best_move

    