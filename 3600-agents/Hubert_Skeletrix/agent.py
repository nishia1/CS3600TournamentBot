from collections.abc import Callable
from typing import Tuple
import time
import numpy as np

from game import board, move, enums


N = enums.BOARD_SIZE * enums.BOARD_SIZE
DIRS = list(enums.Direction)
DELTA = {
    enums.Direction.UP: (0, -1),
    enums.Direction.DOWN: (0, 1),
    enums.Direction.LEFT: (-1, 0),
    enums.Direction.RIGHT: (1, 0),
}


def _idx(x, y):
    return y * enums.BOARD_SIZE + x


def _loc(i):
    return (i % enums.BOARD_SIZE, i // enums.BOARD_SIZE)


def _step(x, y, d):
    dx, dy = DELTA[d]
    return x + dx, y + dy


def _primed_run(bs, x, y, d):
    run = 0
    cx, cy = x, y
    while True:
        cx, cy = _step(cx, cy, d)
        if not bs.is_valid_cell((cx, cy)):
            break
        if bs.get_cell((cx, cy)) == enums.Cell.PRIMED:
            run += 1
        else:
            break
    return run


def _immediate_carpet_value(bs):
    x, y = bs.player_worker.get_location()
    best = 0
    for d in DIRS:
        r = _primed_run(bs, x, y, d)
        if r > 0:
            best = max(best, enums.CARPET_POINTS_TABLE.get(min(r, enums.BOARD_SIZE - 1), 0))
    return best


def _immediate_carpet_value_enemy(bs):
    ox, oy = bs.opponent_worker.get_location()
    px, py = bs.player_worker.get_location()
    best = 0
    for d in DIRS:
        run = 0
        cx, cy = ox, oy
        while True:
            cx, cy = _step(cx, cy, d)
            if not bs.is_valid_cell((cx, cy)):
                break
            if (cx, cy) == (px, py):
                break
            if bs.get_cell((cx, cy)) == enums.Cell.PRIMED:
                run += 1
            else:
                break
        if run > 0:
            best = max(best, enums.CARPET_POINTS_TABLE.get(min(run, enums.BOARD_SIZE - 1), 0))
    return best


def _open_neighbors(bs, pos):
    x, y = pos
    cnt = 0
    for d in DIRS:
        nx, ny = _step(x, y, d)
        if not bs.is_valid_cell((nx, ny)):
            continue
        if (nx, ny) == bs.opponent_worker.get_location():
            continue
        if bs.get_cell((nx, ny)) != enums.Cell.BLOCKED:
            cnt += 1
    return cnt


class _RatHMM:
    _NOISE = {
        enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
        enums.Cell.SPACE: (0.7, 0.15, 0.15),
        enums.Cell.PRIMED: (0.1, 0.8, 0.1),
        enums.Cell.CARPET: (0.1, 0.1, 0.8),
    }
    _OFF = (-1, 0, 1, 2)
    _OFF_P = (0.12, 0.70, 0.12, 0.06)

    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self.b = np.ones(N, dtype=np.float64) / N

    def predict(self):
        self.b = self.b @ self.T
        self.b = np.nan_to_num(self.b, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(self.b.sum())
        if s > 1e-12:
            self.b /= s
        else:
            self.b = np.ones(N, dtype=np.float64) / N

    def update(self, bs, noise_idx, dist):
        wx, wy = bs.player_worker.get_location()
        like = np.zeros(N, dtype=np.float64)
        for i in range(N):
            x, y = _loc(i)
            c = bs.get_cell((x, y))
            if c == enums.Cell.BLOCKED:
                continue
            pn = self._NOISE[c][noise_idx]
            td = abs(x - wx) + abs(y - wy)
            pd = 0.0
            for off, p in zip(self._OFF, self._OFF_P):
                if max(0, td + off) == dist:
                    pd += p
            like[i] = pn * pd

        s = float(like.sum())
        if s > 1e-12:
            self.b = self.b * like
            total = float(self.b.sum())
            if total > 1e-12 and np.isfinite(total):
                self.b /= total
            else:
                self.b = np.ones(N, dtype=np.float64) / N
        else:
            mask = np.array(
                [0.0 if bs.get_cell(_loc(i)) == enums.Cell.BLOCKED else 1.0 for i in range(N)],
                dtype=np.float64,
            )
            ms = float(mask.sum())
            self.b = (mask / ms) if ms > 1e-12 else (np.ones(N, dtype=np.float64) / N)


def _evaluate(bs, belief):
    my_pts = bs.player_worker.get_points()
    op_pts = bs.opponent_worker.get_points()
    turns_left = bs.player_worker.turns_left

    if turns_left > 24:
        p_w = 8.2
    elif turns_left > 12:
        p_w = 9.2
    else:
        p_w = 10.4

    score = p_w * (my_pts - op_pts)
    score += 2.4 * _immediate_carpet_value(bs)
    score -= 3.0 * _immediate_carpet_value_enemy(bs)

    px, py = bs.player_worker.get_location()
    ox, oy = bs.opponent_worker.get_location()
    score += 0.6 * _open_neighbors(bs, (px, py))
    score += 0.2 * (abs(px - ox) + abs(py - oy))

    moves = len(bs.get_valid_moves(exclude_search=True))
    score += 0.5 * moves
    if moves <= 2:
        score -= 4.0

    best_i = int(np.argmax(belief))
    bx, by = _loc(best_i)
    best_p = float(belief[best_i])
    d = abs(px - bx) + abs(py - by)
    score += 1.5 * best_p / (1.0 + d)

    return score


def _move_order_score(bs, m, belief):
    px, py = bs.player_worker.get_location()
    val = 0.0
    if m.move_type == enums.MoveType.CARPET:
        val += 18.0 + 2.0 * m.roll_length
    elif m.move_type == enums.MoveType.PRIME:
        nx, ny = _step(px, py, m.direction)
        val += 2.0
        if bs.is_valid_cell((nx, ny)):
            val += 1.5 * float(belief[_idx(nx, ny)])
    else:
        nx, ny = _step(px, py, m.direction)
        val += 1.0
        if bs.is_valid_cell((nx, ny)):
            val += 1.2 * float(belief[_idx(nx, ny)])
    return val


def _ordered_moves(bs, belief, max_branch):
    moves = bs.get_valid_moves(exclude_search=True)
    if not moves:
        return []
    ordered = sorted(moves, key=lambda mv: _move_order_score(bs, mv, belief), reverse=True)
    return ordered[:max_branch]


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            T = np.full((N, N), 1.0 / N, dtype=np.float64)
        self.hmm = _RatHMM(T)
        self.search_cd = 0

    def _negamax(self, bs, belief, depth, alpha, beta, deadline, max_branch):
        if bs.is_game_over() or depth == 0 or time.perf_counter() >= deadline:
            return _evaluate(bs, belief)

        moves = _ordered_moves(bs, belief, max_branch)
        if not moves:
            return _evaluate(bs, belief)

        next_belief = belief @ self.hmm.T
        best = -1e18
        for m in moves:
            child = bs.forecast_move(m, check_ok=True)
            if child is None:
                continue
            # forecast_move does not swap perspective; do that explicitly for adversarial recursion.
            child.reverse_perspective()
            val = -self._negamax(child, next_belief, depth - 1, -beta, -alpha, deadline, max_branch)
            if val > best:
                best = val
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _best_move_ab(self, bs, belief, time_left_seconds):
        turns_left = bs.player_worker.turns_left
        depth = 4 if turns_left > 10 else 3
        max_branch = 9 if turns_left > 8 else 11

        if time_left_seconds <= 20:
            depth = 2
            max_branch = 8
        elif time_left_seconds > 70 and turns_left > 12:
            depth = 5
            max_branch = 7

        budget = 0.18 if time_left_seconds > 35 else 0.10
        deadline = time.perf_counter() + budget

        moves = _ordered_moves(bs, belief, max_branch)
        if not moves:
            return None

        best_move = None
        best_val = -1e18
        alpha, beta = -1e18, 1e18
        next_belief = belief @ self.hmm.T

        for m in moves:
            if time.perf_counter() >= deadline:
                break
            child = bs.forecast_move(m, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()
            val = -self._negamax(child, next_belief, depth - 1, -beta, -alpha, deadline, max_branch)
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

        return best_move
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "Hubert_Skeletrix: HMM + alpha-beta"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        noise_idx, dist = sensor_data
        noise_idx = int(noise_idx)

        if callable(time_left):
            try:
                time_left_seconds = float(time_left())
            except Exception:
                time_left_seconds = 0.0
        else:
            try:
                time_left_seconds = float(time_left)
            except Exception:
                time_left_seconds = 0.0

        if self.search_cd > 0:
            self.search_cd -= 1

        # HMM belief update.
        self.hmm.predict()
        self.hmm.update(board, noise_idx, dist)
        belief = np.nan_to_num(self.hmm.b.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        s = float(belief.sum())
        if s > 1e-12:
            belief /= s
        else:
            belief = np.ones(N, dtype=np.float64) / N

        # Immediate high-value carpet takes precedence.
        px, py = board.player_worker.get_location()
        best_carpet = None
        best_pts = -1
        for d in DIRS:
            r = _primed_run(board, px, py, d)
            if r <= 0:
                continue
            pts = enums.CARPET_POINTS_TABLE.get(min(r, enums.BOARD_SIZE - 1), 0)
            if pts > best_pts:
                best_pts = pts
                best_carpet = move.Move.carpet(d, r)
        if best_carpet is not None and board.is_valid_move(best_carpet):
            if best_pts >= 4 or board.player_worker.turns_left <= 14:
                return best_carpet

        # Rare search when belief is very concentrated and no urgent carpet.
        if self.search_cd == 0 and time_left_seconds > 18 and board.player_worker.turns_left > 14:
            best_i = int(np.argmax(belief))
            best_p = float(belief[best_i])
            if best_p >= 0.55 and best_pts < 3:
                self.search_cd = 3
                return move.Move.search(_loc(best_i))

        # Main alpha-beta action.
        best = self._best_move_ab(board, belief, time_left_seconds)
        if best is not None and board.is_valid_move(best):
            return best

        # Safe fallback.
        moves = board.get_valid_moves(exclude_search=True)
        if moves:
            return max(moves, key=lambda mv: _move_order_score(board, mv, belief))
        return move.Move.search(_loc(int(np.argmax(belief))))
