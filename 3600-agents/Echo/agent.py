from collections import deque
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np

from game import move, enums
from game.enums import Cell, Direction, Noise, BOARD_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_P = {
    Cell.BLOCKED: (0.50, 0.30, 0.20),
    Cell.SPACE:   (0.70, 0.15, 0.15),
    Cell.PRIMED:  (0.10, 0.80, 0.10),
    Cell.CARPET:  (0.10, 0.10, 0.80),
}

_DIST_P = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

_OPP_DIR = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _idx(loc): return loc[1] * BOARD_SIZE + loc[0]
def _loc(i): return (i % BOARD_SIZE, i // BOARD_SIZE)

def _step(loc, d): return enums.loc_after_direction(loc, d)

def _dist(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])


def _cluster_score(belief, idx):
    cx, cy = _loc(idx)
    s = 0.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < 8 and 0 <= y < 8:
                s += belief[_idx((x, y))]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# HMM
# ─────────────────────────────────────────────────────────────────────────────

class _RatHMM:
    def __init__(self, T):
        self.N = BOARD_SIZE * BOARD_SIZE
        self.T = np.asarray(T, dtype=np.float64) if T is not None else np.full((self.N, self.N), 1/self.N)

        b = np.zeros(self.N)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        self.prior = b / b.sum()
        self.b = self.prior.copy()

        self.lx = np.array([i % 8 for i in range(self.N)])
        self.ly = np.array([i // 8 for i in range(self.N)])

    def reset(self):
        self.b = self.prior.copy()

    def miss(self, loc):
        self.b[_idx(loc)] = 0
        self._norm()

    def predict(self):
        self.b = self.b @ self.T

    def update(self, bs, noise, dist_obs):
        wx, wy = bs.player_worker.get_location()
        nidx = int(noise)

        noise_lk = np.array([
            _NOISE_P.get(bs.get_cell(_loc(i)), _NOISE_P[Cell.SPACE])[nidx]
            for i in range(self.N)
        ])

        d = np.abs(self.lx - wx) + np.abs(self.ly - wy)
        dist_lk = np.zeros(self.N)

        for off, p in _DIST_P.items():
            dist_lk[d + off == dist_obs] += p

        self.b *= noise_lk * dist_lk
        self._norm()

    def best_search(self):
        i = int(np.argmax(self.b))
        p = self.b[i]
        ev = 6*p - 2
        return (_loc(i), ev) if ev > 0 else None

    def _norm(self):
        s = self.b.sum()
        self.b = self.prior.copy() if s < 1e-12 else self.b / s


# ─────────────────────────────────────────────────────────────────────────────
# Move scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_move(m, pos, bs, belief):

    mtype = m.move_type
    ox, oy = bs.opponent_worker.get_location()

    if mtype == move.MoveType.SEARCH:
        p = belief[_idx(m.search_loc)]
        if p < 0.30:
            return -10
        return 10 * p

    if mtype == move.MoveType.CARPET:
        k = getattr(m, "roll_length", 1)
        base = k * 2

        opp_bonus = abs(pos[0]-ox) + abs(pos[1]-oy)

        if base < 4:
            return -3

        return base + opp_bonus * 0.2

    d = m.direction
    dx, dy = enums.loc_after_direction((0,0), d)

    x, y = pos
    nx, ny = x + dx, y + dy

    target = np.argmax(belief)
    tx, ty = _loc(target)

    toward = abs(x-tx)+abs(y-ty) - abs(nx-tx)-abs(ny-ty)

    opp_bonus = abs(nx-ox) + abs(ny-oy)

    line_bonus = 0
    cx, cy = nx, ny
    for _ in range(4):
        cx += dx
        cy += dy
        if 0 <= cx < 8 and 0 <= cy < 8:
            line_bonus += 1

    if mtype == move.MoveType.PRIME:
        return line_bonus*2.5 + toward*2 + opp_bonus*0.3 + 1

    center = -(abs(nx-3.5)+abs(ny-3.5))*0.3
    cluster = _cluster_score(belief, target)

    return center + cluster*3 + toward*2.5 + opp_bonus*0.2


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left=None):
        self.hmm = _RatHMM(transition_matrix)
        self.plan = None
        self.search_cd = 0

    def play(self, bs, sensor, time_left):

        noise, dist = sensor
        t = time_left()

        if self.search_cd > 0:
            self.search_cd -= 1

        my_loc, hit = bs.player_search
        if my_loc:
            self.search_cd = 2
            if hit:
                self.hmm.reset()
            else:
                self.hmm.miss(my_loc)

        opp_loc, opp_hit = bs.opponent_search
        if opp_loc:
            if opp_hit:
                self.hmm.reset()
            else:
                self.hmm.miss(opp_loc)

        self.hmm.predict()
        self.hmm.update(bs, noise, dist)

        c = None
        for d in Direction:
            nxt = _step(bs.player_worker.get_location(), d)
            if bs.is_valid_cell(nxt):
                c = move.Move.carpet(d, 1)
                break

        # SEARCH DECISION
        if self.search_cd == 0 and t > 1.0:
            sr = self.hmm.best_search()
            if sr:
                loc, ev = sr
                if ev > 2:
                    return move.Move.search(loc)

        # MOVE
        valid = bs.get_valid_moves(exclude_search=True)
        if not valid:
            return move.Move.search((0,0))

        return max(valid, key=lambda m: _score_move(
            m,
            bs.player_worker.get_location(),
            bs,
            self.hmm.b
        ))