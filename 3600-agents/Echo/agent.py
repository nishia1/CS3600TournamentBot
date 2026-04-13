from collections.abc import Callable
import numpy as np

from game import move, enums
from game.enums import Cell, Direction, BOARD_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _idx(loc): return loc[1] * BOARD_SIZE + loc[0]
def _loc(i): return (i % BOARD_SIZE, i // BOARD_SIZE)
def _step(loc, d): return enums.loc_after_direction(loc, d)


# ─────────────────────────────────────────────────────────────────────────────
# HMM
# ─────────────────────────────────────────────────────────────────────────────

class _RatHMM:
    def __init__(self, T):
        self.N = BOARD_SIZE * BOARD_SIZE
        self.T = np.asarray(T, dtype=np.float64) if T is not None else np.full((self.N, self.N), 1/self.N)
        self.b = np.ones(self.N) / self.N

    def predict(self):
        self.b = self.b @ self.T

    def update(self, bs, noise, dist):
        wx, wy = bs.player_worker.get_location()

        likelihood = np.ones(self.N)

        for i in range(self.N):
            x, y = _loc(i)

            cell = bs.get_cell((x, y))

            if cell == Cell.BLOCKED:
                p_noise = [0.5, 0.3, 0.2][noise]
            elif cell == Cell.PRIMED:
                p_noise = [0.1, 0.8, 0.1][noise]
            elif cell == Cell.CARPET:
                p_noise = [0.1, 0.1, 0.8][noise]
            else:
                p_noise = [0.7, 0.15, 0.15][noise]

            true_d = abs(x - wx) + abs(y - wy)
            p_dist = 0.7 if dist == true_d else 0.1

            likelihood[i] = p_noise * p_dist

        self.b *= likelihood
        s = self.b.sum()
        if s > 1e-9:
            self.b /= s


# ─────────────────────────────────────────────────────────────────────────────
# Carpet selection (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def _best_carpet(bs):
    loc = bs.player_worker.get_location()
    best = None
    best_score = 0

    for d in Direction:
        run = 0
        cur = loc

        for _ in range(7):
            cur = _step(cur, d)
            if not bs.is_valid_cell(cur):
                break
            if bs.get_cell(cur) == Cell.PRIMED:
                run += 1
            else:
                break

        if run >= 2:
            score = run * run  # strong preference for longer carpets
            if score > best_score:
                best_score = score
                best = move.Move.carpet(d, run)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Move scoring (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def _score_move(m, pos, bs, belief):

    mtype = m.move_type
    ox, oy = bs.opponent_worker.get_location()

    # SEARCH
    if mtype == move.MoveType.SEARCH:
        p = belief[_idx(m.search_loc)]
        if p < 0.30:
            return -10
        return 10 * p

    # CARPET
    if mtype == move.MoveType.CARPET:
        k = getattr(m, "roll_length", 1)
        if k < 2:
            return -5
        return k * k  # strong scaling reward

    # Movement
    d = m.direction
    nx, ny = _step(pos, d)

    # Tile scoring (CRITICAL FIX)
    next_cell = bs.get_cell((nx, ny))
    tile_score = 0

    if next_cell == Cell.CARPET:
        tile_score -= 6   # avoid carpet loops

    elif next_cell == Cell.PRIMED:
        tile_score -= 2   # avoid wasting primed tiles

    elif next_cell == Cell.SPACE:
        tile_score += 4   # expand territory

    # Rat targeting
    target = np.argmax(belief)
    tx, ty = _loc(target)

    toward = abs(pos[0]-tx)+abs(pos[1]-ty) - abs(nx-tx)-abs(ny-ty)

    # Opponent avoidance
    opp_bonus = abs(nx-ox) + abs(ny-oy)

    # Line bonus (for future carpets)
    dx = nx - pos[0]
    dy = ny - pos[1]

    line_bonus = 0
    cx, cy = nx, ny
    for _ in range(4):
        cx += dx
        cy += dy
        if 0 <= cx < 8 and 0 <= cy < 8:
            if bs.get_cell((cx, cy)) == Cell.SPACE:
                line_bonus += 1
            else:
                break

    # PRIME
    if mtype == move.MoveType.PRIME:
        return (
            line_bonus * 2.5
            + toward * 2
            + opp_bonus * 0.3
            + tile_score
            + 1
        )

    # PLAIN
    center = -(abs(nx-3.5)+abs(ny-3.5))*0.3

    return (
        center
        + toward * 2.5
        + opp_bonus * 0.2
        + tile_score
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.hmm = _RatHMM(transition_matrix)
        self.search_cd = 0

    def play(self, bs, sensor, time_left):

        noise, dist = sensor

        # cooldown
        if self.search_cd > 0:
            self.search_cd -= 1

        # update HMM
        self.hmm.predict()
        self.hmm.update(bs, noise, dist)

        belief = self.hmm.b

        # ─────────────────────────────
        # 1. TAKE GOOD CARPET
        # ─────────────────────────────
        c = _best_carpet(bs)
        if c:
            return c

        # ─────────────────────────────
        # 2. SEARCH (if strong)
        # ─────────────────────────────
        if self.search_cd == 0:
            best_idx = int(np.argmax(belief))
            p = belief[best_idx]

            if p > 0.45:
                self.search_cd = 2
                return move.Move.search(_loc(best_idx))

        # ─────────────────────────────
        # 3. NORMAL MOVE
        # ─────────────────────────────
        moves = bs.get_valid_moves(exclude_search=True)

        if not moves:
            return move.Move.search((0, 0))

        return max(
            moves,
            key=lambda m: _score_move(
                m,
                bs.player_worker.get_location(),
                bs,
                belief
            )
        )