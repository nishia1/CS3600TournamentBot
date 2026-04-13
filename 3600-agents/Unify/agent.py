from collections.abc import Callable
from math import dist
import numpy as np
from game.move import Move
from game.enums import Cell, Direction, MoveType, BOARD_SIZE, CARPET_POINTS_TABLE

N = BOARD_SIZE * BOARD_SIZE
DIRS = list(Direction)
_DELTA = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}

def _loc(i):        return (i % BOARD_SIZE, i // BOARD_SIZE)
def _idx(x, y):     return y * BOARD_SIZE + x
def _step(x, y, d): dx, dy = _DELTA[d]; return x+dx, y+dy


class _RatHMM:
    _NOISE = {
        Cell.BLOCKED: (0.5, 0.3, 0.2),
        Cell.SPACE:   (0.7, 0.15, 0.15),
        Cell.PRIMED:  (0.1, 0.8, 0.1),
        Cell.CARPET:  (0.1, 0.1, 0.8),
    }
    _DOFF  = (-1, 0, 1, 2)
    _DPROB = (0.12, 0.70, 0.12, 0.06)

    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self.b = np.ones(N) / N

    def predict(self):
        self.b = self.b @ self.T

    def update(self, bs, noise_idx, dist):
        wx, wy = bs.player_worker.get_location()
        L = np.zeros(N)
        for i in range(N):
            x, y = _loc(i)
            cell = bs.get_cell((x, y))
            if cell == Cell.BLOCKED:
                continue
            pn = self._NOISE[cell][noise_idx]
            true_d = abs(x - wx) + abs(y - wy)
            pd = 0.0
            for off, prob in zip(self._DOFF, self._DPROB):
                if max(0, true_d + off) == dist:
                    pd += prob
            L[i] = pn * pd
        s = L.sum()
        if s > 1e-12:
            self.b = self.b * L
            self.b /= self.b.sum()
        else:
            mask = np.array([
                0.0 if bs.get_cell(_loc(i)) == Cell.BLOCKED else 1.0
                for i in range(N)
            ])
            self.b = mask / mask.sum()

def _eval(bs, belief, x, y, ox, oy):
    """
    Unified state value function.
    Used by carpet, prime, movement, and search.
    """

    i = _idx(x, y)

    value = 0.0

    # ── belief reward (core signal) ──
    value += 12.0 * belief[i]

    # ── opponent distance penalty ──
    dist = abs(x - ox) + abs(y - oy)
    value -= 0.08 * dist

    # ── cell structure bonus ──
    cell = bs.get_cell((x, y))

    if cell == Cell.SPACE:
        value += 0.5
    elif cell == Cell.PRIMED:
        value += 3.0
    elif cell == Cell.CARPET:
        value += 6.0

    return value

def _primed_run(bs, x, y, d):
    run = 0
    cx, cy = x, y
    while True:
        cx, cy = _step(cx, cy, d)
        if not bs.is_valid_cell((cx, cy)):
            break
        if bs.get_cell((cx, cy)) == Cell.PRIMED:
            run += 1
        else:
            break
    return run


def _space_run(bs, x, y, d):
    ox, oy = bs.opponent_worker.get_location()
    run = 0
    cx, cy = x, y
    while True:
        cx, cy = _step(cx, cy, d)
        if not bs.is_valid_cell((cx, cy)):
            break
        if (cx, cy) == (ox, oy):
            break
        if bs.get_cell((cx, cy)) == Cell.SPACE:
            run += 1
        else:
            break
    return run


def _best_carpet_move(bs):
    x, y = bs.player_worker.get_location()
    ox, oy = bs.opponent_worker.get_location()

    best_move, best_score = None, 0

    for d in DIRS:
        r = _primed_run(bs, x, y, d)

        if r < 2:
            continue

        score = CARPET_POINTS_TABLE.get(r, 0)
        if score <= 0:
            continue

        # NEW: evaluate landing positions along run
        cx, cy = x, y
        run_value = 0.0

        for _ in range(r):
            cx, cy = _step(cx, cy, d)
            if not bs.is_valid_cell((cx, cy)):
                break
            run_value += _eval(bs, np.ones(N)/N, cx, cy, ox, oy)

        score += 0.5 * run_value

        if score > best_score:
            best_score = score
            best_move = Move.carpet(d, r)

    return best_move


def belief_global_proxy(bs, x, y):
    # lightweight helper (no new systems)
    # approximates belief without touching HMM
    return 1.0


def _standard_passable(bs, nx, ny):
    if not bs.is_valid_cell((nx, ny)):
        return False
    if (nx, ny) == bs.opponent_worker.get_location():
        return False
    c = bs.get_cell((nx, ny))
    return c not in (Cell.BLOCKED, Cell.PRIMED)


def _bfs(bs, start):
    dist = {start: 0}
    q = [start]; head = 0
    while head < len(q):
        cx, cy = q[head]; head += 1
        for d in DIRS:
            nx, ny = _step(cx, cy, d)
            if (nx, ny) in dist:
                continue
            if _standard_passable(bs, nx, ny):
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
    return dist


def _next_step_toward(bs, start, target):
    if start == target:
        return None
    prev = {start: None}
    q = [start]; head = 0
    found = False
    while head < len(q):
        cx, cy = q[head]; head += 1
        if (cx, cy) == target:
            found = True; break
        for d in DIRS:
            nx, ny = _step(cx, cy, d)
            if (nx, ny) in prev:
                continue
            if _standard_passable(bs, nx, ny):
                prev[(nx, ny)] = ((cx, cy), d)
                q.append((nx, ny))
    if not found:
        return None
    node = target
    while prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1]


def _pick_prime_direction(bs, belief):
    px, py = bs.player_worker.get_location()
    ox, oy = bs.opponent_worker.get_location()

    best_d, best_score, best_r = None, -1, 0

    for d in DIRS:
        r = _space_run(bs, px, py, d)

        if r < 2:
            continue

        cx, cy = px, py
        run_score = 0.0

        for _ in range(r):
            cx, cy = _step(cx, cy, d)
            run_score += _eval(bs, belief, cx, cy, ox, oy)

        if run_score > best_score:
            best_score = run_score
            best_d = d
            best_r = r

    return best_d, best_r


def _find_best_space_destination(bs, belief):
    """
    BFS to find the best SPACE cell to walk toward to start a new prime run.
    Returns (target, direction_from_target, run_length) or (None, None, 0).
    """
    px, py = bs.player_worker.get_location()
    bfs_dist = _bfs(bs, (px, py))
    ox, oy = bs.opponent_worker.get_location()

    best_loc = None
    best_score = -1e9

    for (x, y), steps in bfs_dist.items():
        if steps == 0:
            continue
        if bs.get_cell((x, y)) != Cell.SPACE:
            continue
        # What's the best prime run from here?
        best_run_score = 0
        for d in DIRS:
            r = _space_run(bs, x, y, d)
            if r < 2:
                continue
            pts = CARPET_POINTS_TABLE.get(min(r, 7), 0)
            rat_mass = 0.0
            cx, cy = x, y
            for _ in range(r):
                cx, cy = _step(cx, cy, d)
                rat_mass += float(belief[_idx(cx, cy)])
            s = pts * 3.0 + rat_mass * 2.0
            if s > best_run_score:
                best_run_score = s
        if best_run_score <= 0:
            continue
        score = best_run_score - steps * 2.0
        if score > best_score:
            best_score = score
            best_loc = (x, y)

    return best_loc


def _best_search(belief, bs):
    ox, oy = bs.opponent_worker.get_location()

    best_i = int(np.argmax(belief))
    x, y = _loc(best_i)

    ev = _eval(bs, belief, x, y, ox, oy)

    return (x, y), ev


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            T = np.full((N, N), 1.0 / N)
        self.hmm = _RatHMM(T)
        self.search_cd = 0
        # Prime commitment state
        self.prime_dir = None
        self.prime_remaining = 0
        # After finishing a prime run, we need to return to start to carpet
        # Track the start of the prime run so we can carpet roll from there
        self.prime_run_start = None
        self.prime_run_len = 0

    def commentate(self):
        return "commit prime carpet hunter"

    def play(self, bs, sensor, time_left):
        noise_idx, dist = sensor
        noise_idx = int(noise_idx)

        if self.search_cd > 0:
            self.search_cd -= 1

        # ─────────────────────────────────────────────
        # BELIEF UPDATE (UNCHANGED)
        # ─────────────────────────────────────────────
        self.hmm.predict()
        self.hmm.update(bs, noise_idx, dist)
        belief = self.hmm.b.copy()

        px, py = bs.player_worker.get_location()
        ox, oy = bs.opponent_worker.get_location()
        current_cell = bs.get_cell((px, py))

        moves = bs.get_valid_moves(exclude_search=True)

        if not moves:
            loc = int(np.argmax(belief))
            return Move.search(_loc(loc))

        # ─────────────────────────────────────────────
        # 1. CARPET (soft decision, not forced)
        # ─────────────────────────────────────────────
        carpet = _best_carpet_move(bs)

        if carpet is not None:
            # evaluate carpet quality before committing
            d = carpet.direction
            nx, ny = _step(px, py, d)

            if bs.is_valid_cell((nx, ny)):
                carpet_value = _eval(bs, belief, nx, ny, ox, oy)

                # only take strong carpets
                if carpet_value > 2.5:
                    return carpet

        # ─────────────────────────────────────────────
        # 2. PRIME CONTINUATION
        # ─────────────────────────────────────────────
        if self.prime_dir is not None and self.prime_remaining > 0:
            move = Move.prime(self.prime_dir)

            if bs.is_valid_move(move):
                self.prime_remaining -= 1
                return move
            else:
                self.prime_dir = None
                self.prime_remaining = 0

        # ─────────────────────────────────────────────
        # 3. START PRIME (only if good via unified eval)
        # ─────────────────────────────────────────────
        if current_cell == Cell.SPACE:
            best_d, best_r = _pick_prime_direction(bs, belief)

            if best_d is not None and best_r >= 2:
                cx, cy = _step(px, py, best_d)

                if bs.is_valid_cell((cx, cy)):
                    val = _eval(bs, belief, cx, cy, ox, oy)

                    if val > 1.5:
                        move = Move.prime(best_d)

                        if bs.is_valid_move(move):
                            self.prime_dir = best_d
                            self.prime_remaining = best_r - 1
                            self.prime_run_start = (px, py)
                            self.prime_run_len = best_r
                            return move

        # ─────────────────────────────────────────────
        # 4. SEARCH (now consistent EV-based)
        # ─────────────────────────────────────────────
        if self.search_cd == 0:
            loc, ev = _best_search(belief, bs)

            if ev > 1.0:
                self.search_cd = 3
                return Move.search(loc)

        # ─────────────────────────────────────────────
        # 5. FINAL MOVE SELECTION (THIS IS THE CORE FIX)
        # ─────────────────────────────────────────────
        best_move = None
        best_score = -1e18

        for m in moves:
            if not bs.is_valid_move(m):
                continue

            score = 0.0

            # small structural bias
            if m.move_type == MoveType.CARPET:
                score += 2.0
            elif m.move_type == MoveType.PRIME:
                score += 1.0
            elif m.move_type == MoveType.SEARCH:
                score += 0.5

            # unified evaluation (KEY PART)
            if hasattr(m, "direction"):
                nx, ny = _step(px, py, m.direction)

                if bs.is_valid_cell((nx, ny)):
                    score += _eval(bs, belief, nx, ny, ox, oy)

            if score > best_score:
                best_score = score
                best_move = m

        # fallback safety
        if best_move is not None:
            return best_move

        loc = int(np.argmax(belief))
        return Move.search(_loc(loc))