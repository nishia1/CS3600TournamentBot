from collections.abc import Callable
import numpy as np
from game.move import Move
from game.enums import Cell, Direction, MoveType, BOARD_SIZE, CARPET_POINTS_TABLE

N = BOARD_SIZE * BOARD_SIZE
DIRS = list(Direction)
_DELTA = {Direction.UP:(0,-1), Direction.DOWN:(0,1), Direction.LEFT:(-1,0), Direction.RIGHT:(1,0)}

def _idx(x, y): return y * BOARD_SIZE + x
def _loc(i): return (i % BOARD_SIZE, i // BOARD_SIZE)
def _step(x, y, d):
    dx, dy = _DELTA[d]
    return x+dx, y+dy


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
                reported = max(0, true_d + off)
                if reported == dist:
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


def _run_length(bs, x, y, d):
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
    best_move = None
    best_score = -999
    for d in DIRS:
        r = _run_length(bs, x, y, d)
        if r >= 1:
            score = CARPET_POINTS_TABLE.get(r, -99)
            if score > best_score:
                best_score = score
                best_move = Move.carpet(d, r)
    return best_move if best_score > 0 else None


def _score_carpet_potential(bs, x, y):
    best = 0
    for d in DIRS:
        pr = _run_length(bs, x, y, d)
        if pr >= 2:
            best = max(best, CARPET_POINTS_TABLE.get(pr, 0))
        sr = _space_run(bs, x, y, d)
        if sr >= 2:
            best = max(best, CARPET_POINTS_TABLE.get(min(sr, 7), 0) * 0.4)
    return best


def _bfs_distances(bs, start):
    ox, oy = bs.opponent_worker.get_location()
    dist = {start: 0}
    queue = [start]
    head = 0
    while head < len(queue):
        cx, cy = queue[head]; head += 1
        for d in DIRS:
            nx, ny = _step(cx, cy, d)
            if not bs.is_valid_cell((nx, ny)):
                continue
            if (nx, ny) in dist:
                continue
            if (nx, ny) == (ox, oy):
                continue
            cell = bs.get_cell((nx, ny))
            if cell == Cell.BLOCKED or cell == Cell.PRIMED:
                continue
            dist[(nx, ny)] = dist[(cx, cy)] + 1
            queue.append((nx, ny))
    return dist


def _next_step_toward(bs, start, target):
    if start == target:
        return None
    ox, oy = bs.opponent_worker.get_location()
    prev = {start: None}
    queue = [start]
    head = 0
    found = False
    while head < len(queue):
        cx, cy = queue[head]; head += 1
        if (cx, cy) == target:
            found = True
            break
        for d in DIRS:
            nx, ny = _step(cx, cy, d)
            if not bs.is_valid_cell((nx, ny)):
                continue
            if (nx, ny) in prev:
                continue
            if (nx, ny) == (ox, oy):
                continue
            cell = bs.get_cell((nx, ny))
            if cell == Cell.BLOCKED or cell == Cell.PRIMED:
                continue
            prev[(nx, ny)] = ((cx, cy), d)
            queue.append((nx, ny))
    if not found:
        return None
    node = target
    while prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1]


def _evaluate_positions(bs, belief, bfs_dist):
    scores = {}
    for (x, y), d in bfs_dist.items():
        if d == 0:
            continue
        cell = bs.get_cell((x, y))
        if cell == Cell.BLOCKED or cell == Cell.PRIMED:
            continue
        carpet_val = _score_carpet_potential(bs, x, y)
        rat_prox = 0.0
        for i in range(N):
            p = belief[i]
            if p < 1e-5:
                continue
            rx, ry = _loc(i)
            md = abs(x - rx) + abs(y - ry)
            rat_prox += p / (1 + md)
        dist_penalty = d * 1.5
        scores[(x, y)] = carpet_val * 3.0 + rat_prox * 4.0 - dist_penalty
    return scores


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            T = np.full((N, N), 1.0 / N)
        self.hmm = _RatHMM(T)
        self.search_cd = 0
        self.prime_dir = None
        self.prime_remaining = 0

    def commentate(self):
        return "priming and carpeting"

    def play(self, bs, sensor, time_left):
        noise_idx, dist = sensor
        noise_idx = int(noise_idx)

        if self.search_cd > 0:
            self.search_cd -= 1

        self.hmm.predict()
        self.hmm.update(bs, noise_idx, dist)
        belief = self.hmm.b

        px, py = bs.player_worker.get_location()
        current_cell = bs.get_cell((px, py))

        # 1. CARPET if profitable
        carpet = _best_carpet_move(bs)
        if carpet is not None:
            return carpet

        # 2. SEARCH if EV > 0
        if self.search_cd == 0:
            best_i = int(np.argmax(belief))
            best_p = belief[best_i]
            if 6.0 * best_p - 2.0 > 0:
                self.search_cd = 3
                return Move.search(_loc(best_i))

        # 3. PRIME if on SPACE
        if current_cell == Cell.SPACE:
            # Continue committed prime run
            if self.prime_dir is not None and self.prime_remaining > 0:
                candidate = Move.prime(self.prime_dir)
                if bs.is_valid_move(candidate):
                    self.prime_remaining -= 1
                    if self.prime_remaining == 0:
                        self.prime_dir = None
                    return candidate
                else:
                    self.prime_dir = None
                    self.prime_remaining = 0

            # Pick new prime direction
            best_d, best_r = None, 0
            for d in DIRS:
                r = _space_run(bs, px, py, d)
                if r > best_r:
                    best_r = r
                    best_d = d
            if best_d is not None and best_r >= 2:
                candidate = Move.prime(best_d)
                if bs.is_valid_move(candidate):
                    self.prime_dir = best_d
                    self.prime_remaining = best_r - 1
                    return candidate

        # 4. MOVE toward best destination
        bfs = _bfs_distances(bs, (px, py))
        scores = _evaluate_positions(bs, belief, bfs)

        if scores:
            target = max(scores, key=scores.__getitem__)
            d = _next_step_toward(bs, (px, py), target)
            if d is not None:
                nx, ny = _step(px, py, d)
                next_cell = bs.get_cell((nx, ny))
                # Prime on the way if it starts a good run
                if next_cell == Cell.SPACE and current_cell == Cell.SPACE:
                    ahead_r = _space_run(bs, nx, ny, d)
                    if ahead_r >= 2:
                        candidate = Move.prime(d)
                        if bs.is_valid_move(candidate):
                            self.prime_dir = d
                            self.prime_remaining = ahead_r
                            return candidate
                plain = Move.plain(d)
                if bs.is_valid_move(plain):
                    return plain

        # 5. Fallback: any valid move
        moves = bs.get_valid_moves(exclude_search=True)
        if moves:
            primes = [m for m in moves if m.move_type == MoveType.PRIME]
            if primes:
                return primes[0]
            return moves[0]

        # Absolute fallback
        best_i = int(np.argmax(belief))
        return Move.search(_loc(best_i))