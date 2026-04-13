from collections.abc import Callable
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
    best_move, best_score = None, 0
    for d in DIRS:
        r = _primed_run(bs, x, y, d)
        if r >= 1:
            score = CARPET_POINTS_TABLE.get(r, -99)
            if score > best_score:
                best_score = score
                best_move = Move.carpet(d, r)
    return best_move


def _carpet_points(roll_len):
    capped = min(max(1, int(roll_len)), BOARD_SIZE - 1)
    return CARPET_POINTS_TABLE.get(capped, 0)


def _distance_multiplier(steps):
    # Multiply value by a distance term so closer opportunities are preferred.
    return 1.0 / (steps + 1.0)


def _all_carpet_options(bs, start):
    dist = _bfs(bs, start)
    if start not in dist:
        dist[start] = 0

    options = []
    for (x, y), steps in dist.items():
        for d in DIRS:
            roll = _primed_run(bs, x, y, d)
            if roll < 1:
                continue
            points = _carpet_points(roll)
            options.append({
                "kind": "carpet",
                "cell": (x, y),
                "direction": d,
                "roll": roll,
                "steps": steps,
                "points": points,
            })
    return options


def _all_prime_options(bs, start, belief):
    dist = _bfs(bs, start)
    if start not in dist:
        dist[start] = 0

    options = []
    for (x, y), steps in dist.items():
        if bs.get_cell((x, y)) != Cell.SPACE:
            continue
        for d in DIRS:
            run = _space_run(bs, x, y, d)
            if run < 2:
                continue

            rat_mass = 0.0
            cx, cy = x, y
            for _ in range(run):
                cx, cy = _step(cx, cy, d)
                rat_mass += float(belief[_idx(cx, cy)])

            points = _carpet_points(run)
            options.append({
                "kind": "prime",
                "cell": (x, y),
                "direction": d,
                "run": run,
                "steps": steps,
                "points": points,
                "rat_mass": rat_mass,
            })
    return options


def _best_distance_weighted_option(carpet_options, prime_options):
    best = None

    for o in carpet_options:
        base_value = o["points"] * 3.0 + o["roll"]
        score = base_value * _distance_multiplier(o["steps"])
        candidate = {"score": score, **o}
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    for o in prime_options:
        base_value = o["points"] * 2.0 + o["rat_mass"] * 3.0
        score = base_value * _distance_multiplier(o["steps"])
        candidate = {"score": score, **o}
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


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
    """
    Pick the best direction to start a new prime run from current position.
    Scores each direction by:
      - Length of space run (longer = more carpet points)
      - Belief mass in that direction (rat proximity bonus)
    Returns (direction, run_length) or (None, 0).
    """
    px, py = bs.player_worker.get_location()
    best_d, best_score, best_r = None, -1, 0
    for d in DIRS:
        r = _space_run(bs, px, py, d)
        if r < 2:
            continue
        carpet_pts = CARPET_POINTS_TABLE.get(min(r, 7), 0)
        # Belief mass along this run
        rat_mass = 0.0
        cx, cy = px, py
        for _ in range(r):
            cx, cy = _step(cx, cy, d)
            rat_mass += float(belief[_idx(cx, cy)])
        score = carpet_pts * 3.0 + rat_mass * 2.0
        if score > best_score:
            best_score = score
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
    best_i = int(np.argmax(belief))
    best_p = float(belief[best_i])
    ev = 6.0 * best_p - 2.0
    return _loc(best_i), ev


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
        return "global prime/carpet optimizer"

    def play(self, bs, sensor, time_left):
        noise_idx, dist = sensor
        noise_idx = int(noise_idx)

        if self.search_cd > 0:
            self.search_cd -= 1

        self.hmm.predict()
        self.hmm.update(bs, noise_idx, dist)
        belief = self.hmm.b.copy()

        px, py = bs.player_worker.get_location()
        current_cell = bs.get_cell((px, py))
        turns_left = bs.player_worker.turns_left

        # 1) Continue committed prime run first.
        if self.prime_dir is not None and self.prime_remaining > 0:
            c = Move.prime(self.prime_dir)
            if bs.is_valid_move(c):
                self.prime_remaining -= 1
                if self.prime_remaining == 0:
                    self.prime_dir = None
                return c
            else:
                # Run blocked (opponent stepped in, etc.) — abandon
                self.prime_dir = None
                self.prime_remaining = 0
                self.prime_run_start = None

        # 2) Evaluate all reachable carpet and prime opportunities.
        start = (px, py)
        carpet_options = _all_carpet_options(bs, start)
        prime_options = _all_prime_options(bs, start, belief)
        best_option = _best_distance_weighted_option(carpet_options, prime_options)

        if best_option is not None:
            if best_option["kind"] == "carpet":
                if best_option["steps"] == 0:
                    c = Move.carpet(best_option["direction"], best_option["roll"])
                    if bs.is_valid_move(c):
                        self.prime_dir = None
                        self.prime_remaining = 0
                        self.prime_run_start = None
                        return c
                else:
                    d = _next_step_toward(bs, start, best_option["cell"])
                    if d is not None:
                        c = Move.plain(d)
                        if bs.is_valid_move(c):
                            return c

            if best_option["kind"] == "prime":
                if best_option["steps"] == 0 and current_cell == Cell.SPACE:
                    c = Move.prime(best_option["direction"])
                    if bs.is_valid_move(c):
                        self.prime_dir = best_option["direction"]
                        self.prime_remaining = best_option["run"] - 1
                        self.prime_run_start = start
                        self.prime_run_len = best_option["run"]
                        return c
                else:
                    d = _next_step_toward(bs, start, best_option["cell"])
                    if d is not None:
                        c = Move.plain(d)
                        if bs.is_valid_move(c):
                            return c

        # 3) Search when no high-value board action is available.
        if self.search_cd == 0:
            loc, ev = _best_search(belief, bs)
            if ev > 1.5:
                self.search_cd = 3
                return Move.search(loc)

        # 4) Search at lower EV threshold if truly stuck.
        if self.search_cd == 0:
            loc, ev = _best_search(belief, bs)
            if ev > 0:
                self.search_cd = 3
                return Move.search(loc)

        # 5) Fallback: any valid move.
        moves = bs.get_valid_moves(exclude_search=True)
        if moves:
            for m in moves:
                if m.move_type == MoveType.PRIME and bs.is_valid_move(m):
                    return m
            for m in moves:
                if m.move_type == MoveType.PLAIN and bs.is_valid_move(m):
                    return m
            return moves[0]

        loc, _ = _best_search(belief, bs)
        return Move.search(loc)