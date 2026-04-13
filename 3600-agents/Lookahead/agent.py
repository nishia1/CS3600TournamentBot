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


CARPET_WEIGHT = 3.2
CARPET_ROLL_WEIGHT = 0.6
CARPET_ON_CELL_BONUS = 1.2
PRIME_WEIGHT = 3.0
RAT_WEIGHT = 2.0
PRIME_RUN_WEIGHT = 0.25
PRIME_ON_CELL_BONUS = 0.8
DISTANCE_POWER = 0.95
PRIME_MAX_STEPS = 1
PRIME_MAX_RUN = 5
PRIME_MIN_POINTS = 4
STALL_CARPET_TURNS = 4
SEARCH_HIGH_EV = 0.95
SEARCH_LOW_EV = 0.0
SEARCH_STALL_TURNS = 8
MIDGAME_TURNS = 18
ENDGAME_TURNS = 10
OPPONENT_AVOID_WEIGHT = 2.4
PATH_COST_WEIGHT = 1.1
CONTROL_STICK_TURNS = 6
OPEN_SPACE_WEIGHT = 2.2
EDGE_PENALTY = 0.9
STEAL_MIN_POINTS = 4
STEAL_RACE_BONUS = 1.8
STEAL_DENIAL_WEIGHT = 1.4
PRIME_CANCEL_MARGIN = 1.8
TACTICAL_POINT_WEIGHT = 7.0
TACTICAL_FUTURE_CARPET_WEIGHT = 1.1
TACTICAL_CONTROL_WEIGHT = 1.2
TACTICAL_OPP_DIST_WEIGHT = 0.4
TACTICAL_BAD_PRIME_PENALTY = 6.0
TACTICAL_PLAIN_GAP_PENALTY = 0.2
TACTICAL_MIN_SCORE = 2.5
CARPET_PLAN_MIN_POINTS = 3
TACTICAL_CARPET_BONUS = 6.5
PRESSURE_TRAIL_POINTS = 3
PRESSURE_ENDGAME_TURNS = 12
DEFEND_MIN_POINTS = 3
DEFEND_THREAT_WINDOW = 1
DEFEND_URGENCY_WEIGHT = 2.0

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


def _prime_bridge_value(bs, x, y, d):
    """
    If we PRIME from (x,y) toward d, estimate how much primed chain we connect to
    from the next square after moving.
    """
    if bs.get_cell((x, y)) != Cell.SPACE:
        return 0

    nx, ny = _step(x, y, d)
    if not bs.is_valid_cell((nx, ny)):
        return 0
    if (nx, ny) == bs.opponent_worker.get_location():
        return 0
    if bs.get_cell((nx, ny)) != Cell.SPACE:
        return 0

    return _primed_run(bs, nx, ny, d)


def _prime_time_feasible(turns_left, steps_to_start, run_len):
    """
    Conservative time budget:
    steps_to_start (movement/setup) + run_len (prime turns) + 1 (carpet turn).
    """
    needed = int(steps_to_start) + max(1, int(run_len)) + 1
    return turns_left >= needed


def _estimate_continue_prime_value(bs, turns_left, prime_remaining):
    """
    Approximate value of continuing current prime commitment.
    Includes immediate prime point plus discounted future carpet conversion value.
    """
    rem = max(1, int(prime_remaining))
    if not _prime_time_feasible(turns_left, 0, rem):
        return -1e9
    future_carpet = _carpet_points(rem)
    return 1.0 + 0.65 * future_carpet


def _estimate_best_alt_value(bs, start):
    """
    Approximate best alternative value from steal/carpet opportunities.
    """
    best = -1e9

    steal = _best_steal_option(bs, start)
    if steal is not None:
        s = steal["points"] * 3.8 / (steal["steps"] + 1.0) + steal["roll"] * 0.5
        best = max(best, s)

    defend = _best_defend_option(bs, start)
    if defend is not None:
        s = defend["points"] * 4.1 / (defend["steps"] + 1.0) + defend["roll"] * 0.6
        best = max(best, s)

    for o in _all_carpet_options(bs, start):
        s = o["points"] * 3.2 / (o["steps"] + 1.0) + o["roll"] * 0.5
        best = max(best, s)

    return best


def _best_reachable_carpet_option(bs, start, turns_left, min_points=CARPET_PLAN_MIN_POINTS, prefer_contested=False):
    options = _all_carpet_options(bs, start)
    ox, oy = bs.opponent_worker.get_location()
    opp_dist = _bfs_avoiding(bs, (ox, oy), start)
    best = None
    best_score = -1e9
    for o in options:
        if o["points"] < min_points:
            continue
        # Need at least steps to reach + one carpet turn.
        if turns_left < o["steps"] + 1:
            continue

        them = opp_dist.get(o["cell"], 99)
        race_margin = them - o["steps"]
        urgency = 0.0
        if them <= o["steps"] + 1:
            urgency = (o["steps"] + 1 - them) * 2.0

        score = (
            o["points"] * 4.0
            + o["roll"] * 0.8
            - o["steps"] * 1.2
            + race_margin * 0.9
            + urgency
        )
        if prefer_contested and them <= o["steps"] + 1:
            score += 2.0
        if score > best_score:
            best_score = score
            best = o
    return best


def _best_tactical_move(bs, turns_left):
    """
    One-ply tactical selector for consistency: immediate gain + future conversion +
    local control. Returns (move, score) or (None, -inf).
    """
    moves = bs.get_valid_moves(exclude_search=True)
    if not moves:
        return None, -1e9

    cur_points = bs.player_worker.get_points()
    cur_pos = bs.player_worker.get_location()
    cur_cell = bs.get_cell(cur_pos)

    best_move = None
    best_score = -1e9

    for m in moves:
        nxt = bs.forecast_move(m, check_ok=True)
        if nxt is None:
            continue

        nxt_points = nxt.player_worker.get_points()
        immediate = nxt_points - cur_points

        nx, ny = nxt.player_worker.get_location()
        ox, oy = nxt.opponent_worker.get_location()

        max_roll = 0
        for d in DIRS:
            max_roll = max(max_roll, _primed_run(nxt, nx, ny, d))

        score = 0.0
        score += immediate * TACTICAL_POINT_WEIGHT
        score += _carpet_points(max_roll) * TACTICAL_FUTURE_CARPET_WEIGHT
        score += _control_open_score(nxt, nx, ny) * TACTICAL_CONTROL_WEIGHT
        score += (abs(nx - ox) + abs(ny - oy)) * TACTICAL_OPP_DIST_WEIGHT

        if m.move_type == MoveType.CARPET:
            score += TACTICAL_CARPET_BONUS

        if m.move_type == MoveType.PRIME and not _prime_time_feasible(turns_left, 0, max(1, max_roll)):
            score -= TACTICAL_BAD_PRIME_PENALTY

        if m.move_type == MoveType.PLAIN and cur_cell == Cell.SPACE:
            score -= TACTICAL_PLAIN_GAP_PENALTY

        if score > best_score:
            best_score = score
            best_move = m

    return best_move, best_score


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
    # Heavier distance discount to avoid wandering while opponent cashes carpets.
    return 1.0 / ((steps + 1.0) ** DISTANCE_POWER)


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
        base_value = o["points"] * CARPET_WEIGHT + o["roll"] * CARPET_ROLL_WEIGHT
        if o["steps"] == 0:
            base_value += CARPET_ON_CELL_BONUS
        score = base_value * _distance_multiplier(o["steps"])
        candidate = {"score": score, **o}
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    for o in prime_options:
        base_value = o["points"] * PRIME_WEIGHT + o["rat_mass"] * RAT_WEIGHT
        base_value += o["run"] * PRIME_RUN_WEIGHT
        if o["steps"] == 0:
            base_value += PRIME_ON_CELL_BONUS
        score = base_value * _distance_multiplier(o["steps"])
        candidate = {"score": score, **o}
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def _best_distance_weighted_carpet_option(carpet_options):
    best = None
    for o in carpet_options:
        base_value = o["points"] * CARPET_WEIGHT + o["roll"] * CARPET_ROLL_WEIGHT
        if o["steps"] == 0:
            base_value += CARPET_ON_CELL_BONUS
        score = base_value * _distance_multiplier(o["steps"])
        candidate = {"score": score, **o}
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def _best_steal_option(bs, start):
    """
    Find a high-value carpet opportunity that we can likely claim before opponent.
    Returns an option dict from _all_carpet_options, or None.
    """
    carpet_options = _all_carpet_options(bs, start)
    if not carpet_options:
        return None

    ox, oy = bs.opponent_worker.get_location()
    opp_dist = _bfs_avoiding(bs, (ox, oy), start)

    best = None
    best_score = -1e9
    for o in carpet_options:
        if o["points"] < STEAL_MIN_POINTS:
            continue

        us = o["steps"]
        them = opp_dist.get(o["cell"], 99)
        # Need to arrive no later than opponent to realistically steal/deny.
        if us > them:
            continue

        # Denial estimate: if opponent reaches this cell, estimate their best
        # immediate convertible primed roll from that location.
        ox, oy = o["cell"]
        opp_best_roll = 0
        for d in DIRS:
            opp_best_roll = max(opp_best_roll, _primed_run(bs, ox, oy, d))
        denied_points = _carpet_points(opp_best_roll)

        race_margin = them - us
        score = (
            o["points"] * 5.0
            + o["roll"] * 1.5
            + race_margin * STEAL_RACE_BONUS
            + denied_points * STEAL_DENIAL_WEIGHT
        )
        if score > best_score:
            best_score = score
            best = o

    return best


def _best_defend_option(bs, start):
    """
    Identify valuable carpet opportunities that are threatened by opponent arrival.
    """
    carpet_options = _all_carpet_options(bs, start)
    if not carpet_options:
        return None

    ox, oy = bs.opponent_worker.get_location()
    opp_dist = _bfs_avoiding(bs, (ox, oy), start)

    best = None
    best_score = -1e9
    for o in carpet_options:
        if o["points"] < DEFEND_MIN_POINTS:
            continue

        us = o["steps"]
        them = opp_dist.get(o["cell"], 99)

        # Threatened if opponent can arrive now or soon.
        if them > us + DEFEND_THREAT_WINDOW:
            continue

        urgency = max(0, us + DEFEND_THREAT_WINDOW - them)
        score = (
            o["points"] * 4.2
            + o["roll"] * 1.2
            + urgency * DEFEND_URGENCY_WEIGHT
            - us * 0.8
        )
        if score > best_score:
            best_score = score
            best = o

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


def _bfs_avoiding(bs, start, avoid_loc):
    dist = {start: 0}
    q = [start]
    head = 0
    while head < len(q):
        cx, cy = q[head]
        head += 1
        for d in DIRS:
            nx, ny = _step(cx, cy, d)
            if (nx, ny) in dist:
                continue
            if not bs.is_valid_cell((nx, ny)):
                continue
            if (nx, ny) == avoid_loc:
                continue
            c = bs.get_cell((nx, ny))
            if c in (Cell.BLOCKED, Cell.PRIMED):
                continue
            dist[(nx, ny)] = dist[(cx, cy)] + 1
            q.append((nx, ny))
    return dist


def _next_step_toward(bs, start, target, avoid=None):
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
            if avoid is not None and (nx, ny) == avoid and (cx, cy) == start:
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
    Returns (direction, run_length, bridge_value) or (None, 0, 0).
    """
    px, py = bs.player_worker.get_location()
    best_d, best_score, best_r, best_bridge = None, -1, 0, 0
    for d in DIRS:
        r = _space_run(bs, px, py, d)
        bridge = _prime_bridge_value(bs, px, py, d)
        if r < 2 and bridge <= 0:
            continue
        carpet_pts = CARPET_POINTS_TABLE.get(min(r, 7), 0)
        # Belief mass along this run
        rat_mass = 0.0
        cx, cy = px, py
        for _ in range(r):
            cx, cy = _step(cx, cy, d)
            rat_mass += float(belief[_idx(cx, cy)])
        score = carpet_pts * 3.0 + rat_mass * 2.0 + bridge * 4.0
        if score > best_score:
            best_score = score
            best_d = d
            best_r = r
            best_bridge = bridge
    return best_d, best_r, best_bridge


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
            # Reduce rat-chasing: score mostly by setup quality and board control.
            s = pts * 3.0 + r * 0.6
            if s > best_run_score:
                best_run_score = s
        if best_run_score <= 0:
            continue

        opp_dist = abs(x - ox) + abs(y - oy)
        score = best_run_score + opp_dist * OPPONENT_AVOID_WEIGHT - steps * PATH_COST_WEIGHT
        if score > best_score:
            best_score = score
            best_loc = (x, y)

    return best_loc


def _control_open_score(bs, x, y):
    score = 0.0
    for d in DIRS:
        nx, ny = _step(x, y, d)
        if not bs.is_valid_cell((nx, ny)):
            continue
        c = bs.get_cell((nx, ny))
        if c == Cell.SPACE:
            score += 1.0
        elif c == Cell.PRIMED:
            score += 0.35
    # Prefer cells that can extend into longer lanes.
    best_lane = 0
    for d in DIRS:
        best_lane = max(best_lane, _space_run(bs, x, y, d))
    score += best_lane * 0.6
    return score


def _pick_board_control_target(bs, start, previous_target=None):
    px, py = start
    ox, oy = bs.opponent_worker.get_location()
    bfs_dist = _bfs(bs, start)

    best_target = None
    best_score = -1e9
    for (x, y), steps in bfs_dist.items():
        if steps == 0:
            continue
        if bs.get_cell((x, y)) != Cell.SPACE:
            continue

        open_score = _control_open_score(bs, x, y)
        opp_dist = abs(x - ox) + abs(y - oy)
        edge_dist = min(x, y, BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y)

        score = (
            open_score * OPEN_SPACE_WEIGHT
            + opp_dist * OPPONENT_AVOID_WEIGHT
            - steps * PATH_COST_WEIGHT
            - (2 - min(edge_dist, 2)) * EDGE_PENALTY
        )

        # Keep pursuing prior target when still good to reduce oscillation.
        if previous_target is not None and (x, y) == previous_target:
            score += 2.0

        if score > best_score:
            best_score = score
            best_target = (x, y)

    return best_target


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
        self.last_points = 0
        self.no_gain_turns = 0
        self.control_target = None
        self.control_target_age = 0
        self.prev_pos = None

    def _ret(self, current_pos, move):
        self.prev_pos = current_pos
        return move

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
        my_points = bs.player_worker.get_points()
        opp_points = bs.opponent_worker.get_points()

        if my_points > self.last_points:
            self.no_gain_turns = 0
        else:
            self.no_gain_turns += 1
        self.last_points = my_points

        score_gap = my_points - opp_points
        pressure_mode = (turns_left <= PRESSURE_ENDGAME_TURNS) or (score_gap <= -PRESSURE_TRAIL_POINTS)
        # 1) Carpet immediately if profitable.
        carpet = _best_carpet_move(bs)
        if carpet is not None:
            self.prime_dir = None
            self.prime_remaining = 0
            self.prime_run_start = None
            return self._ret((px, py), carpet)

        # 1.5) Steal/deny valuable primed lanes before opponent can convert.
        steal = _best_steal_option(bs, (px, py))
        if steal is not None:
            if steal["steps"] == 0:
                c = Move.carpet(steal["direction"], steal["roll"])
                if bs.is_valid_move(c):
                    self.prime_dir = None
                    self.prime_remaining = 0
                    self.prime_run_start = None
                    self.control_target = None
                    self.control_target_age = 0
                    return self._ret((px, py), c)
            else:
                d = _next_step_toward(bs, (px, py), steal["cell"], avoid=self.prev_pos)
                if d is None:
                    d = _next_step_toward(bs, (px, py), steal["cell"])
                if d is not None:
                    # When racing to steal, prefer arriving quickly over creating more primed residue.
                    c = Move.plain(d)
                    if bs.is_valid_move(c):
                        return self._ret((px, py), c)

        # 1.6) Defend threatened primed lanes from being stolen.
        defend = _best_defend_option(bs, (px, py))
        if defend is not None:
            if defend["steps"] == 0:
                c = Move.carpet(defend["direction"], defend["roll"])
                if bs.is_valid_move(c):
                    self.prime_dir = None
                    self.prime_remaining = 0
                    self.prime_run_start = None
                    return self._ret((px, py), c)
            else:
                d = _next_step_toward(bs, (px, py), defend["cell"], avoid=self.prev_pos)
                if d is None:
                    d = _next_step_toward(bs, (px, py), defend["cell"])
                if d is not None:
                    # Rush to conversion point; don't lay extra prime while defending.
                    c = Move.plain(d)
                    if bs.is_valid_move(c):
                        return self._ret((px, py), c)

        # 1.75) If there are good reachable carpets, route toward converting them.
        plan_min_points = 2 if pressure_mode else CARPET_PLAN_MIN_POINTS
        carpet_plan = _best_reachable_carpet_option(
            bs,
            (px, py),
            turns_left,
            min_points=plan_min_points,
            prefer_contested=pressure_mode,
        )
        if carpet_plan is not None:
            if carpet_plan["steps"] == 0:
                c = Move.carpet(carpet_plan["direction"], carpet_plan["roll"])
                if bs.is_valid_move(c):
                    self.prime_dir = None
                    self.prime_remaining = 0
                    self.prime_run_start = None
                    return self._ret((px, py), c)
            else:
                d = _next_step_toward(bs, (px, py), carpet_plan["cell"], avoid=self.prev_pos)
                if d is None:
                    d = _next_step_toward(bs, (px, py), carpet_plan["cell"])
                if d is not None:
                    c = Move.plain(d)
                    if bs.is_valid_move(c):
                        return self._ret((px, py), c)

        # 2) Continue committed prime run.
        if self.prime_dir is not None and self.prime_remaining > 0:
            # If there isn't enough time to finish this prime sequence and convert,
            # drop commitment and pivot.
            if turns_left <= self.prime_remaining + 1:
                self.prime_dir = None
                self.prime_remaining = 0
                self.prime_run_start = None
            else:
                continue_value = _estimate_continue_prime_value(bs, turns_left, self.prime_remaining)
                alt_value = _estimate_best_alt_value(bs, (px, py))

                if alt_value > continue_value + PRIME_CANCEL_MARGIN:
                    self.prime_dir = None
                    self.prime_remaining = 0
                    self.prime_run_start = None
                else:
                    c = Move.prime(self.prime_dir)
                    if bs.is_valid_move(c):
                        self.prime_remaining -= 1
                        if self.prime_remaining == 0:
                            self.prime_dir = None
                        return self._ret((px, py), c)
                    else:
                        self.prime_dir = None
                        self.prime_remaining = 0
                        self.prime_run_start = None

        # 2.5) Tactical one-turn choice to reduce sporadic behavior.
        tactical_move, tactical_score = _best_tactical_move(bs, turns_left)
        if (
            tactical_move is not None
            and tactical_score >= TACTICAL_MIN_SCORE
            and tactical_move.move_type == MoveType.CARPET
        ):
            return self._ret((px, py), tactical_move)

        # 3) Start a strong prime run if on SPACE.
        if current_cell == Cell.SPACE:
            best_d, best_r, best_bridge = _pick_prime_direction(bs, belief)
            if best_d is not None:
                expected = CARPET_POINTS_TABLE.get(min(best_r, BOARD_SIZE - 1), 0)
                if ((best_r >= 3 and expected >= 4) or best_bridge >= 2) and _prime_time_feasible(turns_left, 0, best_r):
                    c = Move.prime(best_d)
                    if bs.is_valid_move(c):
                        if best_r >= 2:
                            self.prime_dir = best_d
                            self.prime_remaining = best_r - 1
                        else:
                            self.prime_dir = None
                            self.prime_remaining = 0
                        self.prime_run_start = (px, py)
                        self.prime_run_len = best_r
                        return self._ret((px, py), c)

        # 4) Search when EV is high.
        if self.search_cd == 0:
            loc, ev = _best_search(belief, bs)
            high_search_threshold = 1.15 if pressure_mode else 1.8
            if ev > high_search_threshold:
                self.search_cd = 3
                return self._ret((px, py), Move.search(loc))

        # 5) Move toward best prime-building space destination.
        # Keep a stable board-control target for several turns.
        bfs_now = _bfs(bs, (px, py))
        keep_target = (
            self.control_target is not None
            and self.control_target in bfs_now
            and self.control_target != (px, py)
            and self.control_target_age < CONTROL_STICK_TURNS
        )
        if keep_target:
            target = self.control_target
            self.control_target_age += 1
        else:
            target = _pick_board_control_target(bs, (px, py), self.control_target)
            self.control_target = target
            self.control_target_age = 0

        if target is not None and target != (px, py):
            d = _next_step_toward(bs, (px, py), target, avoid=self.prev_pos)
            if d is None:
                d = _next_step_toward(bs, (px, py), target)
            if d is not None:
                if current_cell == Cell.SPACE:
                    sr = _space_run(bs, px, py, d)
                    bridge = _prime_bridge_value(bs, px, py, d)
                    if (sr >= 2 or bridge > 0) and _prime_time_feasible(turns_left, 0, max(sr, 1)):
                        c = Move.prime(d)
                        if bs.is_valid_move(c):
                            if sr >= 2:
                                self.prime_dir = d
                                self.prime_remaining = sr - 1
                            else:
                                self.prime_dir = None
                                self.prime_remaining = 0
                            self.prime_run_start = (px, py)
                            self.prime_run_len = sr
                            return self._ret((px, py), c)
                c = Move.plain(d)
                if bs.is_valid_move(c):
                    return self._ret((px, py), c)

        if target == (px, py):
            self.control_target = None
            self.control_target_age = 0

        # 6) Lower-threshold search if stuck.
        if self.search_cd == 0:
            loc, ev = _best_search(belief, bs)
            low_search_threshold = 0.35 if (pressure_mode and self.no_gain_turns >= 10) else 0.8
            if ev > low_search_threshold and self.no_gain_turns >= 10:
                self.search_cd = 3
                return self._ret((px, py), Move.search(loc))

        # 7) Fallback: any valid move.
        moves = bs.get_valid_moves(exclude_search=True)
        if moves:
            for m in moves:
                if m.move_type == MoveType.PRIME and bs.is_valid_move(m):
                    return self._ret((px, py), m)
            for m in moves:
                if m.move_type == MoveType.PLAIN and bs.is_valid_move(m):
                    return self._ret((px, py), m)
            return self._ret((px, py), moves[0])

        loc, _ = _best_search(belief, bs)
        return self._ret((px, py), Move.search(loc))