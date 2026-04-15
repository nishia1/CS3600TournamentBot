from collections.abc import Callable
import json
import pathlib
import random
import time
import numpy as np

from game.move import Move
from game.enums import Cell, Direction, MoveType, BOARD_SIZE, CARPET_POINTS_TABLE


N = BOARD_SIZE * BOARD_SIZE
DIRS = list(Direction)
DELTA = {
	Direction.UP: (0, -1),
	Direction.DOWN: (0, 1),
	Direction.LEFT: (-1, 0),
	Direction.RIGHT: (1, 0),
}

LOW_TIME_GREEDY_SECONDS = 8.0
LOW_TIME_SHALLOW_SECONDS = 20.0
ENDGAME_CONVERT_TURNS = 16
CLUTCH_TURNS = 10
CLUTCH_GAP = 8
RUSH_CARPET_MAX_STEPS = 3
RUSH_CARPET_MIN_POINTS = 4

CONTEST_MIN_POINTS = 3
ROOT_OPP_THREAT_WEIGHT = 2.8
NO_PRIME_MOBILITY_CUTOFF = 1
NO_PRIME_CARPET_EXCEPTION = 4
ROOT_RESPONSE_WEIGHT = 0.85
SEARCH_BELIEF_MIN = 0.58
RL_STATE_FILE = "cassie_rl_state.json"

_SEARCH_TT = {}
_REPLY_TT = {}

# Submission mode: use fixed tuned constants and avoid runtime file I/O.
SUBMISSION_MODE = True
SUBMISSION_PROFILE = {
	"convert_soft_threshold": 1,
	"contest_min_points": 3,
	"root_threat_scale": 0.95,
	"endgame_convert_turns": 20,
}

DEFAULT_RL_STATE = {
	"version": 2,
	"epsilon": 0.18,
	"matches_seen": 0,
	"mutation_period": 40,
	"profiles": {
		"balanced": {
			"convert_soft_threshold": 2,
			"contest_min_points": 3,
			"root_threat_scale": 1.0,
			"endgame_convert_turns": 16,
			"q": 0.0,
			"n": 0,
			"q_seat": {},
			"n_seat": {},
		},
		"deny_heavy": {
			"convert_soft_threshold": 2,
			"contest_min_points": 2,
			"root_threat_scale": 1.22,
			"endgame_convert_turns": 18,
			"q": 0.0,
			"n": 0,
			"q_seat": {},
			"n_seat": {},
		},
		"cash_fast": {
			"convert_soft_threshold": 1,
			"contest_min_points": 3,
			"root_threat_scale": 0.95,
			"endgame_convert_turns": 20,
			"q": 0.0,
			"n": 0,
			"q_seat": {},
			"n_seat": {},
		},
	},
}


def idx_xy(x, y):
	return y * BOARD_SIZE + x


def loc_i(i):
	return (i % BOARD_SIZE, i // BOARD_SIZE)


def step_xy(x, y, d):
	dx, dy = DELTA[d]
	return x + dx, y + dy


class RatHMM:
	NOISE = {
		Cell.BLOCKED: (0.5, 0.3, 0.2),
		Cell.SPACE: (0.7, 0.15, 0.15),
		Cell.PRIMED: (0.1, 0.8, 0.1),
		Cell.CARPET: (0.1, 0.1, 0.8),
	}
	OFF = (-1, 0, 1, 2)
	OFF_P = (0.12, 0.70, 0.12, 0.06)

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
			x, y = loc_i(i)
			cell = bs.get_cell((x, y))
			if cell == Cell.BLOCKED:
				continue
			pn = self.NOISE[cell][noise_idx]
			td = abs(x - wx) + abs(y - wy)
			pd = 0.0
			for off, p in zip(self.OFF, self.OFF_P):
				if max(0, td + off) == dist:
					pd += p
			like[i] = pn * pd

		s = like.sum()
		if s > 1e-12:
			self.b = self.b * like
			total = float(self.b.sum())
			if total > 1e-12 and np.isfinite(total):
				self.b /= total
			else:
				self.b = np.ones(N, dtype=np.float64) / N
		else:
			mask = np.array([
				0.0 if bs.get_cell(loc_i(i)) == Cell.BLOCKED else 1.0
				for i in range(N)
			], dtype=np.float64)
			ms = float(mask.sum())
			if ms > 1e-12:
				self.b = mask / ms
			else:
				self.b = np.ones(N, dtype=np.float64) / N


def primed_run(bs, x, y, d):
	r = 0
	cx, cy = x, y
	while True:
		cx, cy = step_xy(cx, cy, d)
		if not bs.is_valid_cell((cx, cy)):
			break
		if bs.get_cell((cx, cy)) == Cell.PRIMED:
			r += 1
		else:
			break
	return r


def space_run_to(bs, x, y, d, stop_loc):
	r = 0
	cx, cy = x, y
	while True:
		cx, cy = step_xy(cx, cy, d)
		if not bs.is_valid_cell((cx, cy)):
			break
		if (cx, cy) == stop_loc:
			break
		if bs.get_cell((cx, cy)) == Cell.SPACE:
			r += 1
		else:
			break
	return r


def space_run(bs, x, y, d):
	return space_run_to(bs, x, y, d, bs.opponent_worker.get_location())


def immediate_carpet_value(bs):
	x, y = bs.player_worker.get_location()
	best = 0
	for d in DIRS:
		r = primed_run(bs, x, y, d)
		if r > 0:
			best = max(best, CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0))
	return best


def immediate_carpet_value_enemy(bs):
	ox, oy = bs.opponent_worker.get_location()
	px, py = bs.player_worker.get_location()
	best = 0
	for d in DIRS:
		run = 0
		cx, cy = ox, oy
		while True:
			cx, cy = step_xy(cx, cy, d)
			if not bs.is_valid_cell((cx, cy)):
				break
			if (cx, cy) == (px, py):
				break
			if bs.get_cell((cx, cy)) == Cell.PRIMED:
				run += 1
			else:
				break
		if run > 0:
			best = max(best, CARPET_POINTS_TABLE.get(min(run, BOARD_SIZE - 1), 0))
	return best


def open_neighbors(bs, pos):
	x, y = pos
	c = 0
	for d in DIRS:
		nx, ny = step_xy(x, y, d)
		if not bs.is_valid_cell((nx, ny)):
			continue
		if (nx, ny) == bs.opponent_worker.get_location():
			continue
		if bs.get_cell((nx, ny)) != Cell.BLOCKED:
			c += 1
	return c


def cell_potential(bs, pos):
	return cell_potential_for(bs, pos, bs.opponent_worker.get_location())


def cell_potential_for(bs, pos, stop_loc):
	x, y = pos
	if not bs.is_valid_cell((x, y)):
		return 0.0
	cell = bs.get_cell((x, y))
	if cell == Cell.BLOCKED:
		return 0.0

	best = 0.0
	for d in DIRS:
		if cell == Cell.PRIMED:
			r = primed_run(bs, x, y, d)
		else:
			r = space_run_to(bs, x, y, d, stop_loc)
		if r < 2:
			continue
		points = CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0)
		best = max(best, points + 0.7 * carpet_roll_bonus(r))
	return best


def best_reachable_potential(bs, start):
	return best_reachable_potential_for(bs, start, bs.opponent_worker.get_location())


def best_reachable_potential_for(bs, start, stop_loc):
	dist = bfs_dist(bs, start)
	best_score = 0.0
	best_cell = None
	best_steps = 0
	for pos, steps in dist.items():
		pot = cell_potential_for(bs, pos, stop_loc)
		if pot <= 0.0:
			continue
		score = pot / ((steps + 1.0) ** 0.92)
		if score > best_score:
			best_score = score
			best_cell = pos
			best_steps = steps
	return best_cell, best_steps, best_score


def standard_passable(bs, nx, ny):
	if not bs.is_valid_cell((nx, ny)):
		return False
	if (nx, ny) == bs.opponent_worker.get_location():
		return False
	c = bs.get_cell((nx, ny))
	return c not in (Cell.BLOCKED, Cell.PRIMED)


def bfs_dist(bs, start):
	dist = {start: 0}
	q = [start]
	head = 0
	while head < len(q):
		cx, cy = q[head]
		head += 1
		for d in DIRS:
			nx, ny = step_xy(cx, cy, d)
			if (nx, ny) in dist:
				continue
			if standard_passable(bs, nx, ny):
				dist[(nx, ny)] = dist[(cx, cy)] + 1
				q.append((nx, ny))
	return dist


def bfs_dist_avoiding(bs, start, avoid):
	dist = {start: 0}
	q = [start]
	head = 0
	while head < len(q):
		cx, cy = q[head]
		head += 1
		for d in DIRS:
			nx, ny = step_xy(cx, cy, d)
			if (nx, ny) in dist:
				continue
			if (nx, ny) == avoid:
				continue
			if standard_passable(bs, nx, ny):
				dist[(nx, ny)] = dist[(cx, cy)] + 1
				q.append((nx, ny))
	return dist


def next_step_toward(bs, start, target):
	if start == target:
		return None
	prev = {start: None}
	q = [start]
	head = 0
	found = False
	while head < len(q):
		cx, cy = q[head]
		head += 1
		if (cx, cy) == target:
			found = True
			break
		for d in DIRS:
			nx, ny = step_xy(cx, cy, d)
			if (nx, ny) in prev:
				continue
			if standard_passable(bs, nx, ny):
				prev[(nx, ny)] = ((cx, cy), d)
				q.append((nx, ny))
	if not found:
		return None
	node = target
	while prev[node][0] != start:
		node = prev[node][0]
	return prev[node][1]


def all_carpet_options(bs, start):
	dist = bfs_dist(bs, start)
	if start not in dist:
		dist[start] = 0
	options = []
	for (x, y), steps in dist.items():
		for d in DIRS:
			r = primed_run(bs, x, y, d)
			if r < 1:
				continue
			points = CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0)
			options.append({"cell": (x, y), "steps": steps, "direction": d, "roll": r, "points": points})
	return options


def carpet_roll_bonus(roll):
	# Push strategy toward long carpets that are worth disproportionately more points.
	bonus = 0.0
	if roll >= 4:
		bonus += (roll - 3) * 2.4
	if roll >= 6:
		bonus += (roll - 5) * 3.6
	return bonus


def best_nearby_carpet_option(bs, start, turns_left, max_steps=1, min_points=1):
	options = all_carpet_options(bs, start)
	if not options:
		return None

	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < min_points:
			continue
		if o["steps"] > max_steps:
			continue
		if turns_left < o["steps"] + 1:
			continue
		score = o["points"] * 6.0 + o["roll"] * 1.2 + carpet_roll_bonus(o["roll"]) - o["steps"] * 2.5
		if o["steps"] == 0:
			score += 4.0
		if score > best_score:
			best_score = score
			best = o
	return best


def best_reachable_carpet_option(bs, start, turns_left, min_points=3):
	options = all_carpet_options(bs, start)
	if not options:
		return None
	ox, oy = bs.opponent_worker.get_location()
	opp_dist = bfs_dist_avoiding(bs, (ox, oy), start)
	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < min_points:
			continue
		if turns_left < o["steps"] + 1:
			continue
		them = opp_dist.get(o["cell"], 99)
		race_margin = them - o["steps"]
		score = o["points"] * 4.3 + o["roll"] * 0.8 + carpet_roll_bonus(o["roll"]) - o["steps"] * 0.9 + race_margin * 0.9
		if score > best_score:
			best_score = score
			best = o
	return best


def best_contested_carpet_option(bs, start, min_points=CONTEST_MIN_POINTS):
	options = all_carpet_options(bs, start)
	if not options:
		return None
	ox, oy = bs.opponent_worker.get_location()
	opp_dist = bfs_dist_avoiding(bs, (ox, oy), start)
	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < min_points:
			continue
		us = o["steps"]
		them = opp_dist.get(o["cell"], 99)
		if us > them + 1:
			continue
		race = them - us
		s = o["points"] * 4.8 + o["roll"] * 0.9 + carpet_roll_bonus(o["roll"]) + race * 1.3 - us * 0.65
		if s > best_score:
			best_score = s
			best = o
	return best


def conservative_eval(bs):
	my_pts = bs.player_worker.get_points()
	op_pts = bs.opponent_worker.get_points()
	score = 8.0 * (my_pts - op_pts)

	my_immediate = immediate_carpet_value(bs)
	op_immediate = immediate_carpet_value_enemy(bs)
	score += 2.5 * my_immediate
	score -= 3.2 * op_immediate

	px, py = bs.player_worker.get_location()
	ox, oy = bs.opponent_worker.get_location()
	if bs.get_cell((px, py)) == Cell.SPACE:
		best_prime = 0
		for d in DIRS:
			r = space_run(bs, px, py, d)
			if r >= 2:
				best_prime = max(best_prime, CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0))
		score += 0.9 * best_prime

	score += 0.2 * (abs(px - ox) + abs(py - oy))
	score += 0.6 * open_neighbors(bs, (px, py))

	mobility = len(bs.get_valid_moves(exclude_search=True))
	score += 0.5 * mobility
	if mobility <= 2:
		score -= 3.5
	if mobility <= 1:
		score -= 8.0
	return score


def evaluate(bs, belief):
	my_pts = bs.player_worker.get_points()
	op_pts = bs.opponent_worker.get_points()
	turns_left = bs.player_worker.turns_left
	if turns_left > 28:
		point_weight = 8.1
		mobility_weight = 0.72
		threat_weight = 2.7
	elif turns_left > 14:
		point_weight = 9.8
		mobility_weight = 0.60
		threat_weight = 3.1
	else:
		point_weight = 11.0
		mobility_weight = 0.48
		threat_weight = 3.6

	score = point_weight * (my_pts - op_pts)

	my_immediate = immediate_carpet_value(bs)
	op_immediate = immediate_carpet_value_enemy(bs)
	score += 1.55 * my_immediate
	score -= threat_weight * op_immediate

	px, py = bs.player_worker.get_location()
	ox, oy = bs.opponent_worker.get_location()
	score += 0.45 * open_neighbors(bs, (px, py))
	score += 0.18 * (abs(px - ox) + abs(py - oy))

	my_potential = cell_potential(bs, (px, py))
	best_cell, best_steps, best_potential = best_reachable_potential(bs, (px, py))
	opp_best_cell, opp_best_steps, opp_best_potential = best_reachable_potential_for(bs, (ox, oy), (px, py))
	score += 1.0 * my_potential
	score += 0.95 * best_potential
	score -= 1.5 * opp_best_potential
	if best_cell is not None:
		score += 0.25 * open_neighbors(bs, best_cell)
		score -= 0.12 * best_steps
	if opp_best_cell is not None:
		score -= 0.18 * open_neighbors(bs, opp_best_cell)
		score += 0.10 * opp_best_steps

	mobility = len(bs.get_valid_moves(exclude_search=True))
	score += mobility_weight * mobility
	if mobility <= 2:
		score -= 4.0
	if mobility <= 1:
		score -= 9.0

	# Belief-guided pressure: slightly reward standing near likely rat mass.
	best_i = int(np.argmax(belief))
	best_p = float(belief[best_i])
	bx, by = loc_i(best_i)
	d = abs(px - bx) + abs(py - by)
	score += 1.6 * best_p / (1.0 + d)

	cons = conservative_eval(bs)
	return 0.74 * score + 0.26 * cons


def move_heuristic(bs, m, belief, action_bias):
	px, py = bs.player_worker.get_location()
	val = 0.35 * float(action_bias.get(m.move_type, 0.0))
	if m.move_type == MoveType.CARPET:
		cx, cy = px, py
		for _ in range(m.roll_length):
			cx, cy = step_xy(cx, cy, m.direction)
		val += 20.0 + 2.1 * m.roll_length + 0.8 * carpet_roll_bonus(m.roll_length)
		val += 0.9 * cell_potential(bs, (cx, cy))
		val += 0.2 * open_neighbors(bs, (cx, cy))
	elif m.move_type == MoveType.PRIME:
		sr = space_run(bs, px, py, m.direction)
		nx, ny = step_xy(px, py, m.direction)
		val += 2.0 + 0.9 * sr + 0.8 * CARPET_POINTS_TABLE.get(min(sr, BOARD_SIZE - 1), 0)
		if bs.is_valid_cell((nx, ny)):
			val += 0.8 * cell_potential(bs, (nx, ny))
	elif m.move_type == MoveType.PLAIN:
		nx, ny = step_xy(px, py, m.direction)
		if bs.is_valid_cell((nx, ny)):
			next_potential = cell_potential(bs, (nx, ny))
			val += 0.7 + 2.0 * float(belief[idx_xy(nx, ny)]) + 0.75 * next_potential
			if next_potential < 2.0:
				val -= 1.1
	else:
		val -= 1.8
	return val


def immediate_point_delta(bs, move):
	child = bs.forecast_move(move, check_ok=True)
	if child is None:
		return -999.0
	return float(child.player_worker.get_points() - bs.player_worker.get_points())


def _best_reply_eval(bs, belief, action_bias, max_branch):
	key = transposition_key(bs, belief, 1, 0)
	if key in _REPLY_TT:
		return _REPLY_TT[key]

	moves = top_ordered_moves(bs, belief, action_bias, max_branch)
	if not moves:
		val = evaluate(bs, belief)
		_REPLY_TT[key] = val
		return val

	worst = 1e18
	for m in moves:
		child = bs.forecast_move(m, check_ok=True)
		if child is None:
			continue
		child.reverse_perspective()
		val = evaluate(child, belief)
		if val < worst:
			worst = val

	if worst == 1e18:
		worst = evaluate(bs, belief)
	_REPLY_TT[key] = worst
	return worst


def top_ordered_moves(bs, belief, action_bias, max_branch):
	moves = bs.get_valid_moves(exclude_search=True)
	if not moves:
		return []
	ordered = sorted(moves, key=lambda mv: move_heuristic(bs, mv, belief, action_bias), reverse=True)
	return ordered[:max_branch]


def tactical_volatility(bs):
	my_immediate = immediate_carpet_value(bs)
	op_immediate = immediate_carpet_value_enemy(bs)
	mobility = len(bs.get_valid_moves(exclude_search=True))
	return my_immediate >= 2 or op_immediate >= 2 or mobility <= 3


def transposition_key(bs, belief, depth, extension):
	clean_belief = np.nan_to_num(np.asarray(belief, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
	s = float(clean_belief.sum())
	if s > 1e-12:
		clean_belief /= s
	belief_key = tuple(int(round(float(v) * 1000.0)) for v in clean_belief)
	return (
		depth,
		extension,
		bs.turn_count,
		bs.is_player_a_turn,
		bs._space_mask,
		bs._primed_mask,
		bs._carpet_mask,
		bs._blocked_mask,
		bs.player_worker.get_location(),
		bs.opponent_worker.get_location(),
		bs.player_worker.get_points(),
		bs.opponent_worker.get_points(),
		bs.player_worker.turns_left,
		bs.opponent_worker.turns_left,
		belief_key,
	)


class PlayerAgent:
	def __init__(self, board, transition_matrix=None, time_left: Callable = None):
		if transition_matrix is not None:
			T = np.array(transition_matrix, dtype=np.float64)
		else:
			T = np.full((N, N), 1.0 / N)
		self.hmm = RatHMM(T)
		self.search_cd = 0

		self.q_type = {
			MoveType.PLAIN: 0.0,
			MoveType.PRIME: 0.0,
			MoveType.CARPET: 0.0,
			MoveType.SEARCH: 0.0,
		}
		self.alpha = 0.08
		self.prev_points = 0
		self.prev_opp_points = 0
		self.prev_action_type = None
		self._agent_dir = pathlib.Path(__file__).resolve().parent
		self._rl_state_path = self._agent_dir / RL_STATE_FILE
		self._rl_state = self._load_rl_state() if not SUBMISSION_MODE else json.loads(json.dumps(DEFAULT_RL_STATE))
		self._seat_key = None
		if SUBMISSION_MODE:
			self._profile_name = "submission_fixed"
			self._profile = SUBMISSION_PROFILE.copy()
			self._profile_ready = True
		else:
			self._profile_name = "balanced"
			self._profile = DEFAULT_RL_STATE["profiles"]["balanced"].copy()
			self._profile_ready = False
		self.convert_soft_threshold = int(self._profile.get("convert_soft_threshold", 2))
		self.contest_min_points = int(self._profile.get("contest_min_points", CONTEST_MIN_POINTS))
		self.root_threat_scale = float(self._profile.get("root_threat_scale", 1.0))
		self.endgame_convert_turns = int(self._profile.get("endgame_convert_turns", ENDGAME_CONVERT_TURNS))
		self._final_tuned = False

	def commentate(self):
		return f"cassie: hmm + mcts+ab hybrid [{self._profile_name}]"

	def _load_rl_state(self):
		if not self._rl_state_path.exists():
			return json.loads(json.dumps(DEFAULT_RL_STATE))
		try:
			with self._rl_state_path.open("r", encoding="utf-8") as f:
				loaded = json.load(f)
			if "profiles" not in loaded:
				return json.loads(json.dumps(DEFAULT_RL_STATE))
			for name, profile in DEFAULT_RL_STATE["profiles"].items():
				loaded["profiles"].setdefault(name, profile.copy())
				loaded["profiles"][name].setdefault("q", 0.0)
				loaded["profiles"][name].setdefault("n", 0)
				loaded["profiles"][name].setdefault("q_seat", {})
				loaded["profiles"][name].setdefault("n_seat", {})
			loaded.setdefault("epsilon", DEFAULT_RL_STATE["epsilon"])
			loaded.setdefault("matches_seen", 0)
			loaded.setdefault("mutation_period", DEFAULT_RL_STATE["mutation_period"])
			return loaded
		except Exception:
			return json.loads(json.dumps(DEFAULT_RL_STATE))

	def _save_rl_state(self):
		if SUBMISSION_MODE:
			return
		try:
			with self._rl_state_path.open("w", encoding="utf-8") as f:
				json.dump(self._rl_state, f, indent=2)
		except Exception:
			pass

	def _infer_seat_key(self, bs):
		if self._seat_key is not None:
			return
		x, y = bs.player_worker.get_location()
		self._seat_key = f"{x},{y}"

	def _profile_value(self, profile):
		base_q = float(profile.get("q", 0.0))
		if self._seat_key is None:
			return base_q
		q_seat = profile.get("q_seat", {})
		n_seat = profile.get("n_seat", {})
		seat_q = float(q_seat.get(self._seat_key, base_q))
		seat_n = int(n_seat.get(self._seat_key, 0))
		seat_weight = min(0.45, seat_n / 60.0)
		return (1.0 - seat_weight) * base_q + seat_weight * seat_q

	def _select_profile(self):
		if SUBMISSION_MODE:
			return "submission_fixed", SUBMISSION_PROFILE.copy()
		profiles = self._rl_state["profiles"]
		epsilon = float(self._rl_state.get("epsilon", 0.18))
		names = list(profiles.keys())
		if not names:
			return "balanced", DEFAULT_RL_STATE["profiles"]["balanced"].copy()

		if random.random() < epsilon:
			name = random.choice(names)
		else:
			name = max(names, key=lambda n: self._profile_value(profiles[n]) + random.uniform(0.0, 1e-6))
		return name, profiles[name]

	def _apply_profile_constants(self):
		self.convert_soft_threshold = int(self._profile.get("convert_soft_threshold", 2))
		self.contest_min_points = int(self._profile.get("contest_min_points", CONTEST_MIN_POINTS))
		self.root_threat_scale = float(self._profile.get("root_threat_scale", 1.0))
		self.endgame_convert_turns = int(self._profile.get("endgame_convert_turns", ENDGAME_CONVERT_TURNS))

	def _ensure_profile_selected(self, bs):
		if self._profile_ready:
			return
		self._infer_seat_key(bs)
		self._profile_name, self._profile = self._select_profile()
		self._apply_profile_constants()
		self._profile_ready = True

	def _maybe_mutate_profiles(self):
		profiles = self._rl_state.get("profiles", {})
		if len(profiles) < 2:
			return
		period = max(20, int(self._rl_state.get("mutation_period", 40)))
		seen = int(self._rl_state.get("matches_seen", 0))
		if seen < 1 or seen % period != 0:
			return

		best_name = max(profiles.keys(), key=lambda k: float(profiles[k].get("q", 0.0)))
		worst_name = min(profiles.keys(), key=lambda k: float(profiles[k].get("q", 0.0)))
		if best_name == worst_name:
			return

		best = profiles[best_name]
		worst = profiles[worst_name]
		worst["convert_soft_threshold"] = int(max(1, min(3, int(best.get("convert_soft_threshold", 2)) + random.choice([-1, 0, 1]))))
		worst["contest_min_points"] = int(max(2, min(4, int(best.get("contest_min_points", 3)) + random.choice([-1, 0, 1]))))
		worst["root_threat_scale"] = float(max(0.85, min(1.35, float(best.get("root_threat_scale", 1.0)) + random.uniform(-0.08, 0.08))))
		worst["endgame_convert_turns"] = int(max(14, min(22, int(best.get("endgame_convert_turns", 16)) + random.choice([-2, -1, 0, 1, 2]))))

	def _update_profile_after_match(self, final_gap):
		if SUBMISSION_MODE:
			return
		if self._final_tuned:
			return
		self._final_tuned = True

		profiles = self._rl_state["profiles"]
		if self._profile_name not in profiles:
			return

		p = profiles[self._profile_name]
		n = int(p.get("n", 0))
		q = float(p.get("q", 0.0))
		reward = float(np.tanh(final_gap / 12.0))
		if final_gap > 0:
			reward += 0.10
		elif final_gap < 0:
			reward -= 0.10
		reward = max(-1.0, min(1.0, reward))

		alpha = 0.25 / (1.0 + 0.05 * n)
		p["q"] = q + alpha * (reward - q)
		p["n"] = n + 1

		if self._seat_key is not None:
			q_seat = p.setdefault("q_seat", {})
			n_seat = p.setdefault("n_seat", {})
			seat_n = int(n_seat.get(self._seat_key, 0))
			seat_q = float(q_seat.get(self._seat_key, p["q"]))
			seat_alpha = 0.30 / (1.0 + 0.08 * seat_n)
			q_seat[self._seat_key] = seat_q + seat_alpha * (reward - seat_q)
			n_seat[self._seat_key] = seat_n + 1

		self._rl_state["matches_seen"] = int(self._rl_state.get("matches_seen", 0)) + 1
		self._maybe_mutate_profiles()

		eps = float(self._rl_state.get("epsilon", 0.18))
		self._rl_state["epsilon"] = max(0.05, eps * 0.995)
		self._save_rl_state()

	def _record_action(self, m):
		if m is not None:
			self.prev_action_type = m.move_type
		return m

	def _update_rl(self, bs):
		my_points = bs.player_worker.get_points()
		opp_points = bs.opponent_worker.get_points()
		reward = (my_points - self.prev_points) - 0.55 * (opp_points - self.prev_opp_points)
		reward = max(-4.0, min(4.0, reward))
		self.prev_points = my_points
		self.prev_opp_points = opp_points
		if self.prev_action_type is not None:
			old = self.q_type[self.prev_action_type]
			self.q_type[self.prev_action_type] = old + self.alpha * (reward - old)

	def _forecast_next(self, bs, move):
		child = bs.forecast_move(move, check_ok=True)
		if child is None:
			return None
		child.reverse_perspective()
		return child

	def _negamax(self, bs, belief, depth, alpha, beta, deadline, max_branch, extension=0):
		key = transposition_key(bs, belief, depth, extension)
		cached = _SEARCH_TT.get(key)
		if cached is not None:
			return cached

		if bs.is_game_over() or time.perf_counter() >= deadline:
			val = evaluate(bs, belief)
			_SEARCH_TT[key] = val
			return val
		if depth == 0:
			if extension < 1 and tactical_volatility(bs):
				val = self._negamax(bs, belief, 1, alpha, beta, deadline, max_branch, extension + 1)
				_SEARCH_TT[key] = val
				return val
			val = evaluate(bs, belief)
			_SEARCH_TT[key] = val
			return val

		moves = top_ordered_moves(bs, belief, self.q_type, max_branch)
		if not moves:
			val = evaluate(bs, belief)
			_SEARCH_TT[key] = val
			return val

		next_belief = belief @ self.hmm.T
		best = -1e18
		for m in moves:
			child = self._forecast_next(bs, m)
			if child is None:
				continue
			val = -self._negamax(child, next_belief, depth - 1, -beta, -alpha, deadline, max_branch, extension)
			if val > best:
				best = val
			alpha = max(alpha, best)
			if alpha >= beta:
				break
		_SEARCH_TT[key] = best
		return best

	def _best_move_ab(self, bs, belief, time_left_seconds, score_gap):
		turns_left = bs.player_worker.turns_left
		depth = 5 if turns_left > 12 else 4
		max_branch = 8 if turns_left > 8 else 10

		_, pot_steps, pot_score = best_reachable_potential(bs, bs.player_worker.get_location())
		if pot_score >= 8.0 and pot_steps <= 2 and time_left_seconds > 22:
			depth = 5
			max_branch = 8
		elif pot_score >= 6.0 and pot_steps <= 3 and time_left_seconds > 30:
			depth = 4
			max_branch = 8

		if time_left_seconds <= LOW_TIME_SHALLOW_SECONDS:
			depth = 2
			max_branch = 8
		elif time_left_seconds > 65 and depth < 5 and abs(score_gap) <= 10:
			depth = 5
			max_branch = 7
		if (turns_left <= CLUTCH_TURNS or abs(score_gap) <= CLUTCH_GAP) and time_left_seconds > 35:
			depth = 4
			max_branch = 8 if turns_left > 8 else 10

		budget = 0.26 if time_left_seconds > 40 else 0.14
		if abs(score_gap) <= 5 and turns_left <= 16 and time_left_seconds > 20:
			budget += 0.04
		deadline = time.perf_counter() + budget

		moves = top_ordered_moves(bs, belief, self.q_type, max_branch)
		if not moves:
			return None

		best_move = None
		best_val = -1e18
		alpha, beta = -1e18, 1e18
		next_belief = belief @ self.hmm.T
		for m in moves:
			if time.perf_counter() >= deadline:
				break
			board_after = bs.forecast_move(m, check_ok=True)
			if board_after is None:
				continue
			immediate_gain = float(board_after.player_worker.get_points() - bs.player_worker.get_points())

			# Trap-prime filter.
			if m.move_type == MoveType.PRIME:
				mob_after = len(board_after.get_valid_moves(exclude_search=True))
				my_near_carpet = immediate_carpet_value(board_after)
				if mob_after <= NO_PRIME_MOBILITY_CUTOFF and my_near_carpet < NO_PRIME_CARPET_EXCEPTION:
					continue

			opp_threat = immediate_carpet_value_enemy(board_after)
			child = board_after
			child.reverse_perspective()
			val = -self._negamax(child, next_belief, depth - 1, -beta, -alpha, deadline, max_branch)

			# Explicitly score the opponent's best immediate reply after our move.
			reply_eval = _best_reply_eval(board_after, next_belief, self.q_type, max_branch)
			val += ROOT_RESPONSE_WEIGHT * (reply_eval - evaluate(board_after, next_belief))

			# Reward immediate scoring while keeping enough weight on denial/position.
			if turns_left > 22:
				val += 1.0 * immediate_gain
			elif turns_left > 12:
				val += 1.4 * immediate_gain
			else:
				val += 2.0 * immediate_gain
			if immediate_gain <= 0 and turns_left <= 20 and score_gap <= 0:
				val -= 1.3

			# Root tactical penalty for handing over easy points.
			threat_weight = ROOT_OPP_THREAT_WEIGHT
			threat_weight *= self.root_threat_scale
			if score_gap <= -5:
				threat_weight *= 1.1
			elif score_gap >= 5:
				threat_weight *= 1.2
			if turns_left <= 24 and score_gap < 0:
				threat_weight += 0.4
			val -= threat_weight * opp_threat

			if val > best_val:
				best_val = val
				best_move = m
			alpha = max(alpha, best_val)
			if alpha >= beta:
				break

		return best_move

	def play(self, bs, sensor, time_left):
		self._ensure_profile_selected(bs)
		self._update_rl(bs)

		noise_idx, dist = sensor
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

		self.hmm.predict()
		self.hmm.update(bs, noise_idx, dist)
		belief = np.nan_to_num(self.hmm.b.copy(), nan=0.0, posinf=0.0, neginf=0.0)
		belief_sum = float(belief.sum())
		if belief_sum > 1e-12:
			belief /= belief_sum
		else:
			belief = np.ones(N, dtype=np.float64) / N
		_SEARCH_TT.clear()
		_REPLY_TT.clear()

		turns_left = bs.player_worker.turns_left
		my_points = bs.player_worker.get_points()
		opp_points = bs.opponent_worker.get_points()
		score_gap = my_points - opp_points
		enemy_threat = immediate_carpet_value_enemy(bs)
		if turns_left <= 1 or bs.is_game_over():
			self._update_profile_after_match(score_gap)
		mobility_now = len(bs.get_valid_moves(exclude_search=True))

		# 1) Immediate conversion if available.
		best_carpet = None
		best_carpet_pts = -1
		best_carpet_roll = 0
		px, py = bs.player_worker.get_location()
		for d in DIRS:
			r = primed_run(bs, px, py, d)
			if r > 0:
				pts = CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0)
				if pts > best_carpet_pts:
					best_carpet_pts = pts
					best_carpet_roll = r
					best_carpet = Move.carpet(d, r)

		if (
			best_carpet is not None
			and bs.is_valid_move(best_carpet)
			and (
				best_carpet_pts >= 3
				or (best_carpet_pts >= 3 and (enemy_threat >= 3 or turns_left <= 20 or score_gap < 0))
				or (best_carpet_roll >= 4 and best_carpet_pts >= 2)
				or (best_carpet_pts >= 2 and turns_left <= 24)
				or (best_carpet_pts >= self.convert_soft_threshold and (score_gap >= -4 or turns_left <= 24))
				or turns_left <= self.endgame_convert_turns
				or mobility_now <= 2
			)
		):
			return self._record_action(best_carpet)

		# 2) Short cash route. Prefer taking real points over wandering or probing.
		short_cash = best_reachable_carpet_option(
			bs,
			bs.player_worker.get_location(),
			turns_left,
			min_points=2 if score_gap <= 0 or turns_left <= 18 else 3,
		)
		short_cash_steps_cap = 2
		if short_cash is not None and short_cash["steps"] <= short_cash_steps_cap:
			if short_cash["steps"] == 0:
				c = Move.carpet(short_cash["direction"], short_cash["roll"])
				if bs.is_valid_move(c):
					return self._record_action(c)
			else:
				d = next_step_toward(bs, bs.player_worker.get_location(), short_cash["cell"])
				if d is not None:
					c = Move.plain(d)
					if bs.is_valid_move(c):
						return self._record_action(c)

		# 3) Hard low-time policy.
		if time_left_seconds <= LOW_TIME_GREEDY_SECONDS:
			start = bs.player_worker.get_location()
			near_cash = best_nearby_carpet_option(bs, start, turns_left, max_steps=2, min_points=2)
			if near_cash is not None:
				if near_cash["steps"] == 0:
					c = Move.carpet(near_cash["direction"], near_cash["roll"])
					if bs.is_valid_move(c):
						return self._record_action(c)
				else:
					d = next_step_toward(bs, start, near_cash["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							return self._record_action(c)

			moves = bs.get_valid_moves(exclude_search=True)
			if moves:
				for m in moves:
					if m.move_type == MoveType.CARPET and bs.is_valid_move(m):
						return self._record_action(m)
				for m in moves:
					if m.move_type == MoveType.PLAIN and bs.is_valid_move(m):
						return self._record_action(m)
				return self._record_action(moves[0])

		start = bs.player_worker.get_location()
		pot_cell, pot_steps, pot_score = best_reachable_potential(bs, start)

		if enemy_threat >= 4:
			contested = best_contested_carpet_option(bs, start, min_points=max(2, self.contest_min_points - 1))
			if contested is not None:
				if contested["steps"] == 0:
					c = Move.carpet(contested["direction"], contested["roll"])
					if bs.is_valid_move(c):
						return self._record_action(c)
				else:
					d = next_step_toward(bs, start, contested["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							return self._record_action(c)

		# 4) Fast carpet rush (1-3 steps), biased to long lines.
		if turns_left <= 24 or score_gap < 0 or enemy_threat >= 3:
			rush_steps = RUSH_CARPET_MAX_STEPS if time_left_seconds > LOW_TIME_SHALLOW_SECONDS else 2
			rush_min_points = RUSH_CARPET_MIN_POINTS if turns_left >= 12 else 3
			if pot_score >= 8.0 and pot_steps <= 2:
				rush_min_points = min(rush_min_points, 3)
			if enemy_threat >= 4:
				rush_steps = max(2, rush_steps - 1)
			if score_gap < 0:
				rush_min_points = max(3, rush_min_points - 1)
			if score_gap <= -6 or enemy_threat >= 5:
				rush_min_points = 2
			near_cash = best_nearby_carpet_option(bs, start, turns_left, max_steps=rush_steps, min_points=rush_min_points)
			if near_cash is not None:
				if near_cash["steps"] == 0:
					c = Move.carpet(near_cash["direction"], near_cash["roll"])
					if bs.is_valid_move(c):
						return self._record_action(c)
				else:
					d = next_step_toward(bs, start, near_cash["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							return self._record_action(c)

		# 5) Contested conversion denial before search.
		if (
			enemy_threat >= 3
			or score_gap <= -4
			or (turns_left <= 28 and score_gap <= -2 and enemy_threat >= 2)
			or (turns_left <= 20 and enemy_threat >= 2)
		):
			contested = best_contested_carpet_option(bs, start, min_points=self.contest_min_points)
			if contested is not None:
				if contested["steps"] == 0:
					c = Move.carpet(contested["direction"], contested["roll"])
					if bs.is_valid_move(c):
						return self._record_action(c)
				else:
					d = next_step_toward(bs, start, contested["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							return self._record_action(c)

		# Late anti-swing: prioritize reachable conversion denial when opponent has clear threat.
		if turns_left <= 14 and enemy_threat >= 2:
			cash = best_reachable_carpet_option(bs, start, turns_left, min_points=2)
			if cash is not None:
				if cash["steps"] == 0:
					c = Move.carpet(cash["direction"], cash["roll"])
					if bs.is_valid_move(c):
						return self._record_action(c)
				else:
					d = next_step_toward(bs, start, cash["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							return self._record_action(c)

		# 6) Endgame route to guaranteed carpets.
		if turns_left <= self.endgame_convert_turns:
			cash = best_reachable_carpet_option(bs, start, turns_left)
			if cash is not None:
				if cash["steps"] == 0:
					c = Move.carpet(cash["direction"], cash["roll"])
					if bs.is_valid_move(c):
						return self._record_action(c)
				else:
					d = next_step_toward(bs, start, cash["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							return self._record_action(c)

		# 7) Controlled belief-guided search only when it is likely to pay off.
		if self.search_cd == 0 and time_left_seconds > 18.0 and enemy_threat <= 1 and turns_left > 14 and short_cash is None:
			best_i = int(np.argmax(belief))
			best_p = float(belief[best_i])
			sorted_belief = np.sort(belief)
			second_p = float(sorted_belief[-2]) if sorted_belief.size > 1 else 0.0
			if best_p >= 0.48 and (best_p - second_p) >= 0.14 and mobility_now >= 3:
				self.search_cd = 3
				return self._record_action(Move.search(loc_i(best_i)))

		# 8) Alpha-beta tactical move.
		move = self._best_move_ab(bs, belief, time_left_seconds, score_gap)
		if move is not None and bs.is_valid_move(move):
			return self._record_action(move)

		# 9) Fallback.
		moves = bs.get_valid_moves(exclude_search=True)
		if moves:
			best_fallback = max(
				moves,
				key=lambda mv: (immediate_point_delta(bs, mv), move_heuristic(bs, mv, belief, self.q_type)),
			)
			if bs.is_valid_move(best_fallback):
				return self._record_action(best_fallback)
			for m in moves:
				if bs.is_valid_move(m):
					return self._record_action(m)

		return self._record_action(Move.search((0, 0)))
