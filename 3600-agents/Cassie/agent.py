from collections.abc import Callable
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

MAX_BRANCH = 14
LOW_TIME_GREEDY_SECONDS = 10.0
MID_TIME_SECONDS = 40.0
ENDGAME_TURNS = 12
THREAT_CARPET_POINTS = 4


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
			self.b /= self.b.sum()
		else:
			mask = np.array([
				0.0 if bs.get_cell(loc_i(i)) == Cell.BLOCKED else 1.0
				for i in range(N)
			], dtype=np.float64)
			self.b = mask / mask.sum()


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


def space_run(bs, x, y, d):
	ox, oy = bs.opponent_worker.get_location()
	r = 0
	cx, cy = x, y
	while True:
		cx, cy = step_xy(cx, cy, d)
		if not bs.is_valid_cell((cx, cy)):
			break
		if (cx, cy) == (ox, oy):
			break
		if bs.get_cell((cx, cy)) == Cell.SPACE:
			r += 1
		else:
			break
	return r


def standard_passable(bs, nx, ny):
	if not bs.is_valid_cell((nx, ny)):
		return False
	if (nx, ny) == bs.opponent_worker.get_location():
		return False
	c = bs.get_cell((nx, ny))
	return c not in (Cell.BLOCKED, Cell.PRIMED)


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
			options.append({
				"cell": (x, y),
				"steps": steps,
				"direction": d,
				"roll": r,
				"points": points,
			})
	return options


def best_contested_carpet_option(bs, start):
	options = all_carpet_options(bs, start)
	if not options:
		return None

	ox, oy = bs.opponent_worker.get_location()
	opp_dist = bfs_dist_avoiding(bs, (ox, oy), start)

	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < THREAT_CARPET_POINTS:
			continue
		us = o["steps"]
		them = opp_dist.get(o["cell"], 99)
		if us > them + 1:
			continue
		race = them - us
		s = o["points"] * 5.0 + o["roll"] * 0.8 + race * 1.3 - us * 0.9
		if s > best_score:
			best_score = s
			best = o
	return best


def best_nearby_carpet_option(bs, start, turns_left):
	options = all_carpet_options(bs, start)
	if not options:
		return None

	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < 2:
			continue
		if o["steps"] > 1:
			continue
		if turns_left < o["steps"] + 1:
			continue

		s = o["points"] * 6.0 + o["roll"] * 1.1 - o["steps"] * 2.8
		if o["steps"] == 0:
			s += 4.0
		if s > best_score:
			best_score = s
			best = o
	return best


def option_to_action(bs, start, opt):
	if opt is None:
		return None
	if start == opt["cell"]:
		m = Move.carpet(opt["direction"], opt["roll"])
		return m if bs.is_valid_move(m) else None
	d = next_step_toward(bs, start, opt["cell"])
	if d is None:
		return None
	m = Move.plain(d)
	return m if bs.is_valid_move(m) else None


def cell_potential(bs, x, y):
	if not bs.is_valid_cell((x, y)):
		return -1e9
	c = bs.get_cell((x, y))
	if c == Cell.BLOCKED:
		return -1e9

	p = 0.0
	if c == Cell.SPACE:
		p += 0.8
	elif c == Cell.PRIMED:
		p += 1.3
	elif c == Cell.CARPET:
		p += 1.0

	open_n = 0
	best_lane = 0
	for d in DIRS:
		nx, ny = step_xy(x, y, d)
		if bs.is_valid_cell((nx, ny)) and bs.get_cell((nx, ny)) != Cell.BLOCKED:
			open_n += 1
		best_lane = max(best_lane, space_run(bs, x, y, d))
	p += 0.7 * open_n
	p += 0.5 * best_lane

	edge_dist = min(x, y, BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y)
	p += 0.25 * min(edge_dist, 2)
	return p


def immediate_carpet_move(bs):
	x, y = bs.player_worker.get_location()
	best = None
	best_pts = -1
	for d in DIRS:
		r = primed_run(bs, x, y, d)
		if r < 1:
			continue
		pts = CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0)
		if pts > best_pts:
			best_pts = pts
			best = Move.carpet(d, r)
	return best, best_pts


def evaluate(bs, belief):
	my_pts = bs.player_worker.get_points()
	op_pts = bs.opponent_worker.get_points()
	score = 9.0 * (my_pts - op_pts)

	# Expected positional value over belief.
	exp_val = 0.0
	for i in range(N):
		x, y = loc_i(i)
		b = float(belief[i])
		if b <= 1e-9:
			continue
		exp_val += b * cell_potential(bs, x, y)
	score += 3.0 * exp_val

	# Immediate tactical conversion and threat.
	my_carpet = 0
	mx, my = bs.player_worker.get_location()
	for d in DIRS:
		my_carpet = max(my_carpet, CARPET_POINTS_TABLE.get(min(primed_run(bs, mx, my, d), BOARD_SIZE - 1), 0))

	ox, oy = bs.opponent_worker.get_location()
	op_carpet = 0
	for d in DIRS:
		op_carpet = max(op_carpet, CARPET_POINTS_TABLE.get(min(primed_run(bs, ox, oy, d), BOARD_SIZE - 1), 0))

	turns_left = bs.player_worker.turns_left
	threat_w = 2.0 if turns_left > ENDGAME_TURNS else 2.8
	score += 2.2 * my_carpet
	score -= threat_w * op_carpet

	# Belief focus bonus encourages reaching high-value likely rat regions.
	dm = bfs_dist(bs, (mx, my))
	best_i = int(np.argmax(belief))
	best_p = float(belief[best_i])
	bx, by = loc_i(best_i)
	if (bx, by) in dm:
		score += 2.4 * best_p * (cell_potential(bs, bx, by) / (1.0 + dm[(bx, by)]))

	score += 0.6 * open_neighbors(bs, (mx, my))

	score += 0.7 * len(bs.get_valid_moves(exclude_search=True))
	return score


def order_moves(bs, belief, moves):
	px, py = bs.player_worker.get_location()
	scored = []
	for m in moves:
		s = 0.0
		if m.move_type == MoveType.CARPET:
			s += 22.0 + 2.0 * m.roll_length
		elif m.move_type == MoveType.PRIME:
			sr = space_run(bs, px, py, m.direction)
			s += 5.0 + 1.3 * sr
			if open_neighbors(bs, (px, py)) <= 1:
				s -= 6.0
		elif m.move_type == MoveType.PLAIN:
			nx, ny = step_xy(px, py, m.direction)
			if bs.is_valid_cell((nx, ny)):
				s += cell_potential(bs, nx, ny)
				s += 3.0 * float(belief[idx_xy(nx, ny)])
				s += 0.5 * open_neighbors(bs, (nx, ny))
		scored.append((s, m))
	scored.sort(key=lambda t: t[0], reverse=True)
	return [m for _, m in scored]


class PlayerAgent:
	def __init__(self, board, transition_matrix=None, time_left: Callable = None):
		if transition_matrix is not None:
			T = np.array(transition_matrix, dtype=np.float64)
		else:
			T = np.full((N, N), 1.0 / N)
		self.hmm = RatHMM(T)
		self.search_cd = 0

	def commentate(self):
		return "cassie: hmm + expectiminimax"

	def _negamax(self, bs, depth, alpha, beta, belief, deadline, max_branch):
		if depth == 0 or bs.is_game_over() or time.perf_counter() >= deadline:
			return evaluate(bs, belief)

		moves = bs.get_valid_moves(exclude_search=True)
		if not moves:
			return evaluate(bs, belief)

		ordered = order_moves(bs, belief, moves)[:max_branch]
		next_belief = belief @ self.hmm.T

		best = -1e18
		for m in ordered:
			child = bs.forecast_move(m, check_ok=True)
			if child is None:
				continue
			child.reverse_perspective()
			val = -self._negamax(child, depth - 1, -beta, -alpha, next_belief, deadline, max_branch)
			if val > best:
				best = val
			alpha = max(alpha, best)
			if alpha >= beta:
				break
		return best

	def _best_move_search(self, bs, belief, time_left_seconds, score_gap):
		turns_left = bs.player_worker.turns_left
		if time_left_seconds <= LOW_TIME_GREEDY_SECONDS:
			return None

		depth = 2
		max_branch = MAX_BRANCH
		if time_left_seconds > MID_TIME_SECONDS and turns_left <= 14:
			depth = 3
			max_branch = 10
		if time_left_seconds > 30.0 and (turns_left <= 10 or abs(score_gap) <= 8):
			depth = 3
			max_branch = 10

		budget = 0.18 if time_left_seconds > MID_TIME_SECONDS else 0.09
		deadline = time.perf_counter() + budget

		moves = bs.get_valid_moves(exclude_search=True)
		if not moves:
			return None
		ordered = order_moves(bs, belief, moves)[:max_branch]

		best_move = None
		best_val = -1e18
		alpha, beta = -1e18, 1e18
		next_belief = belief @ self.hmm.T

		for m in ordered:
			if time.perf_counter() >= deadline:
				break
			child = bs.forecast_move(m, check_ok=True)
			if child is None:
				continue
			child.reverse_perspective()
			val = -self._negamax(child, depth - 1, -beta, -alpha, next_belief, deadline, max_branch)
			if val > best_val:
				best_val = val
				best_move = m
			alpha = max(alpha, best_val)
		return best_move

	def play(self, bs, sensor, time_left):
		noise_idx, dist = sensor
		noise_idx = int(noise_idx)

		if callable(time_left):
			try:
				tl = float(time_left())
			except Exception:
				tl = 0.0
		else:
			try:
				tl = float(time_left)
			except Exception:
				tl = 0.0

		if self.search_cd > 0:
			self.search_cd -= 1

		self.hmm.predict()
		self.hmm.update(bs, noise_idx, dist)
		belief = self.hmm.b.copy()

		start = bs.player_worker.get_location()
		score_gap = bs.player_worker.get_points() - bs.opponent_worker.get_points()
		enemy_now = 0
		ox, oy = bs.opponent_worker.get_location()
		for d in DIRS:
			enemy_now = max(enemy_now, CARPET_POINTS_TABLE.get(min(primed_run(bs, ox, oy, d), BOARD_SIZE - 1), 0))

		# 1) Immediate carpet conversion first.
		carpet, carpet_pts = immediate_carpet_move(bs)
		if carpet is not None and bs.is_valid_move(carpet):
			if carpet_pts >= 2 or bs.player_worker.turns_left <= 12:
				return carpet

		# 1b) Deny strong enemy conversions by racing/stealing contested carpets.
		if enemy_now >= THREAT_CARPET_POINTS:
			deny = best_contested_carpet_option(bs, start)
			a = option_to_action(bs, start, deny)
			if a is not None:
				return a

		# 1c) Nearby conversion routing reduces dropped points.
		near = best_nearby_carpet_option(bs, start, bs.player_worker.turns_left)
		a = option_to_action(bs, start, near)
		if a is not None:
			return a

		# 2) Expectiminimax search move.
		m = self._best_move_search(bs, belief, tl, score_gap)
		if m is not None and bs.is_valid_move(m):
			return m

		# 3) Controlled search usage.
		if self.search_cd == 0 and tl > 6.0:
			best_i = int(np.argmax(belief))
			best_p = float(belief[best_i])
			if best_p > 0.18:
				self.search_cd = 3
				return Move.search(loc_i(best_i))

		# 4) Greedy fallback for low time.
		moves = bs.get_valid_moves(exclude_search=True)
		if moves:
			# Prefer carpet, then plain to keep mobility.
			for mv in moves:
				if mv.move_type == MoveType.CARPET and bs.is_valid_move(mv):
					return mv
			for mv in moves:
				if mv.move_type == MoveType.PLAIN and bs.is_valid_move(mv):
					return mv
			return moves[0]

		return Move.search((0, 0))
