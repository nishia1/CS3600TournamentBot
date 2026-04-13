from collections.abc import Callable
import math
import numpy as np

from game.move import Move
from game.enums import Cell, Direction, MoveType, BOARD_SIZE, CARPET_POINTS_TABLE


DIRS = list(Direction)
DELTA = {
	Direction.UP: (0, -1),
	Direction.DOWN: (0, 1),
	Direction.LEFT: (-1, 0),
	Direction.RIGHT: (1, 0),
}

CONTEST_MIN_POINTS = 4
OPP_THREAT_WEIGHT = 3.2
ROOT_OPP_THREAT_WEIGHT = 2.4
ENDGAME_CONVERT_TURNS = 12
ENDGAME_ROUTE_MIN_POINTS = 3
NEARBY_CARPET_STEPS = 1
NEARBY_CARPET_MIN_POINTS = 1
CLUTCH_AB_DEPTH = 4
CLUTCH_TURNS = 10
CLUTCH_GAP = 8
TRAP_PRIME_ROOT_PENALTY = 10.0
NO_PRIME_MOBILITY_CUTOFF = 1
NO_PRIME_CARPET_EXCEPTION = 4
LOW_TIME_GREEDY_SECONDS = 8.0
LOW_TIME_SHALLOW_SECONDS = 20.0


def step_xy(x, y, d):
	dx, dy = DELTA[d]
	return x + dx, y + dy


def primed_run(bs, x, y, d):
	run = 0
	cx, cy = x, y
	while True:
		cx, cy = step_xy(cx, cy, d)
		if not bs.is_valid_cell((cx, cy)):
			break
		if bs.get_cell((cx, cy)) == Cell.PRIMED:
			run += 1
		else:
			break
	return run


def space_run(bs, x, y, d):
	ox, oy = bs.opponent_worker.get_location()
	run = 0
	cx, cy = x, y
	while True:
		cx, cy = step_xy(cx, cy, d)
		if not bs.is_valid_cell((cx, cy)):
			break
		if (cx, cy) == (ox, oy):
			break
		if bs.get_cell((cx, cy)) == Cell.SPACE:
			run += 1
		else:
			break
	return run


def immediate_carpet_value(bs):
	x, y = bs.player_worker.get_location()
	best = 0
	for d in DIRS:
		r = primed_run(bs, x, y, d)
		if r > 0:
			best = max(best, CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0))
	return best


def immediate_carpet_value_enemy(bs):
	# Estimate enemy's immediate carpet value from current board state.
	x, y = bs.opponent_worker.get_location()
	px, py = bs.player_worker.get_location()
	best = 0
	for d in DIRS:
		run = 0
		cx, cy = x, y
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


def future_prime_potential(bs):
	x, y = bs.player_worker.get_location()
	best = 0
	if bs.get_cell((x, y)) != Cell.SPACE:
		return 0
	for d in DIRS:
		r = space_run(bs, x, y, d)
		if r >= 2:
			best = max(best, CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0))
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


def best_reachable_carpet_option(bs, start, turns_left, min_points=ENDGAME_ROUTE_MIN_POINTS):
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
		score = o["points"] * 4.0 + o["roll"] * 0.8 - o["steps"] * 1.1 + race_margin * 0.7
		if score > best_score:
			best_score = score
			best = o
	return best


def best_nearby_carpet_option(bs, start, turns_left):
	"""Prioritize very close carpet cash-outs to avoid leaving free points nearby."""
	options = all_carpet_options(bs, start)
	if not options:
		return None

	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < NEARBY_CARPET_MIN_POINTS:
			continue
		if o["steps"] > NEARBY_CARPET_STEPS:
			continue
		if turns_left < o["steps"] + 1:
			continue

		# Very strong preference for immediate conversion, then one-step conversions.
		score = o["points"] * 6.0 + o["roll"] * 1.2 - o["steps"] * 3.0
		if o["steps"] == 0:
			score += 5.0
		if score > best_score:
			best_score = score
			best = o

	return best


def best_contested_carpet_option(bs, start):
	options = all_carpet_options(bs, start)
	if not options:
		return None
	ox, oy = bs.opponent_worker.get_location()
	opp_dist = bfs_dist_avoiding(bs, (ox, oy), start)
	best = None
	best_score = -1e9
	for o in options:
		if o["points"] < CONTEST_MIN_POINTS:
			continue
		us = o["steps"]
		them = opp_dist.get(o["cell"], 99)
		if us > them + 1:
			continue
		race = them - us
		s = o["points"] * 4.5 + o["roll"] * 0.8 + race * 1.1 - us * 0.7
		if s > best_score:
			best_score = s
			best = o
	return best


def evaluate(bs):
	# Board is evaluated from current player perspective.
	my_pts = bs.player_worker.get_points()
	op_pts = bs.opponent_worker.get_points()
	score = 8.0 * (my_pts - op_pts)

	score += 2.5 * immediate_carpet_value(bs)
	score -= OPP_THREAT_WEIGHT * immediate_carpet_value_enemy(bs)
	score += 1.0 * future_prime_potential(bs)

	px, py = bs.player_worker.get_location()
	ox, oy = bs.opponent_worker.get_location()
	score += 0.2 * (abs(px - ox) + abs(py - oy))
	score += 0.6 * open_neighbors(bs, (px, py))

	# Mobility term to avoid self-trapping and pacing in tight corridors.
	mobility = len(bs.get_valid_moves(exclude_search=True))
	score += 0.5 * mobility
	if mobility <= 2:
		score -= 3.5
	if mobility <= 1:
		score -= 8.0
	return score


def move_heuristic(bs, m, action_bias):
	# Fast ordering signal for alpha-beta efficiency.
	val = 0.35 * float(action_bias.get(m.move_type, 0.0))
	if m.move_type == MoveType.CARPET:
		val += 18.0 + 2.0 * m.roll_length
	elif m.move_type == MoveType.PRIME:
		px, py = bs.player_worker.get_location()
		sr = space_run(bs, px, py, m.direction)
		val += 1.8 + 0.7 * sr
	else:
		val += 1.0
	return val


def top_ordered_moves(bs, action_bias, max_branch):
	moves = bs.get_valid_moves(exclude_search=True)
	if not moves:
		return []
	ordered = sorted(moves, key=lambda mv: move_heuristic(bs, mv, action_bias), reverse=True)
	return ordered[:max_branch]


class PlayerAgent:
	def __init__(self, board, transition_matrix=None, time_left: Callable = None):
		self.search_cd = 0

		# Lightweight RL bandit over move types.
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

	def commentate(self):
		return "joey: alpha-beta + rl-bias"

	def _update_rl(self, bs):
		my_points = bs.player_worker.get_points()
		opp_points = bs.opponent_worker.get_points()
		reward = (my_points - self.prev_points) - 0.5 * (opp_points - self.prev_opp_points)
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
		# Next turn is opponent: flip perspective for minimax recursion.
		child.reverse_perspective()
		return child

	def _negamax(self, bs, depth, alpha, beta, max_branch):
		if depth == 0 or bs.is_game_over():
			return evaluate(bs)

		moves = top_ordered_moves(bs, self.q_type, max_branch)
		if not moves:
			return evaluate(bs)

		best = -1e18
		for m in moves:
			child = self._forecast_next(bs, m)
			if child is None:
				continue
			val = -self._negamax(child, depth - 1, -beta, -alpha, max_branch)
			if val > best:
				best = val
			alpha = max(alpha, best)
			if alpha >= beta:
				break
		return best

	def _best_move_ab(self, bs, time_left_seconds, score_gap):
		turns_left = bs.player_worker.turns_left

		depth = 3
		max_branch = 10 if turns_left > 8 else 12
		if time_left_seconds <= LOW_TIME_SHALLOW_SECONDS:
			depth = 2
			max_branch = 8
		if (turns_left <= CLUTCH_TURNS or abs(score_gap) <= CLUTCH_GAP) and time_left_seconds > 35:
			depth = CLUTCH_AB_DEPTH
			max_branch = 8 if turns_left > 8 else 10

		moves = top_ordered_moves(bs, self.q_type, max_branch)
		if not moves:
			return None

		best_move = None
		best_val = -1e18
		alpha, beta = -1e18, 1e18

		for m in moves:
			board_after = bs.forecast_move(m, check_ok=True)
			if board_after is None:
				continue

			# Hard safety gate: avoid trap-prime moves unless they clearly convert soon.
			if m.move_type == MoveType.PRIME:
				mob_after = len(board_after.get_valid_moves(exclude_search=True))
				my_near_carpet = immediate_carpet_value(board_after)
				if mob_after <= NO_PRIME_MOBILITY_CUTOFF and my_near_carpet < NO_PRIME_CARPET_EXCEPTION:
					continue

			opp_threat = immediate_carpet_value_enemy(board_after)

			# If trailing, bias less toward threat-avoidance to seek comeback lines.
			if score_gap <= -5:
				root_threat_weight = ROOT_OPP_THREAT_WEIGHT * 0.65
			elif score_gap >= 5:
				root_threat_weight = ROOT_OPP_THREAT_WEIGHT * 1.25
			else:
				root_threat_weight = ROOT_OPP_THREAT_WEIGHT

			child = board_after
			child.reverse_perspective()
			val = -self._negamax(child, depth - 1, -beta, -alpha, max_branch)
			val -= root_threat_weight * opp_threat

			# Extra root safeguard: avoid prime moves that trap us without near conversion.
			if m.move_type == MoveType.PRIME:
				mob_after = len(board_after.get_valid_moves(exclude_search=True))
				my_near_carpet = immediate_carpet_value(board_after)
				if mob_after <= 2 and my_near_carpet <= 0:
					val -= TRAP_PRIME_ROOT_PENALTY
			if val > best_val:
				best_val = val
				best_move = m
			alpha = max(alpha, best_val)

		return best_move

	def play(self, bs, sensor, time_left):
		# RL update from realized reward since our last action.
		self._update_rl(bs)

		# Engine passes time_left as a callable in live matches.
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

		turns_left = bs.player_worker.turns_left
		my_points = bs.player_worker.get_points()
		opp_points = bs.opponent_worker.get_points()
		score_gap = my_points - opp_points

		# Fast local conversion if available.
		px, py = bs.player_worker.get_location()
		best_carpet = None
		best_carpet_pts = -1
		for d in DIRS:
			r = primed_run(bs, px, py, d)
			if r > 0:
				pts = CARPET_POINTS_TABLE.get(min(r, BOARD_SIZE - 1), 0)
				if pts > best_carpet_pts:
					best_carpet_pts = pts
					best_carpet = Move.carpet(d, r)
		if (
			best_carpet is not None
			and bs.is_valid_move(best_carpet)
			and (best_carpet_pts >= 4 or turns_left <= ENDGAME_CONVERT_TURNS)
		):
			self.prev_action_type = best_carpet.move_type
			return best_carpet

		# Hard low-time safety: no deep search, use fastest greedy policy.
		if time_left_seconds <= LOW_TIME_GREEDY_SECONDS:
			start = bs.player_worker.get_location()
			near_cash = best_nearby_carpet_option(bs, start, turns_left)
			if near_cash is not None:
				if near_cash["steps"] == 0:
					c = Move.carpet(near_cash["direction"], near_cash["roll"])
					if bs.is_valid_move(c):
						self.prev_action_type = c.move_type
						return c
				else:
					d = next_step_toward(bs, start, near_cash["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							self.prev_action_type = c.move_type
							return c

			moves = bs.get_valid_moves(exclude_search=True)
			if moves:
				for m in moves:
					if m.move_type == MoveType.CARPET and bs.is_valid_move(m):
						self.prev_action_type = m.move_type
						return m
				for m in moves:
					if m.move_type == MoveType.PLAIN and bs.is_valid_move(m):
						self.prev_action_type = m.move_type
						return m
				self.prev_action_type = moves[0].move_type
				return moves[0]

		# Always secure nearby carpet points before broader planning.
		start = bs.player_worker.get_location()
		near_cash = best_nearby_carpet_option(bs, start, turns_left)
		if near_cash is not None:
			if near_cash["steps"] == 0:
				c = Move.carpet(near_cash["direction"], near_cash["roll"])
				if bs.is_valid_move(c):
					self.prev_action_type = c.move_type
					return c
			else:
				d = next_step_toward(bs, start, near_cash["cell"])
				if d is not None:
					c = Move.plain(d)
					if bs.is_valid_move(c):
						self.prev_action_type = c.move_type
						return c

		# Endgame: route to cashable carpets instead of speculative setup.
		if turns_left <= ENDGAME_CONVERT_TURNS:
			start = bs.player_worker.get_location()
			cash = best_reachable_carpet_option(bs, start, turns_left)
			if cash is not None:
				if cash["steps"] == 0:
					c = Move.carpet(cash["direction"], cash["roll"])
					if bs.is_valid_move(c):
						self.prev_action_type = c.move_type
						return c
				else:
					d = next_step_toward(bs, start, cash["cell"])
					if d is not None:
						c = Move.plain(d)
						if bs.is_valid_move(c):
							self.prev_action_type = c.move_type
							return c

		# Contested conversion: defend/steal high-value carpet lanes before search.
		start = bs.player_worker.get_location()
		contested = best_contested_carpet_option(bs, start)
		if contested is not None:
			if contested["steps"] == 0:
				c = Move.carpet(contested["direction"], contested["roll"])
				if bs.is_valid_move(c):
					self.prev_action_type = c.move_type
					return c
			else:
				d = next_step_toward(bs, start, contested["cell"])
				if d is not None:
					c = Move.plain(d)
					if bs.is_valid_move(c):
						self.prev_action_type = c.move_type
						return c

		move = self._best_move_ab(bs, time_left_seconds, score_gap)
		if move is not None and bs.is_valid_move(move):
			self.prev_action_type = move.move_type
			return move

		# Very rare fallback.
		moves = bs.get_valid_moves(exclude_search=True)
		if moves:
			for m in moves:
				if m.move_type == MoveType.CARPET and bs.is_valid_move(m):
					self.prev_action_type = m.move_type
					return m
			self.prev_action_type = moves[0].move_type
			return moves[0]

		self.prev_action_type = MoveType.SEARCH
		return Move.search((0, 0))
