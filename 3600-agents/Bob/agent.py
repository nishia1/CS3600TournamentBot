from collections.abc import Callable
from collections import deque
from typing import List, Set, Tuple
import random
import time
import jax.numpy as jnp

from game.board import Board
from game.enums import Cell, BOARD_SIZE, Noise, MoveType
from game.move import Move
from game.rat import NOISE_PROBS, DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS, manhattan_distance

CARPET_POINTS = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

class RatHMM:
    def __init__(self, board, transition_matrix):
        self.T = jnp.array(transition_matrix)
        self.n = BOARD_SIZE * BOARD_SIZE
        self.val = jnp.zeros(self.n)
        self.val = self.val.at[0].set(1.0)
        self.val = jnp.linalg.matrix_power(self.T, 1000)[0]
        xs = jnp.arange(BOARD_SIZE)
        ys = jnp.arange(BOARD_SIZE)
        grid_x, grid_y = jnp.meshgrid(xs, ys)
        self.positions = jnp.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        self.cell_types = jnp.array(
            [board.get_cell((int(x), int(y))).value for x, y in self.positions]
        )
        self.noise_prob_table = jnp.array([NOISE_PROBS[Cell(i)] for i in range(len(NOISE_PROBS))])
        self.dist_probs = jnp.array(DISTANCE_ERROR_PROBS)
        self.dist_offsets = jnp.array(DISTANCE_ERROR_OFFSETS)

    def update(self, sensor_data, board):
        noise, observed_dist = sensor_data
        self.val = self.val @ self.T
        self.cell_types = jnp.array([board.get_cell((int(x), int(y))).value for x, y in self.positions])
        likelihood = self.compute_likelihood(noise, observed_dist, board)
        self.val = self.val * likelihood
        total = self.val.sum()
        self.val = jnp.where(total > 0, self.val / total, jnp.ones_like(self.val) / self.n)

    def compute_likelihood(self, noise, observed_dist, board):
        noise_probs = self.noise_prob_table[self.cell_types, noise.value]
        wx, wy = board.player_worker.get_location()
        worker_pos = jnp.array([wx, wy])
        dists = jnp.abs(self.positions - worker_pos).sum(axis=1)
        possible_obs = dists[:, None] + self.dist_offsets
        possible_obs = jnp.maximum(possible_obs, 0)
        matches = (possible_obs == observed_dist)
        dist_probs = (matches * self.dist_probs).sum(axis=1)
        return noise_probs * dist_probs

    def best_guess(self):
        idx = int(jnp.argmax(self.val))
        return (idx % BOARD_SIZE, idx // BOARD_SIZE), float(self.val[idx])

    def expected_value_of_search(self, pos):
        x, y = pos
        idx = y * BOARD_SIZE + x
        p = float(self.val[idx])
        return 6.0 * p - 2.0

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_hmm = RatHMM(board, transition_matrix)

        self.last_positions = deque(maxlen=6)
        self.visited = set()
        self.last_move_dir = None

        self.turn_count = 0
        self.wrong_guess_cooldown = 0

    def longest_primed_run_from(self, board, start, direction):
        dx, dy = direction
        x, y = start
        length = 0
        x += dx
        y += dy
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if board.get_cell((x, y)) == Cell.PRIMED:
                length += 1
                x += dx
                y += dy
            else:
                break
        return length

    def best_carpet_value_from(self, board, pos):
        best = 0
        for d in [(1,0),(-1,0),(0,1),(0,-1)]:
            run = self.longest_primed_run_from(board, pos, d)
            best = max(best, CARPET_POINTS.get(run, 21))
        return best

    def primed_cluster_value(self, board, px, py):
        score = 0.0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    dist = abs(px - x) + abs(py - y)
                    for d in [(1,0),(0,1)]:
                        f = self.longest_primed_run_from(board, (x,y), d)
                        b = self.longest_primed_run_from(board, (x,y), (-d[0], -d[1]))
                        run = 1 + max(f, b)
                        if run >= 2:
                            val = CARPET_POINTS.get(min(run, 7), 21)
                            score += val * 0.15 + max(0, 6 - dist) * 0.2
        return score

    def evaluate(self, board):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        mobility = (len(board.get_valid_moves()) - len(board.get_valid_moves(enemy = True))) * 0.5
        # px, py = board.player_worker.get_location()
        # ox, oy = board.opponent_worker.get_location()
        # # carpet (reduced dominance)
        # score += self.primed_cluster_value(board, px, py) * 0.28
        # # mobility advantage
        # my_moves = len(board.get_valid_moves())
        # rev = board.get_copy()
        # rev.reverse_perspective()
        # opp_moves = len(rev.get_valid_moves())
        # score += 0.3 * (my_moves - opp_moves)
        # # distance pressure
        # score -= 0.05 * (abs(px - ox) + abs(py - oy))
        # # soft loop detection (NOT over-penalizing movement)
        # if len(self.last_positions) >= 4:
        #     if self.last_positions[-1] == self.last_positions[-3]:
        #         score -= 0.8
        # # movement encouragement (fixes freezing)
        # if len(self.last_positions) >= 2:
        #     if self.last_positions[-1] != self.last_positions[-2]:
        #         score += 0.6
        # # exploration reward (balanced)
        # if (px, py) not in self.visited:
        #     score += 1.2
        # else:
        #     score -= 0.05
        return score + mobility
    
    def move_score(self, board, move):
        if move.move_type == MoveType.CARPET:
            nb = board.forecast_move(move)
            if nb:
                gain = nb.player_worker.get_points() - board.player_worker.get_points()
                return 100 + gain * 10
            return 50
        if move.move_type == MoveType.PRIME:
            nb = board.forecast_move(move)
            if nb:
                x, y = nb.player_worker.get_location()
                return 20 + self.best_carpet_value_from(board, (x, y)) * 5
        if move.move_type == MoveType.PLAIN:
            nb = board.forecast_move(move)
            if nb:
                x, y = nb.player_worker.get_location()
                return 5 + self.best_carpet_value_from(board, (x, y))
        if move.move_type == MoveType.SEARCH:
            return -10
        return 0

    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "ragebait"

    def negamax(self, board, depth, alpha, beta, color):
        if depth == 0 or board.is_game_over():
            return color * self.evaluate(board), None
        moves = board.get_valid_moves()
        if not moves:
            return color * self.evaluate(board), None
        moves.sort(key=lambda m: self.move_score(board, m), reverse=True)
        best_val = -float('inf')
        best_move = moves[0]
        for move in moves:
            nb = board.forecast_move(move)
            nb.reverse_perspective()
            if not nb:
                continue
            val, _ = self.negamax(nb, depth - 1, -beta, -alpha, -color)
            val = -val
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        return best_val, best_move

    # def iterative_deepening(self, board, time_budget=1.5):
    #     start = time.time()
    #     best_move = None
    #     best_val = -float('inf')
    #     for depth in range(1, 6):
    #         if time.time() - start > time_budget * 0.65:
    #             break
    #         val, move = self.negamax(board, depth, -float('inf'), float('inf'), 1)
    #         if move:
    #             best_move = move
    #             best_val = val
    #         if time.time() - start > time_budget * 0.9:
    #             break
    #     return best_val, best_move

    def play(self, board: Board, sensor_data: Tuple, time_left: Callable):
        # self.turn_count += 1
        # self.rat_hmm.update(sensor_data, board)
        # rat_pos, _ = self.rat_hmm.best_guess()
        # rat_ev = self.rat_hmm.expected_value_of_search(rat_pos)
        # pos = board.player_worker.get_location()
        # self.last_positions.append(pos)
        # self.visited.add(pos)
        # # direction tracking
        # if len(self.last_positions) >= 2:
        #     a, b = self.last_positions[-2], self.last_positions[-1]
        #     self.last_move_dir = (b[0] - a[0], b[1] - a[1])
        # moves = board.get_valid_moves()
        # # best immediate carpet
        # best_carpet = None
        # best_gain = 0
        # for m in moves:
        return self.negamax(board, 6, -float('inf'), float('inf'), 1)[1]
        # for m in moves:
        #     if m.move_type == MoveType.CARPET:
        #         nb = board.forecast_move(m)
        #         if nb:
        #             gain = nb.player_worker.get_points() - board.player_worker.get_points()
        #             if gain > best_gain:
        #                 best_gain = gain
        #                 best_carpet = m
        # if best_carpet and best_gain >= 4:
        #     return best_carpet
        # # if rat_ev > 0 and self.wrong_guess_cooldown == 0 and rat_ev > best_gain:
        # #     return Move.search(rat_pos)
        # best_val, best_move = self.iterative_deepening(board)
        # if not best_move:
        #     non_search = [m for m in moves if m.move_type != MoveType.SEARCH]
        #     if non_search:
        #         return max(non_search, key=lambda m: self.move_score(board, m))
        #     # return Move.search(rat_pos)
        # # prevent useless plain-stall
        # if best_move.move_type == MoveType.PLAIN:
        #     primes = [m for m in moves if m.move_type == MoveType.PRIME]
        #     if primes:
        #         best_move = max(primes, key=lambda m: self.move_score(board, m))
        # return best_move