from collections.abc import Callable
from collections import deque, OrderedDict
from typing import List, Set, Tuple
import random
import time
import numpy as np
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

        self.stationary = jnp.linalg.matrix_power(self.T, 50)[0]

        xs = jnp.arange(BOARD_SIZE)
        ys = jnp.arange(BOARD_SIZE)
        grid_x, grid_y = jnp.meshgrid(xs, ys)
        self.positions = jnp.stack([grid_x.flatten(), grid_y.flatten()], axis=1)

        self.noise_prob_table = jnp.array([
            NOISE_PROBS[Cell(i)] for i in range(len(NOISE_PROBS))
        ])

        self.posterior = self.stationary.copy()

    def update(self, sensor_data, board):
        oppRat, oppSearched = board.opponent_search
        myRat, mySearch = board.player_search
        x = [board.player_search, board.opponent_search]
        for y in x:
            rat, search = y
            if search and rat is not None:
                # found
                self.posterior = self.stationary.copy()
            elif not search and rat is not None:
                # not found
                idx = rat[1] * BOARD_SIZE + rat[0]
                self.posterior = jnp.where(jnp.arange(self.n) == idx, 0.0, self.posterior)
            self.posterior = self.posterior @ self.T

        noise, observed_dist = sensor_data
        prior = self.posterior.copy()

        cell_types = jnp.array([
            board.get_cell((int(x), int(y))).value
            for x, y in self.positions
        ])
        noise_probs = self.noise_prob_table[cell_types, noise.value]

        wx, wy = board.player_worker.get_location()
        worker_pos = jnp.array([wx, wy])

        true_dists = jnp.abs(self.positions - worker_pos).sum(axis=1)
        delta = observed_dist - true_dists

        dist_probs = jnp.where(delta == -1, 0.12,
                    jnp.where(delta == 0, 0.7,
                    jnp.where(delta == 1, 0.12,
                    jnp.where(delta == 2, 0.06, 0.0))))

        likelihood = noise_probs * dist_probs
        posterior = prior * likelihood

        total = posterior.sum()
        self.posterior = jnp.where(
            total > 0,
            posterior / total,
            jnp.ones_like(posterior) / self.n
        )

    def best_guess(self):
        idx = int(jnp.argmax(self.posterior))
        x, y = self.positions[idx]
        return (int(x), int(y)), float(self.posterior[idx])

    def confidence(self):
        return float(jnp.max(self.posterior))


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_hmm = RatHMM(board, transition_matrix)

        self.last_positions = deque(maxlen=6)
        self.visited = set()
        self.last_move_dir = None

        self.turn_count = 0
        self.wrong_guess_cooldown = 0
        self.last_search = None

        # --- Transposition table ---
        self.transposition_table = OrderedDict()
        self.TT_SIZE = 1_000_000

        # --- Zobrist hashing ---
        rng = np.random.default_rng(42)
        self.zobrist_table = rng.integers(
            0, 2**63, size=(BOARD_SIZE, BOARD_SIZE, len(Cell)), dtype=np.int64
        )
        self.zobrist_players = rng.integers(
            0, 2**63, size=(BOARD_SIZE * BOARD_SIZE, 2), dtype=np.int64
        )

    def board_hash(self, board):
        h = np.int64(0)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                cell = board.get_cell((x, y))
                h ^= self.zobrist_table[x, y, cell.value]
        px, py = board.player_worker.get_location()
        ox, oy = board.opponent_worker.get_location()
        h ^= self.zobrist_players[px * BOARD_SIZE + py, 0]
        h ^= self.zobrist_players[ox * BOARD_SIZE + oy, 1]
        return int(h)

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
        for d in [(1,0), (0,1)]:
            fwd = self.longest_primed_run_from(board, pos, d)
            bwd = self.longest_primed_run_from(board, pos, (-d[0], -d[1]))
            run = fwd + bwd
            best = max(best, CARPET_POINTS.get(min(run, 7), 21))
        return best

    def evaluate(self, board):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        mobility = (len(board.get_valid_moves()) - len(board.get_valid_moves(enemy=True))) * 0.5
        carpet = 0.8 * self.best_carpet_value_from(board, board.player_worker.get_location())
        return score + mobility + carpet
        #return score

    def move_score(self, board, move, tt_move=None):
        if tt_move and move == tt_move:
            return 10000
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
                return 20 + self.best_carpet_value_from(nb, (x, y)) * 5
        if move.move_type == MoveType.PLAIN:
            nb = board.forecast_move(move)
            if nb:
                x, y = nb.player_worker.get_location()
                return 5 + self.best_carpet_value_from(nb, (x, y))
        if move.move_type == MoveType.SEARCH:
            return -10
        return 0

    def commentate(self):
        if self.last_search:
            pos, conf = self.last_search
            return f"Last Guess: {pos} @ {conf}"
        return "Never searched"

    def negamax_shallow(self, board, depth, alpha, beta):
        """TT-free shallow search for search decision only"""
        if depth == 0 or board.is_game_over():
            return self.evaluate(board), None
        moves = board.get_valid_moves()
        if not moves:
            return self.evaluate(board), None
        moves.sort(key=lambda m: self.move_score(board, m), reverse=True)
        best_val = -float('inf')
        best_move = moves[0]
        for move in moves:
            nb = board.forecast_move(move)
            if not nb:
                continue
            nb.reverse_perspective()
            val, _ = self.negamax_shallow(nb, depth - 1, -beta, -alpha)
            val = -val
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        return best_val, best_move

    def negamax(self, board, depth, alpha, beta):
        h = self.board_hash(board)
        tt_move = None
        if h in self.transposition_table:
            entry = self.transposition_table[h]
            tt_move = entry.get('move')
            if entry['depth'] >= depth:
                flag = entry['flag']
                val = entry['val']
                if flag == 'exact':
                    return val, tt_move
                elif flag == 'lower':
                    alpha = max(alpha, val)
                elif flag == 'upper':
                    beta = min(beta, val)
                if alpha >= beta:
                    return val, tt_move

        if depth == 0 or board.is_game_over():
            return self.evaluate(board), None

        moves = board.get_valid_moves()
        if not moves:
            return self.evaluate(board), None

        moves.sort(key=lambda m: self.move_score(board, m, tt_move=tt_move), reverse=True)

        best_val = -float('inf')
        best_move = moves[0]
        original_alpha = alpha

        for move in moves:
            nb = board.forecast_move(move)
            if not nb:
                continue
            nb.reverse_perspective()
            val, _ = self.negamax(nb, depth - 1, -beta, -alpha)
            val = -val
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

        if best_val <= original_alpha:
            flag = 'upper'
        elif best_val >= beta:
            flag = 'lower'
        else:
            flag = 'exact'

        if len(self.transposition_table) >= self.TT_SIZE:
            self.transposition_table.popitem(last=False)
        self.transposition_table[h] = {
            'val': best_val,
            'move': best_move,
            'depth': depth,
            'flag': flag
        }

        return best_val, best_move

    def play(self, board: Board, sensor_data: Tuple, time_left: Callable):
        # 1. Update HMM belief
        self.rat_hmm.update(sensor_data, board)
        rat_pos, confidence = self.rat_hmm.best_guess()

        # 2. Scale depth based on remaining time
        remaining = time_left()

        # 3. Get our best move value
        a = board.turn_count % 2
        if a == 0:
            depth = 10 if remaining > 120 else 8
        elif a == 1:
            depth = 9 if remaining > 120 else 7

        value, move = self.negamax(board, depth, -float('inf'), float('inf'))

        # 4. Simulate opponent's best response after we search
        search_move = Move.search(rat_pos)
        search_board = board.forecast_move(search_move)
        search_board.reverse_perspective()
        opp_val, _ = self.negamax_shallow(search_board, 3, -float('inf'), float('inf'))

        # 5. Simulate what opponent knows if we guess wrong —
        #    they learn rat is NOT at rat_pos, so their next best guess confidence
        wrong_posterior = self.rat_hmm.posterior.copy()
        idx = rat_pos[1] * BOARD_SIZE + rat_pos[0]
        wrong_posterior = wrong_posterior.at[idx].set(0.0)
        total = wrong_posterior.sum()
        wrong_posterior = jnp.where(total > 0, wrong_posterior / total, wrong_posterior)
        opp_next_conf = float(jnp.max(wrong_posterior))
        opp_next_ev = 6 * opp_next_conf - 2  # what opponent expects to gain next turn

        # 6. Full expected value of searching:
        #    if correct (prob=confidence): gain 4, opponent gets opp_val
        #    if wrong (prob=1-confidence): lose 2, opponent gains opp_next_ev next turn
        search_ev = confidence * (4 - opp_val) + (1 - confidence) * (-2 - max(opp_next_ev, opp_val))

        # 7. Search if EV beats our best move — no threshold, just pure EV
        if search_ev > value:
            self.last_search = (rat_pos, round(confidence, 3))
            return Move.search(rat_pos)

        return move