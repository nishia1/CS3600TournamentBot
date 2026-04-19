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

        # Precompute stationary distribution ONCE
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
        if myRat is not None and mySearch:
            # we found the rat, so now there is a new rat searching around
            # check if opp found it or guessed for it
            if oppRat is not None and oppSearched:
                # opponent also found it, so we reset to stationary (new rat)
                self.posterior = self.stationary.copy()
            elif oppRat is not None and not oppSearched:
                # opponent did not find it, so we can be certain the new rat is not where they searched
                idx = int(oppRat[0] * 8 + oppRat[1])
                self.posterior = jnp.where(
                    jnp.arange(self.n) == idx,
                    0.0,
                    self.posterior
                )
            else:
                # we found it and opponent didnt search so reinit everything
                self.posterior = self.stationary.copy()
        elif myRat is not None and not mySearch:
            # tried to find but failed
            # check if opp found it
            if oppRat is not None and oppSearched:
                # opponent found it, so we need to reset
                self.posterior = self.stationary.copy()
            elif oppRat is not None and not oppSearched:
                # opponent also tried but failed, so we can be certain the rat is not where they searched
                idx = int(oppRat[0] * 8 + oppRat[1])
                self.posterior = jnp.where(
                    jnp.arange(self.n) == idx,
                    0.0,
                    self.posterior
                )
                # also wasn't where i searched
                idx = int(myRat[0] * 8 + myRat[1])
                self.posterior = jnp.where(
                    jnp.arange(self.n) == idx,
                    0.0,
                    self.posterior
                )
            else:
                # we failed and opponent didnt search
                # but def not where i searched
                idx = int(myRat[0] * 8 + myRat[1])
                self.posterior = jnp.where(
                    jnp.arange(self.n) == idx,
                    0.0,
                    self.posterior
                )

        noise, observed_dist = sensor_data

        # --- PROPAGATE FORWARD then use as prior ---
        self.posterior = self.posterior @ self.T
        prior = self.posterior.copy()

        # --- NOISE LIKELIHOOD ---
        cell_types = jnp.array([
            board.get_cell((int(x), int(y))).value
            for x, y in self.positions
        ])
        noise_probs = self.noise_prob_table[cell_types, noise.value]

        # --- DISTANCE LIKELIHOOD ---
        wx, wy = board.player_worker.get_location()
        worker_pos = jnp.array([wx, wy])

        true_dists = jnp.abs(self.positions - worker_pos).sum(axis=1)
        delta = observed_dist - true_dists

        dist_probs = jnp.where(delta == -1, 0.12,
                    jnp.where(delta == 0, 0.7,
                    jnp.where(delta == 1, 0.12,
                    jnp.where(delta == 2, 0.06, 0.0))))

        # --- COMBINE ---
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
            run = fwd + bwd  # total run through current position
            best = max(best, CARPET_POINTS.get(min(run, 7), 21))
        return best

    def evaluate(self, board):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()
        mobility = (len(board.get_valid_moves()) - len(board.get_valid_moves(enemy=True))) * 0.5
        carpet = 0.8 * self.best_carpet_value_from(board, board.player_worker.get_location())
        return score + mobility + carpet

    def move_score(self, board, move, tt_move=None):
        # TT move gets highest priority
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
        return "ragebait"

    def negamax(self, board, depth, alpha, beta):
        # --- Transposition table lookup ---
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

        # --- Store in transposition table ---
        if best_val <= original_alpha:
            flag = 'upper'
        elif best_val >= beta:
            flag = 'lower'
        else:
            flag = 'exact'

        if len(self.transposition_table) >= self.TT_SIZE:
            self.transposition_table.popitem(last=False)  # evict oldest
        self.transposition_table[h] = {
            'val': best_val,
            'move': best_move,
            'depth': depth,
            'flag': flag
        }

        return best_val, best_move

    def iterative_deepening(self, board, time_budget):
        start = time.time()
        best_move = None
        best_val = -float('inf')
        for depth in range(1, 20):
            if time.time() - start > time_budget * 0.5:
                break
            val, move = self.negamax(board, depth, -float('inf'), float('inf'))
            if move:
                best_move = move
                best_val = val
            if time.time() - start > time_budget * 0.9:
                break
        return best_val, best_move

    def play(self, board: Board, sensor_data: Tuple, time_left: Callable):
        # 1. Update HMM belief
        self.rat_hmm.update(sensor_data, board)
        rat_pos, confidence = self.rat_hmm.best_guess()
        expectedRat = 6 * confidence - 2

        # 2. Consider searching — only if confidence is high enough to be worth it
        if confidence > 0.6:
            search_move = Move.search(rat_pos)
            search_board = board.forecast_move(search_move)
            # no reverse_perspective — evaluate from our perspective
            opp_val, _ = self.negamax(search_board, 3, -float('inf'), float('inf'))
            if expectedRat > opp_val:
                return Move.search(rat_pos)

        # 3. Scale depth based on remaining time
        remaining = time_left()
        depth = 10 if remaining > 120 else 8

        _, move = self.negamax(board, depth, -float('inf'), float('inf'))
        return move