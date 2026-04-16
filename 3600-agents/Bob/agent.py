from collections.abc import Callable
from typing import List, Set, Tuple
import random
import jax.numpy as jnp

from game.board import Board
from game.enums import Cell, BOARD_SIZE, Noise, MoveType
from game.move import Move
from game.rat import NOISE_PROBS, DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS, manhattan_distance

# negamax with alpha beta pruning
# rat hmm function

class RatHMM:
    def __init__(self, board, transition_matrix):
        self.T = jnp.array(transition_matrix)
        self.n = BOARD_SIZE * BOARD_SIZE  # 64
        self.val = jnp.zeros(self.n)
        self.val = self.val.at[0].set(1.0)
        self.val = jnp.linalg.matrix_power(self.T, 1000)[0]
        # precomp all possible x y positions
        xs = jnp.arange(BOARD_SIZE)
        ys = jnp.arange(BOARD_SIZE)
        grid_x, grid_y = jnp.meshgrid(xs, ys)
        self.positions = jnp.stack([grid_x.flatten(), grid_y.flatten()], axis=1)  # (64,2)
        # precompute cell types as ints
        self.cell_types = jnp.array([board.get_cell((int(x), int(y))).value for x, y in self.positions])
        # convert NOISE_PROBS to arr (num_cell_types, 3) 
        self.noise_prob_table = jnp.array([NOISE_PROBS[Cell(i)] for i in range(len(NOISE_PROBS))])
        self.dist_probs = jnp.array(DISTANCE_ERROR_PROBS)
        self.dist_offsets = jnp.array(DISTANCE_ERROR_OFFSETS)

    def update(self, sensor_data, board):
        noise, observed_dist = sensor_data
        self.val = self.val @ self.T
        likelihood = self.compute_likelihood(noise, observed_dist, board)
        self.val = self.val * likelihood
        total = self.val.sum()
        self.val = jnp.where(total > 0, self.val / total, jnp.ones_like(self.val) / self.n)

    def compute_likelihood(self, noise, observed_dist, board):
        noise_probs = self.noise_prob_table[self.cell_types, noise.value]  # (64,)
        worker_x, worker_y = board.player_worker.get_location()
        worker_pos = jnp.array([worker_x, worker_y])
        # compute all manhattan distances
        dists = jnp.abs(self.positions - worker_pos).sum(axis=1)  # (64,)
        # compute possible observed distances for each offset
        possible_obs = dists[:, None] + self.dist_offsets  # (64, 4)
        # clamp to >= 0
        possible_obs = jnp.maximum(possible_obs, 0)
        # compare with observed distance
        matches = (possible_obs == observed_dist)  # (64, 4)
        # pick correct probability
        dist_probs = (matches * self.dist_probs).sum(axis=1)  # (64,)
        return noise_probs * dist_probs
    
    def likelycell(self):
        idx = int(jnp.argmax(self.val))
        x = idx % BOARD_SIZE
        y = idx // BOARD_SIZE
        return (x, y)


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """
    def evaluate(self, board):
        score = board.player_worker.get_points() - board.opponent_worker.get_points()

        px, py = board.player_worker.get_location()

        primed = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board.get_cell((x, y)) == Cell.PRIMED:
                    primed += 1
                    dist = abs(px - x) + abs(py - y)
                    score += 0.3 * max(0, 5 - dist)

        myMoves = len(board.get_valid_moves())
        reverseBoard = board.get_copy()
        reverseBoard.reverse_perspective()
        oppMoves = len(reverseBoard.get_valid_moves())

        return score + 0.5 * (myMoves - oppMoves)

    
    def heuristic(self, board, move):
        # want to decide if a move is good or not
        # need to keep a few things in mind
        # 1. difference between my score and the opponents score
        # 2. carpet potential close to you, bring them together to allow to stealing from opponent
        # 3. mobility - more options u have, if opponent has more options than u. thats bad. 
        # can det how many valid moves left for u after this vs how many valid moves left for the opponent (reverse board, and get len of their valid moves)
        score = 0
        # --- simulate move ---
        newBoard = board.forecast_move(move)
        if newBoard is None:
            return -float('inf')
        # --- 1. ACTUAL SCORE CHANGE (MOST IMPORTANT) ---
        scoreDiff = newBoard.player_worker.get_points() - board.player_worker.get_points()
        # encourage change in score or board
        if scoreDiff == 0:
            score -= 15 # discourage doing nothing
        score += 5 * scoreDiff   # heavily reward real points
        # --- 2. MOVE TYPE PRIORITY ---
        if move.move_type == MoveType.PLAIN:
            score -= 5   # discourage useless walking
        elif move.move_type == MoveType.PRIME:
            score += 10   # priming is always good earl
        elif move.move_type == MoveType.CARPET:
            score += 20   # carpets are huge
        elif move.move_type == MoveType.SEARCH:
            score += 1   # small bias
        # --- 3. MOBILITY ---
        myMoves = len(newBoard.get_valid_moves())
        reverseBoard = newBoard.get_copy()
        reverseBoard.reverse_perspective()
        oppMoves = len(reverseBoard.get_valid_moves())
        score += 0.1 * (myMoves - oppMoves)
        # --- 4. LIGHT POSITIONAL PRESSURE ---
        px, py = newBoard.player_worker.get_location()
        ox, oy = newBoard.opponent_worker.get_location()
        dist = abs(px - ox) + abs(py - oy)
        score += -0.1 * dist  # slightly prefer being closer
        # reward potential for carpeting
        # reward being near primed tiles (setup for carpets)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if newBoard.get_cell((x, y)) == Cell.PRIMED:
                    dist = abs(px - x) + abs(py - y)
                    score += 0.05 * (5 - dist)
        # discourage staying in the same place
        # discourage staying in same place / useless moves
        old_pos = board.player_worker.get_location()
        new_pos = newBoard.player_worker.get_location()
        if new_pos == old_pos:
            score -= 20
        return score

    def negamax(self, board, depth, alpha, beta, color):
        if depth == 0 or board.is_game_over() or (board.player_worker.turns_left <= 0 and board.opponent_worker.turns_left <= 0):
            return (color * self.evaluate(board), None)
        childNodes = board.get_valid_moves()
        childNodes = sorted(childNodes, key=lambda move: self.heuristic(board, move), reverse=True)
        value = -float('inf')
        bestMove = None
        for move in childNodes:
            newBoard = board.forecast_move(move)
            if newBoard is None:
                continue
            val, _ = self.negamax(newBoard, depth - 1, -beta, -alpha, -color)
            val = -val
            if (val > value):
                bestMove = move
                value = val
            # value = max(value, -self.negamax(newBoard, depth - 1, -beta, -alpha, -color))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return (value, bestMove)
    
    def search(self, board, depth):
        if (depth == 0):
            return self.evaluate(board)
        else:
            moves = board.get_valid_moves()
            for move in moves:
                newBoard =  board.forecast_move(move)
                self.negamax(newBoard, depth - 1, -float('inf'), float('inf'), -1)


    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        # call rat init code
        self.rat_hmm = RatHMM(board, transition_matrix)
        #pass
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "hi trying really hard to not go clinically insane"

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        #moves = board.get_valid_moves()
        #return random.choice(moves)
        moves = board.get_valid_moves()

        # FORCE priming if available (this breaks the loop behavior)
        prime_moves = [m for m in moves if m.move_type == MoveType.PRIME]
        self.rat_hmm.update(sensor_data, board)
        likely_pos = self.rat_hmm.likelycell()
        # run negamax to find best move, using wikipedia implementation
        prob = float(jnp.max(self.rat_hmm.val))
        likely_pos = self.rat_hmm.likelycell()
        # only guess if confident
        if prob > 0.8:
            return Move.search(likely_pos)
        best_move = None
        best_value = -float('inf')
        for move in board.get_valid_moves():
            newBoard = board.forecast_move(move)
            if newBoard is None:
                continue
            value, _ = self.negamax(newBoard, 4, -float('inf'), float('inf'), -1)
            value = -value
            if value > best_value:
                best_value = value
                best_move = move
        # # check if rat has better value instead
        # rat_pos = self.rat_hmm.likelycell()
        # rat_score = float(jnp.max(self.rat_hmm.val)) * 4
        # if rat_score > best_value:
        #     # we should guess the position of the rat
        #     best_move = Move.search(rat_pos)
        if prime_moves:
            # only force if negamax gives garbage
            if best_move is None or best_value < 1:
                return random.choice(prime_moves)
        return best_move
