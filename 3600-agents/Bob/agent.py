from collections.abc import Callable
from typing import List, Set, Tuple
import random
import jax.numpy as jnp

from game.enums import Cell, BOARD_SIZE
from game import board, move, enums

# negamax with alpha beta pruning
# rat hmm function

class RatHMM:
    def __init__(self, board, transition_matrix):
        self.T = jnp.array(transition_matrix)
        self.n = BOARD_SIZE * BOARD_SIZE  # 64
        self.val = jnp.zeros(self.n)
        self.val = self.val.at[0].set(1.0)
        for _ in range(1000):
            self.val = self.val @ self.T
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
        # to evaluate, decide different between you and the opponent

    
    def heuristic(self, move):
        # want to decide if a move is good or not
        # need to keep a few things in mind
        # 1. difference between my score and the opponents score
        # 2. carpet potential close to you, bring them together to allow to stealing from opponent
        # 3. mobility - more options u have, if opponent has more options than u. thats bad. 
        # can det how many valid moves left for u after this vs how many valid moves left for the opponent (reverse board, and get len of their valid moves)
        scoreDiff = # idk how to do this part
        carpetPotential = # also not sure how to do this part, maybe look at the 4 adjacent cells and see if they are carpets, if they are then add to the potential score
        mobility = len(board.get_valid_moves())
        reverseBoard = board.reverse_perspective()
        opponentMobility = len(reverseBoard.get_valid_moves())
        mobilityScore = mobility - opponentMobility
        return scoreDiff + carpetPotential + mobilityScore

    def negamax(board, depth, alpha, beta, color):
        if (depth == 0 or board.is_game_over()):
            return color * self.evaluate(board)
        childNodes = board.get_valid_moves()
        childNodes = sorted(childNodes, key=lambda move: self.heuristic(move), reverse=True)
        value = -float('inf')
        for move in childNodes:
            newBoard = board.forecast_move(move)
            value = max(value, -negamax(newBoard, depth-1, -beta, -alpha, -color))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    
    def search(self, board, depth):
        if (depth == 0):
            return self.evalute(board);
        else:
            moves = board.get_valid_moves()
            for move in moves:
                newBoard =  board.forecast_move(move)
                negamax(newBoard, depth-1, -float('inf'), float('inf'), -1)


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
        board: board.Board,
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
        self.rat_hmm.update(sensor_data, board)
        likely_pos = self.rat_hmm.likelycell()
        # run negamax to find best move
        best_move = None
        best_value = -float('inf')
        for move in board.get_valid_moves():
            newBoard = board.forecast_move(move)
            value = -self.negamax(newBoard, 3, -float('inf'), float('inf'), -1)
            if value > best_value:
                best_value = value
                best_move = move
        # check if rat has better value instead
        if likely_pos > best_value:
            # we should guess the position of the rat
            best_move = move.Move.guess(likely_pos)
        return best_move
            
