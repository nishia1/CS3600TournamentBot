from collections.abc import Callable
from typing import List, Set, Tuple
import random
import jax.numpy as jnp

from game.enums import Cell, BOARD_SIZE
from game import board, move, enums

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

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        # call rat init code
        self.rat_hmm = RatHMM(board, transition_matrix)
        pass
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "nishi is trying really hard to not go clinically insane"

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
        moves = board.get_valid_moves()
        return random.choice(moves)
