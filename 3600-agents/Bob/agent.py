from collections.abc import Callable
from typing import List, Set, Tuple
import random
import jax.numpy as jnp

from game import board, move, enums

class RatHMM:
    def update(self, sensor_data):
        # update probabilities of the val
        # for each cell in board, update probability by doing rat
        # 1. predict
        self.val = self.val @ self.transition_matrix

        # 2. update (if you have observation)
        self.val *= sensor_data
        self.val /= self.val.sum()
    
    # call this to det which cell most likely to have rat and expected value of move
    # which is just 4 times the cell with the highest probablity of having the rat
    def likelycell(self):
        return jnp.unravel_index(jnp.argmax(self.val), self.val.shape) * 4

    def __init__(self, board, transition_matrix=None):
        # make a jax array of all equal probabilities for each cell in the board inititally
        self.val = jnp.ones((board.size, board.size)) / (board.size * board.size)
        # take the array and make all probabilities equal
        self.val = self.val / jnp.sum(self.val)
        self.transition_matrix = transition_matrix


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
