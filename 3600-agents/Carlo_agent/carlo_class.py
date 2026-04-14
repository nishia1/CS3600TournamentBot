from collections.abc import Callable
from typing import List, Set, Tuple
import random

from numba.core.extending import overload_method

from engine.game.enums import *
import jax.numpy as jnp

# TODO: refactor import for use in testing
from engine.game import board, move, enums



class carlo_node:
    def __init__(self, gm_board:board, action: move, enemy: bool, rat_coords: tuple[int, int], rat_percentage: float):
        """
        Build a carlo node

        Args:
            gm_board: the board associated with this state. You may find board.forecast_move() helpful to get future boards ,
            action: the move that resulted in this state
            enemy: if True assumes current state is the player's turn
            rat_coords: coordinates of highest chance for the rat guess
            rat_percentage: percent chance rat is at rat_coordinates

        """
        # the board representing the current state
        self.state_board = gm_board

        # __pos_actions stores actions possible from current state
        self.__pos_actions = None

        # par_action stores the action that resulted in this state
        # i.e. the parent state's action
        self.par_action = action

        # Remembers if state is "visited" (may be redundant)
        self.explored = False

        # Stores times visited
        self.visits = 0

        # Stores how what points we have won rolling out from this path (can simply represent wins too)
        self.points_from_path = 0

        # Stores utility
        self.utility = 0

        #stores whether this node is on the enemies turn
        self.enemy = enemy

        # a dictionary of action: child pairs that connects this state to it's children
        # is only expanded with expand_children, and NOT with get_possible_actions
        self.child_states = dict()

        # where the rat guess would be
        self.rat_cords = rat_coords

        # what is the chance of our rat guess being right
        self.rat_perc = rat_percentage

        '''
        
        useless code, but undecided about POTENTIAL use
        
        # tile_state is BOARD_SIZE by BOARD_SIZE sized array that holds the state
        # of each tile as an enum
        self.tile_state = jnp.zeros((BOARD_SIZE, BOARD_SIZE))
        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):
                self.tile_state[x - 1, y - 1] = gm_board.get_cell((x, y))
        '''
        if enemy:
            self.our_pos = gm_board.opponent_worker.position
            self.opp_pos = gm_board.player_worker.position

        else:
            self.our_pos = gm_board.player_worker.position
            self.opp_pos = gm_board.opponent_worker.position

    # WARNING: use this to get child actions,
    # it DOES NOT define self.child_states
    # CARES ABOUT ALL POSSIBILITIES REGARDLESS OF 'action_generator' RESTRICTIONS
    def get_possible_actions(self) -> list:
        if self.__pos_actions is None:
            self.__pos_actions = self.state_board.get_valid_moves()
        return self.__pos_actions

    #TODO: ignore_rat_pos is based off of the idea that we might want "different" states to be treated equivalently
    #probably need to save rat position still, because we need to be able to know WHERE to guess, I don't know this is all confusing
    #this function is also incapable of handling mouse guesses because it doesn't know how to update position based on guessing of the player or team
    #or emissions. I will need to fix that
    #this ALSO does not account for the fact that other states may be indistinguishable from eachother: doing a move that doesn't change the environment
    #will cause inefficiencies (Guessing the rat and placing carpet on carpet, if valid, will create two equivalent states)
    def expand_children(self, action_generator = None, ignore_rat_pos = True):
        """
            Initiates all child states based off of an action generator.
            The function defaults to basing children off of __pos_actions if action_generator is left as None

            Args:
                action_generator: defaults to None -
                    This function should take a carlo_node object, and returns a list of actions
                ignore_rat_pos: defaults to True -
                    This decides whether to make a distinction between states with different locations for rat
                    guesses.
        """

        # rounding is necessary so there isnt a difference between states with chances like 0.101 and 0.102
        #TODO: might be best to just make a list of percentages we need to make scenarios for, then initalize nodes for all percent possibilities
        temp_mouse_percentage = round(self.rat_perc, 1)


        if action_generator is None:
            actions = self.get_possible_actions()
            for i in actions:
                self.child_states[i] = self.__init__(self.state_board.forecast_move(i, False), i, not self.enemy, (-1, -1), temp_mouse_percentage)
        else:
            actions = action_generator(self)
            for i in actions:
                self.child_states[i] = self.__init__(self.state_board.forecast_move(i, False), i, not self.enemy, (-1, -1), temp_mouse_percentage)

    # syntactical sugar that gets the type enum for a cell
    def get_cell(self, x, y) -> IntEnum:
        return self.state_board.get_cell((x, y))

    # TODO: decide whether We want to hard code an update_utility function, or rely on children to override this funciton
    def update_utility(self, children_utility):
        '''
            Updates its own utility
        '''
        pass

    #TODO: decide whether We want to hard code a do_rollout function, or rely on children to override this funciton
    def do_rollout(self):
        pass
