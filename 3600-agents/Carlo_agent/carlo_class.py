import math
from collections.abc import Callable
from typing import List, Set, Tuple
import random

from numba.core.extending import overload_method


import jax.numpy as jnp

#### TODO: refactor import for use in testing
from engine.game import move, enums
from engine.game.board import Board as board
from engine.game.enums import *
#############################################

import heapq
import itertools

# Placeholder reward function
def simple_reward(parent=None, action=None, result=None, winner: Result = Result.TIE, result_enemy=False) -> int:
    if (winner is None) or (winner is Result.TIE):
        return 0  # no terminal state yet

    # assuming winner represents which side won
    if winner is Result.ENEMY:
        return -1
    elif winner is Result.PLAYER:
        return 10
    else:
        print("Error went into reward function somehow")
        return 0

# This ghost state is for the rollout functionality, may be usefull for checking points or backtracking random rollout paths
class ghost_state:
    __slots__ = ["state_board", "action"]
    def __init__(self, board, action):
        self.state_board = board
        self.action = action
    def check_win(self):
        self.state_board.check_win()
    def winner(self):
        return self.state_board.get_winner()
class carlo_node:
    def __init__(self, gm_board:board, enemy: bool,  rat_coords: tuple[int, int] = (-1, -1), rat_percentage: float = 0, parent_node = None, action: move.Move = None, exploration_constant: float = 0.5):
        """
        Build a carlo node

        Args:
            gm_board: the board associated with this state. You may find board.forecast_move() helpful to get future boards ,

            enemy: if True assumes current state is the player's turn

            rat_coords: coordinates of highest chance for the rat guess, defaults to (-1, -1)

            rat_percentage: percent chance rat is at rat_coordinates, defaults to 0

            parent_node: the parent of this carlo node, defaults to None

            action: the move that resulted in this state, defaults to None

            exploration_constant: the constant that effects ucb1 and determines
                how frequently we chose to explore over exploit
        """
        # the board representing the current state
        self.state_board = gm_board

        # __pos_actions stores actions possible from current state
        self.__pos_actions = None

        # par_action stores the action that resulted in this state
        # i.e. the parent state's action
        self.par_action = action


        #the node that is the parent: by default is None
        self.par_state = parent_node

        # Remembers if state is "visited" (might be redundant)
        self.explored = False

        # Stores times visited
        self.visits = 0

        # Stores how what points we have won rolling out from this path (can simply represent wins too)
        self.points_from_path = 0

        # Stores utility
        self.utility = 0

        # Stores the score by which things are prioritized
        self.ucb1 = 0

        # The exploration constant used to help calculate ucb
        self.exploration_con = exploration_constant

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

    def __lt__(self, other):
        if self.ucb1 < other.ucb1:
            return True
        else:
            return False

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

            Returns:
                the children of the node for the tree to add members to the heap
        """

        # rounding is necessary so there isnt a difference between states with chances like 0.101 and 0.102
        #TODO: might be best to just make a list of percentages we need to make scenarios for, then initalize nodes for all percent possibilities
        temp_mouse_percentage = round(self.rat_perc, 1)


        if action_generator is None:
            actions = self.get_possible_actions()
            for i in actions:
                self.child_states[i] = self.__init__(self.state_board.forecast_move(i, False),  not self.enemy, (-1, -1), temp_mouse_percentage, self, i, self.exploration_con)
        else:
            actions = action_generator(self)
            for i in actions:
                self.child_states[i] = self.__init__(self.state_board.forecast_move(i, False), not self.enemy, (-1, -1), temp_mouse_percentage, self, i, self.exploration_con)
        return list(self.child_states.values())

    # syntactical sugar that gets the type enum for a cell
    def get_cell(self, x, y) -> IntEnum:
        return self.state_board.get_cell((x, y))

    def update_utility(self, children_points):
        '''
            Updates its own utility, score count, visits
            for the purpose of backpropagation.
            WARNING: Does not update ucb1
            This function is used by the tree when it iterates backwards and finds a node to update
            so this function does not backpropigate on its own

                Args:
                    children_points: the points (NOT UTILITY) that are being passed backwards from the child

        '''
        self.points_from_path += children_points
        self.visits += 1
        self.utility = self.points_from_path / self.visits


    def update_ucb1(self):
        '''

            updates its own ucb1, NOTE this is necessary to run AFTER updating ALL node visits and points

        '''
        N = self.par_state
        if N is None:
            return
        N = self.par_state.visits
        self.ucb1 = self.utility + self.exploration_con * (math.sqrt(math.log(N) / self.visits))

    #TODO: brainstorm a way to make mouse guess relevant
    def do_rollout(self, reward_function: callable):
        """
            randomly explores the child states until it reaches an end state.
            After it is finished, it sets itself as explored.
            NOTE: does not update its own ucb1

            Args:
                reward_function: the function that will determine the rewards for transitioning to a result state
                    it should have a function signature like so:
                    (parent: board = None, action: move.Move = None, result: board, winner: Result, result_enemy: bool) -> int
                    See implimentation for more comments about param expectations
            Returns:
                the points associated with this node

        """
        # parent: a board object that represents the "starting" state
        # action: a move.Move object that represents the move taken from "parent"
        # result: a board object that represents where the action takes the game
        # winner: a Result intEnum that represents whether either team has won as a result, is None if there is no victor
        # result_enemy: whether the result state is the enemies turn
        #   (i.e the enemy will, in the future, perform an action from the result state)
        #
        # Advice: the way this is designed, it might be a good to give *small* penalties when the enemy gets points.


        #KEY NOTE: this functionality iterates over BOARDS, not NODES.
        #Future changes may use the ghost state class from earlier
        parent = self.state_board
        action = random.choice(self.state_board.get_valid_moves(self.enemy, True))
        curr = parent.forecast_move(action)
        enemy = self.enemy

        winner = None

        while winner is None:
            curr.check_win()
            winner = curr.get_winner()
            # interesting note: way things are written, checks for rewards before moving from first state
            self.points_from_path += reward_function(parent, action, curr,  winner, enemy)
            action = random.choice(curr.get_valid_moves(self.enemy, True))
            parent = curr
            curr = curr.forecast_move(action)
            enemy = not enemy

        self.explored = True
        self.visits = 1
        self.utility = self.points_from_path
        return self.points_from_path


# WARNING: CHATGPT ORIGINATING CODE - used in carlo_tree to know where to expand or rollout next
class carlo_priority_heap:
    def __init__(self):
        self.heap = []  # normal priority heap (explored items)
        self.unexplored = []  # not yet explored items

    def add(self, node: carlo_node):
        """
        If node.explored is True → goes to heap
        If False → goes to unexplored list
        """
        if node.explored:
            heapq.heappush(
                self.heap,
                node
            )
        else:
            self.unexplored.append(node)

    def pop(self):
        """
        Priority rule:
        1. If unexplored list has items → return from it first
        2. Otherwise pop from heap
        """

        if len(self.unexplored) > 0:
            # simplest behavior: FIFO for mandatory
            return self.unexplored.pop(0)

        if self.heap:
            _, _, node = heapq.heappop(self.heap)
            return node

        return None  # nothing left

    def __len__(self):
        return len(self.heap) + len(self.unexplored)

# the tree in which the carlo nodes are used
class carlo_tree:
    def __init__(self, head_board: list[board], enemy: bool, reward_function: callable = simple_reward()):
        """
            Args:
                head_board: this is the starting state.
                    This parameter is a list in case there are multiple starting states

                enemy: whether the enemy starts first

                reward_function: the function that will determine the rewards for transitioning to a result state
                    it should have a function signature like so:
                    (parent: board = None, action: MoveType = None, result: board, winner: Result, result_enemy: bool) -> int
                    See implementation of carlo_node.do_rollout for more comments about param expectations

        """
        self.head_nodes: list[carlo_node] = list()
        self.heap = carlo_priority_heap()
        for i in head_board:
            curr = carlo_node(i, enemy, exploration_constant= 0.5)
            self.heap.add(curr)
            self.head_nodes.append(curr)

        self.reward_function = reward_function

        # this is a list of the children nodes of all of the head nodes, which will be updated as the heads are expanded
        # NOTE: this can be incomplete if executed early
        self.choices: list[carlo_node] = list()


        self.total_rollouts = 0

    def step(self):
        """
            pops a single item from the heap and decides whether to rollout or expand
        """
        # if popped node is unvisited: perform rollout and do backprop,
        # afterward re-add the current node
        #
        # otherwise, since it is visited, we expand and add children to the heap,
        # do not re-add the current node

        curr = self.heap.pop()

        if not curr.explored:
            self.rollout_and_backprop(curr)
        else: # this means it is explored, but since it came from the heap that means it hasn't been expanded yet
            self.expand_and_add(curr)




    def rollout_and_backprop(self, curr: carlo_node):
        '''

            performs rollout on curr and then backpropigates changes in points and visits first,
            then updates ucb1 values
            NOTE: re-adds curr to the heap

            Args:
                curr: the node we want to perform rollout on

        '''
        points = curr.do_rollout(self.reward_function)

        node_iter = curr.par_state
        while node_iter is not None:
            node_iter.update_utility(points)
            node_iter = node_iter.par_state

        node_iter = curr
        while node_iter.par_state is not None:
            node_iter.update_ucb1()
            node_iter = node_iter.par_state
        self.heap.add(curr)

    def expand_and_add(self, curr: carlo_node):

        '''

            expands current node and adds the children to the heap.
            this function checks if curr is a header node so that we can update self.choices

            Args:
                curr: the node we want to add to the heap and update their children

        '''

        children = curr.expand_children()

        # this check is to see if we need to add the children to the self.choices instance variable
        if curr in self.head_nodes:
            for i in children:
                self.choices.append(i)
                self.heap.add(i)
        else:
            for i in children:
                self.heap.add(i)

    def make_choice(self):
        best_node = None
        for i in self.choices:
            if best_node is None:
                best_node = i
            elif i.ucb1 > best_node.ucb1:
                best_node = i
        return best_node.par_action

    def check_choices_full(self) -> bool:
        '''
            makes sure that all of the heading nodes have at least been expanded

            Returns:
                True if all head nodes have been expanded
        '''

        for i in self.head_nodes:
            if len(i.child_states) == 0:
                return False

        return True



    #IDEAS: Calculate Reward - may be better to implement in tree

    #IDEAS: Determine if end state


