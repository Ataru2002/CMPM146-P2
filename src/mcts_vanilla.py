
from math import sqrt, log
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice


num_nodes = 1000
explore_faction = 2

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    # Start from the root
    starter = node

    # Traversal
    current_game = state
    current_board = board
    current_identity = bot_identity
    while starter.untried_actions == [] and current_board.is_ended(current_game) != True:
        uct_val = -1000000000
        chosen_node = None
        chosen_move = None
        # Finding the node with the maximum UCT
        for instances in starter.child_nodes.keys():
            #print("Current Stats: ", starter.child_nodes[instances].parent.visits, starter.child_nodes[instances].wins, starter.child_nodes[instances].visits)
            #print("Current UCT: ", ucb(starter.child_nodes[instances], (bot_identity != 1)))
            #print()
            if ucb(starter.child_nodes[instances], (current_identity != bot_identity)) > uct_val:
                uct_val = ucb(starter.child_nodes[instances], (current_identity != bot_identity))
                chosen_node = starter.child_nodes[instances]
                chosen_move = instances
        
        starter = chosen_node
        current_game = current_board.next_state(current_game, chosen_move)
        current_identity = (1 if current_identity == 2 else 2)

    return starter, current_game
    


def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    
    current_board = board
    current_game = state
    if current_board.is_ended(current_game) == True:
        return node, state


    move_chosen = node.untried_actions[0]
    node.untried_actions.pop(0)

    current_game = current_board.next_state(current_game, move_chosen)
    new_node = MCTSNode(parent = node, parent_action = move_chosen, action_list = current_board.legal_actions(current_game))

    node.child_nodes[move_chosen] = new_node
    return new_node, current_game


def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    current_board = board
    current_game = state
    starter = 0 
    while current_board.is_ended(current_game) != True:
        #print("move count: ", starter)
        starter += 1
        move_chosen = choice(current_board.legal_actions(current_game))
        current_game = current_board.next_state(current_game, move_chosen)
    return current_game


def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    node.visits += 1
    if node.parent == None: 
        return
    node.wins += won
    backpropagate(node.parent, won)

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    if node.parent == None:
        # root node
        return -1 
    natural_log = log(node.parent.visits) / log(2.71828)
    exploit = None
    if is_opponent:
        exploit = 1 - (node.wins / node.visits)
    else:
        exploit = (node.wins / node.visits)
    explore = explore_faction * sqrt(natural_log / node.visits)
    return exploit + explore

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    val = -1000000000
    visit = -1
    chosen_move = None
    # Finding the node with the maximum UCT
    for instances in root_node.child_nodes.keys():
        if root_node.child_nodes[instances].wins / root_node.child_nodes[instances].visits > val:
            if root_node.child_nodes[instances].visits > visit:
                val = root_node.child_nodes[instances].wins / root_node.child_nodes[instances].visits
                visit = root_node.child_nodes[instances].visits
                chosen_move = instances
    return chosen_move

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))
    print("Current num_nodes: ", num_nodes)
    for _ in range(num_nodes):
        state = current_state
        node = root_node 
        node, state = traverse_nodes(node, board, state, bot_identity)
        node, state = expand_leaf(node, board, state)
        state = rollout(board, state)
        backpropagate(node, is_win(board, state, bot_identity))


    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    #print(root_node.tree_to_string(horizon=2))
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action
