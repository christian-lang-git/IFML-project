import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURE_SIZE = 626


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((FEATURE_SIZE,len(ACTIONS)))
        #self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


    self.logger.debug("Querying model for action.")
    X = state_to_features(game_state)
    Q = np.dot(X, self.model)
    return ACTIONS[np.argmax(Q)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    feature_vector = np.append(game_state['round'], game_state['step'])
    feature_vector = np.append(feature_vector, game_state['field'])

    bombs = game_state['bombs']
    bomb_count = len(bombs)
    for i in range(bomb_count):
        bomb_position, bomb_countdown = bombs[i]
        feature_vector = np.append(feature_vector, bomb_position)
        feature_vector = np.append(feature_vector, bomb_countdown)
    for i in range(4-bomb_count):
        feature_vector = np.append(feature_vector, (0,0,0))

    feature_vector = np.append(feature_vector, game_state['explosion_map'])

    coins = game_state['coins']
    coin_count = len(coins)
    for i in range(coin_count):
        feature_vector = np.append(feature_vector, coins[i])
    for i in range(9-coin_count):
        feature_vector = np.append(feature_vector, (0,0))
    
    n,s,b,p = game_state['self']
    feature_vector = np.append(feature_vector, s)
    feature_vector = np.append(feature_vector, b)
    feature_vector = np.append(feature_vector, p)
    
    others = game_state['others']
    others_count = len(others)
    for i in range(others_count):
        n,s,b,p = others[i]
        feature_vector = np.append(feature_vector, s)
        feature_vector = np.append(feature_vector, b)
        feature_vector = np.append(feature_vector, p)
    for i in range(3-others_count):
        feature_vector = np.append(feature_vector, (-1,-1,-1,-1))
    
    return feature_vector
