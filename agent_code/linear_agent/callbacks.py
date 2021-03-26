"""
INSTALLATION COMMANDS:
    pip install torch
FOR CUDA SUPPORT ON GTX 1060:
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge
"""
import os
import pickle
import json
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import shuffle#TODO still needed here?

from .dqn import *
from .preprocessing import *
from .bomberman import *
from .cache import *
from ._parameters import HYPER_PARAMETERS

import matplotlib.pyplot as plt #for debugging

"""
HYPER PARAMETERS
All hyper parameters are extracted from _parameters.py
"""
USE_CUDA = HYPER_PARAMETERS["USE_CUDA"]
PLOT_PREPROCESSING = HYPER_PARAMETERS["PLOT_PREPROCESSING"]
DEBUG_TRAINING_RESULT = HYPER_PARAMETERS["DEBUG_TRAINING_RESULT"]
MODEL_ARCHITECTURE = HYPER_PARAMETERS["MODEL_ARCHITECTURE"]
FORCE_LOAD_AGENT = HYPER_PARAMETERS["FORCE_LOAD_AGENT"]
LOAD_AGENT_PATH = HYPER_PARAMETERS["LOAD_AGENT_PATH"]
"""
END OF HYPER PARAMETERS
"""

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
    #region CUDA
    if USE_CUDA and T.cuda.is_available():
        print("CUDA AVAILABLE")
        self.device = T.device("cuda")
    else:                
        print("CUDA NOT AVAILABLE OR DISABLED")
        self.device = T.device("cpu")
    #endregion

    #region load agent from file
    if not self.train:
        print("agent not in training mode")
        with open(LOAD_AGENT_PATH, "rb") as file:
            self.model = pickle.load(file)
            print(self.model)
    elif DEBUG_TRAINING_RESULT: 
        print("agent loaded despite training mode")
        with open("agent.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        if FORCE_LOAD_AGENT and os.path.isfile(LOAD_AGENT_PATH):
            print('model loaded from agent.pt for further training')
            with open(LOAD_AGENT_PATH, "rb") as file:
                self.model = pickle.load(file)
            print(self.model)
        print("agent in training mode")
    if not hasattr(self, 'model'):
        self.model = np.zeros((1445,len(ACTIONS)))
    #endregion

    #region plot processing
    #default value is False since we do not want to destroy our hard disk
    #or impact performance during training.
    self.plot_preprocessing = False
    #gui_mode is not available in the provided framework
    #this check makes it not crash when using the provided framework
    hasattr_gui_mode = hasattr(self, "gui_mode")
    print("has attribute gui_mode", hasattr_gui_mode)
    if hasattr_gui_mode:
        print("gui_mode", self.gui_mode)
        #allow only in gui_mode
        if self.gui_mode:
            self.plot_preprocessing = PLOT_PREPROCESSING
    print("plot_preprocessing", self.plot_preprocessing)
    #endregion

    self.processing_cache = {}
    self.feature_cache = Cache("feature_cache")
    self.visited_cache = Cache("visited_cache")
    self.is_loop = False
    self.last_action_random = False
    self.turn_index = -1

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    step = game_state["step"]
    if step == 1:
        self.processing_cache = {}
        self.feature_cache.reset()
        self.visited_cache.reset()

    self.turn_index += 1

    append_game_state(game_state, self.visited_cache)

    #region loop solution
    #self.is_loop is only set during training
    if self.is_loop:
        #print("loop detected")
        self.logger.debug("loop detected")
        return act_exploration(self, game_state)
    #endregion

    #region Exploration (can occur only during training)
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        #print("act randomly")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        self.last_action_random = True
        return act_exploration(self, game_state)
    #endregion

    #region Exploitation
    self.logger.debug("Querying model for action.")
    self.last_action_random = False
    features = state_to_features_cache_wrapper(turn_index=self.turn_index, game_state=game_state, feature_cache=self.feature_cache, visited_cache=self.visited_cache, processing_cache=self.processing_cache, process_type=MODEL_ARCHITECTURE["process_type"], plot_preprocessing=self.plot_preprocessing)
    #reshape the features to create a batch of size 1
    #features = features.reshape(1, *features.shape)
    #feature_tensor = T.tensor(features, dtype=T.float32, device=self.device)
    actions = np.dot(features, self.model)
    action_index = np.argmax(actions)
    action = ACTIONS[action_index]
    self.logger.debug(f"Select action: {action}")
    return action
    #endregion

def state_to_features_cache_wrapper(turn_index, game_state: dict, feature_cache, visited_cache, processing_cache: dict, process_type, plot_preprocessing) -> np.array:
    has_data, data = feature_cache.get_data(process_type, turn_index)
    if has_data:
        return data

    data = state_to_features(game_state, processing_cache, process_type, plot_preprocessing)
    feature_cache.set_data(process_type, turn_index, data)
    return data

def state_to_features(game_state: dict, processing_cache: dict, process_type, plot_preprocessing) -> np.array:
    """    
    Converts the game state to the input of the model.
    Returns different results depending on the specified architecture.    
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    #special case for raw - no preprocessing is required
    if process_type == PROCESS_CONVOLUTION_RAW or process_type == PROCESS_TODO_SKIP_PREPROCESSING:
        return process(game_state=game_state, preprocessing_result=None, process_type=process_type)
    
    #preprocess independent of which features we actually want to use
    preprocessing_result = preprocess(game_state=game_state, plot=plot_preprocessing, processing_cache=processing_cache)
    #process the results of preprocess depending on the model architecture
    processing_result = process(game_state=game_state, preprocessing_result=preprocessing_result, process_type=process_type)
    return processing_result

def act_exploration(self, game_state):
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
