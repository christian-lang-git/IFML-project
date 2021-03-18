import pickle
import json
import random
from collections import namedtuple, deque
from typing import List
import events as e
import numpy as np
from .dqn import DQN
from .callbacks import *
from .rule_based import *
from .preprocessing import *
from .replay_buffer import ReplayBuffer
from .epoch_logger import EpochLogger
from .bomberman import *
from .cache import *
from .additional_definitions import *
from ._parameters import *

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
HYPER PARAMETERS
All hyper parameters are extracted from _parameters.py
"""
EXPLOIT_SYMMETRY = HYPER_PARAMETERS_TRAIN["EXPLOIT_SYMMETRY"]
USE_8_BATCHES = HYPER_PARAMETERS_TRAIN["USE_8_BATCHES"]
REPLAY_BUFFER_CAPACITY = HYPER_PARAMETERS_TRAIN["REPLAY_BUFFER_CAPACITY"]
BATCH_SIZE = HYPER_PARAMETERS_TRAIN["BATCH_SIZE"]
EPSILON_START = HYPER_PARAMETERS_TRAIN["EPSILON_START"]
EPSILON_END = HYPER_PARAMETERS_TRAIN["EPSILON_END"]
LEARNING_RATE = HYPER_PARAMETERS_TRAIN["LEARNING_RATE"]
GAMMA = HYPER_PARAMETERS_TRAIN["GAMMA"]
TARGET_STEPS = HYPER_PARAMETERS_TRAIN["TARGET_STEPS"]
EPOCH_START_EPSILON_DECAY = HYPER_PARAMETERS_TRAIN["EPOCH_START_EPSILON_DECAY"]
EPOCH_STOP_EPSILON_DECAY = HYPER_PARAMETERS_TRAIN["EPOCH_STOP_EPSILON_DECAY"]
EPOCH_LENGTH_TRAINING = HYPER_PARAMETERS_TRAIN["EPOCH_LENGTH_TRAINING"]
EPOCH_LENGTH_VALIDATION = HYPER_PARAMETERS_TRAIN["EPOCH_LENGTH_VALIDATION"]
LOOP_THRESHOLD = HYPER_PARAMETERS_TRAIN["LOOP_THRESHOLD"]#if the agent uses the same action at the same position LOOP_THRESHOLD times, a loop is detected
LOOP_NUM_CHECKS = HYPER_PARAMETERS_TRAIN["LOOP_NUM_CHECKS"]
SKIP_LOOP = HYPER_PARAMETERS_TRAIN["SKIP_LOOP"]
DISCOUNT_FACTOR = HYPER_PARAMETERS_TRAIN["DISCOUNT_FACTOR"]
"""
END OF HYPER PARAMETERS
"""
"""
CALCULATED PARAMETERS
"""
#NUMBER_ROUNDS = PLANNED_NUMBER_OF_EPOCHS * (EPOCH_LENGTH_TRAINING + EPOCH_LENGTH_VALIDATION)
#NUMBER_TRAINING_ROUNDS = PLANNED_NUMBER_OF_EPOCHS * EPOCH_LENGTH_TRAINING
"""
END OF CALCULATED PARAMETERS
"""

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'reward'))

#Round_Result = namedtuple('Round_Result',
#                        ('round_index', 'score', 'round_reward', 'epsilon'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #region save parameters
    #it is important to keep track of which parameters were used.
    #automating this step makes this a lot easier.
    dict_save = {}
    dict_save["HYPER_PARAMETERS"] = HYPER_PARAMETERS
    dict_save["HYPER_PARAMETERS_TRAIN"] = HYPER_PARAMETERS_TRAIN
    dict_save["HYPER_PARAMETERS_PROCESSING"] = HYPER_PARAMETERS_PROCESSING
    dict_save["GAME_REWARDS"] = GAME_REWARDS
    dict_save["GAME_REWARD_FACTORS"] = GAME_REWARD_FACTORS
    with open('saved_parameters.json', 'w') as file:
        json.dump(dict_save, file, indent=2)
    #endregion
    
    #region PAPER: Algorithm 1 Line 1
    #'''Quote:   Initialize replay memory D to capacity N'''
    #self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY, feature_dim=FEATURE_DIM, device=self.device)
    #self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY, feature_dim=MODEL_ARCHITECTURE["dim_input"], device=self.device, exploit_symmetry=EXPLOIT_SYMMETRY)    
    #endregion

    #region initialize model
    self.transitions = []

    #endregion

    self.epoch_logger_validation = EpochLogger(name="Validation", epoch_length=EPOCH_LENGTH_VALIDATION, epsilon_start=0, epsilon_end=0, epoch_index_start_decay=-1, epoch_index_stop_decay=-1)
    self.epoch_logger_training = EpochLogger(name="Training", epoch_length=EPOCH_LENGTH_TRAINING, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epoch_index_start_decay=EPOCH_START_EPSILON_DECAY, epoch_index_stop_decay=EPOCH_STOP_EPSILON_DECAY)
    
    if DEBUG_TRAINING_RESULT:
        print("WARNING: DEBUG_TRAINING_RESULT")
        self.epoch_logger_current = self.epoch_logger_validation
        self.currently_training = False
        self.epsilon = 0
    else:
        self.epoch_logger_current = self.epoch_logger_training
        self.currently_training = True
        self.epsilon = EPSILON_START
    
    self.epoch_logger_current.start_epoch()
    self.round_index = 0
    self.round_reward = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if self.gui_mode:
        print("----------")

    #this happens at the beginning of a new round
    if self_action == None:
        return
    
    append_game_state(old_game_state, self.visited_cache)
    append_game_state(new_game_state, self.visited_cache)

    event_values=[None] * len(events)

    #when in raw mode, no preprocessing is done and the preprocessing based auxiliary events are skipped
    process_type=MODEL_ARCHITECTURE["process_type"]
    if True:#True if you want to use those events
        insert_events(self, old_game_state=old_game_state, self_action=self_action, new_game_state=new_game_state, events=events, event_values=event_values)
    else:
        pass
    update_round_train_or_validate(self, events=events, event_values=event_values, old_game_state=old_game_state, self_action=self_action, new_game_state=new_game_state, termination_flag=False)

def insert_events_raw(self, new_game_state: dict, events: List[str], event_values):
    agent_coords = new_game_state["self"][3]    
    visited_penalty = - GAME_REWARD_FACTORS[E_VISITED_PENALTY] * new_game_state["visited"][agent_coords]
    events.append(E_VISITED_PENALTY)
    event_values.append(visited_penalty)

def insert_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], event_values: List[float]):
    #old_preprocessed = preprocess(old_game_state, plot=False)
    #old_coin_values = old_preprocessed[1]
    #old_coin_value = old_coin_values[old_coords]
    #print("insert_events state_to_features old: ", PROCESS_LINEAR)
    old_features = state_to_features_cache_wrapper(turn_index=self.turn_index, game_state=old_game_state, feature_cache=self.feature_cache, visited_cache=self.visited_cache, processing_cache=self.processing_cache, process_type=PROCESS_LINEAR_FULL, plot_preprocessing=self.plot_preprocessing)
    #print("insert_events state_to_features new: ", PROCESS_LINEAR)
    new_features = state_to_features_cache_wrapper(turn_index=self.turn_index+1, game_state=new_game_state, feature_cache=self.feature_cache, visited_cache=self.visited_cache, processing_cache=self.processing_cache, process_type=PROCESS_LINEAR_FULL, plot_preprocessing=self.plot_preprocessing)

    insert_events_towards_away(self, old_features=old_features, events=events, event_values=event_values,
        feature_index=LINEAR_INDEX_COIN_VALUE_PLAYER,
        towards_event=E_MOVED_TOWARDS_COIN,
        away_event=E_MOVED_AWAY_FROM_COIN)

    #only check moved towards/away from crate if bomb is ready
    #if old_features[LINEAR_INDEX_BOMB_STATUS]:
    insert_events_towards_away(self, old_features=old_features, events=events, event_values=event_values,
        feature_index=LINEAR_INDEX_CRATE_VALUE_PLAYER, 
        towards_event=E_MOVED_TOWARDS_CRATE,
        away_event=E_MOVED_AWAY_FROM_CRATE)

    #check if bomb was dropped near crate / enemy
    good_bomb = False #if no bomb was dropped it was no good bomb
    for event in events:
        if event == e.BOMB_DROPPED:
            potential = old_features[LINEAR_INDEX_CRATE_POTENTIAL_PLAYER]
            value = GAME_REWARD_FACTORS[E_DROPPED_BOMB_NEAR_CRATE] * potential
            if value > 0:
                events.append(E_DROPPED_BOMB_NEAR_CRATE)
                event_values.append(value)
                good_bomb = True #bomb was near crate --> good bomb
            else:
                sonar = old_features[LINEAR_INDEX_SONAR_PLAYER]
                if sonar < SONAR_BAD_THRESHOLD:
                    #if no crate or enemy is affected it is probably a bad bomb
                    events.append(E_DROPPED_BOMB_BAD)
                    event_values.append(None)
                if sonar > SONAR_GOOD_THRESHOLD:
                    good_bomb = True #bomb was near enemy --> good bomb
                    events.append(E_DROPPED_BOMB_NEAR_ENEMY)
                    event_values.append(None)
            break

    #check if certain death
    insert_events_danger_exceeded(self, old_features=old_features, new_features=new_features, events=events, event_values=event_values)

    #visited penalty is skipped if a good bomb was dropped
    if good_bomb:
        return
    #insert visited penalty
    #player_coords = new_game_state["self"][3]
    #visited_penalty = GAME_REWARD_FACTORS[E_VISITED_PENALTY] * new_game_state["visited"][player_coords]
    visited_penalty = GAME_REWARD_FACTORS[E_VISITED_PENALTY] * new_features[LINEAR_INDEX_VISITED_PENALTY_PLAYER]
    #print("visited penalty", visited_penalty)
    events.append(E_VISITED_PENALTY)
    event_values.append(visited_penalty)
    

def insert_events_towards_away(self, old_features, feature_index, towards_event, away_event, events: List[str], event_values: List[float]):
    #old_preprocessed = preprocess(old_game_state, plot=False)
    #old_coin_values = old_preprocessed[1]
    #old_coin_value = old_coin_values[old_coords]
    old_value = old_features[feature_index]

    new_value = -1
    moved = False
    for event in events:
        if event == e.MOVED_LEFT:
            #print("MOVED_LEFT")
            moved = True
            new_value = old_features[feature_index+1]
        elif event == e.MOVED_RIGHT:
            #print("MOVED_RIGHT")
            moved = True
            new_value = old_features[feature_index+2]
        elif event == e.MOVED_UP:
            #print("MOVED_UP")
            moved = True
            new_value = old_features[feature_index+3]
        elif event == e.MOVED_DOWN:
            #print("MOVED_DOWN")
            moved = True
            new_value = old_features[feature_index+4]

    if moved:
        #print("old_coin_value",old_coin_value)
        #print("new_coin_value",new_coin_value)
        if new_value > old_value:
            #print(f"value should have increased from {old_coin_value} to {new_coin_value}")
            events.append(towards_event)
            event_values.append(GAME_REWARD_FACTORS[towards_event] * (new_value - old_value))
        elif new_value < old_value:
            #print(f"value should have decreased from {old_coin_value} to {new_coin_value}")
            events.append(away_event)
            event_values.append(GAME_REWARD_FACTORS[away_event] * (new_value - old_value))

        if self.gui_mode:
            print(old_value, new_value)

def insert_events_danger_exceeded(self, old_features, new_features, events: List[str], event_values):
    #old_preprocessed = preprocess(old_game_state, plot=False)
    #old_coin_values = old_preprocessed[1]
    #old_coin_value = old_coin_values[old_coords]
    old_value = old_features[LINEAR_INDEX_DANGER_PLAYER]
    if old_value > 0.99:
        #print("DANGER EXCEEDED LAST TURN")
        return
    new_value = new_features[LINEAR_INDEX_DANGER_PLAYER]
    if new_value > 0.99:
        #print(f"value should have increased from {old_coin_value} to {new_coin_value}")
        events.append(E_DANGER_EXCEEDED)
        event_values.append(None)
        #print("DANGER EXCEEDED")

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    
    append_game_state(last_game_state, self.visited_cache)

    agent_data = last_game_state['self']
    score = agent_data[1]

    update_round_train_or_validate(self, events=events, event_values=None, old_game_state=last_game_state, self_action=last_action, new_game_state=None, termination_flag=True)
    finalize_round_train_or_validate(self, score=score)

    #store round results for analysis
    self.logger.info(f'End of round: {self.round_index}. Finished with score: {score} and reward: {self.round_reward} and epsilon: {self.epsilon}')
 
    #prepare next round
    self.round_index += 1
    self.round_reward = 0

    #reset termination request for next round
    self.request_termination = False
    self.is_loop = False
    self.turn_index = -1
    
def reward_from_events(self, events: List[str], event_values: List[float]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    reward_sum = 0
    for index, event in enumerate(events):
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
            if self.gui_mode:
                print(event, GAME_REWARDS[event])
        elif event in GAME_REWARD_FACTORS:
            reward_sum += event_values[index]
            if self.gui_mode:
                print(event, event_values[index])

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def update_round_train_or_validate(self, events, event_values, old_game_state, self_action, new_game_state, termination_flag):
    
    if self_action == None:
        return

    reward = reward_from_events(self, events, event_values)

    hasattr_gui_mode = hasattr(self, "gui_mode")
    if hasattr_gui_mode:
        #allow only in gui_mode
        if self.gui_mode:
            print("action: ", self_action, "reward: ", reward)

    #get integer representation of provided action string
    action_index = INVERSE_ACTIONS[self_action]

    invalid_action_flag = False
    bomb_dropped_flag = False
    bad_bomb_flag = False
    crates = 0
    coins = 0
    kills = 0
    for event in events:
        if event == e.INVALID_ACTION:
            invalid_action_flag = True
        if event == e.BOMB_DROPPED:
            bomb_dropped_flag = True
        elif event == E_DROPPED_BOMB_BAD:
            bad_bomb_flag = True
        elif event == e.CRATE_DESTROYED:
            crates += 1
        elif event == e.COIN_COLLECTED:
            coins += 1
        elif event == e.KILLED_OPPONENT:
            kills += 1

    if self.currently_training:
        train(self, old_game_state=old_game_state, action_index=action_index, reward=reward, new_game_state=new_game_state, termination_flag=termination_flag)
        
    self.epoch_logger_current.update_round(action_index, reward, invalid_action_flag, crates, coins, kills, self.is_loop, bomb_dropped_flag, bad_bomb_flag)

def finalize_round_train_or_validate(self, score):
    self.epoch_logger_current.finalize_round(score)
    end_flag = self.epoch_logger_current.try_finalize_epoch()    
    if end_flag:
        switch_epoch_logger(self)
    self.epsilon = self.epoch_logger_current.epsilon

def switch_epoch_logger(self):
    self.currently_training = not self.currently_training
    if self.currently_training:
        self.epoch_logger_current = self.epoch_logger_training
    else:
        self.epoch_logger_current = self.epoch_logger_validation
        save_agent(self)
    self.epoch_logger_current.start_epoch()

def save_agent(self):
    path = f"saved_agents/agent_{self.epoch_logger_training.epoch_index}.pt"
    with open(path, "wb") as file:
        pickle.dump(self.model, file)

def train(self, old_game_state, action_index, reward, new_game_state, termination_flag):

    features_old = state_to_features_cache_wrapper(turn_index=self.turn_index, game_state=old_game_state, feature_cache=self.feature_cache, visited_cache=self.visited_cache, processing_cache=self.processing_cache, process_type=MODEL_ARCHITECTURE["process_type"], plot_preprocessing=self.plot_preprocessing)
    
    #features_new = 0.0
    #if not termination_flag:
    #    features_new = state_to_features_cache_wrapper(turn_index=self.turn_index+1, game_state=new_game_state, feature_cache=self.feature_cache, visited_cache=self.visited_cache, processing_cache=self.processing_cache, process_type=MODEL_ARCHITECTURE["process_type"], plot_preprocessing=self.plot_preprocessing)

    self.transitions.append(Transition(features_old, reward))


    if termination_flag:
        transition_count = len(self.transitions)
        rewards = np.array([t.reward for t in self.transitions])
        gradient_list = np.empty((transition_count, len(features_old), len(ACTIONS)))
        for i, transition in enumerate(self.transitions):
            X = transition.state
            discout_exponents = np.arange(0, transition_count-i)
            Y = np.sum(DISCOUNT_FACTOR**discout_exponents * rewards[i:])
            if type(X) != type(Y): continue
            gradient = np.dot(X.T, (Y - np.dot(X, self.model)))
        self.model = np.sum(gradient_list,axis=0) / gradient_list.shape[0]

    loss = 0
    self.epoch_logger_current.add_loss(loss)
