import events as e
from .additional_definitions import *
from ._dqn_presets import *

HYPER_PARAMETERS = {
    "USE_CUDA": True,
    "PLOT_PREPROCESSING": True,
    "DEBUG_TRAINING_RESULT": False,
    "MODEL_ARCHITECTURE": MODEL_ARCHITECTURE_DQN_TYPE_L2_FULL,
}

HYPER_PARAMETERS_TRAIN = {
    "EXPLOIT_SYMMETRY": False,
    "REPLAY_BUFFER_CAPACITY": 250000,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "GAMMA": 0.99,
    "TARGET_STEPS": 1000,
    "EPSILON_START": 1,
    "EPSILON_END": 0.1,
    "EPOCH_START_EPSILON_DECAY": 2,
    "EPOCH_STOP_EPSILON_DECAY": 10,
    "EPOCH_LENGTH_TRAINING": 250,
    "EPOCH_LENGTH_VALIDATION": 75,
    "LOOP_THRESHOLD": 3,
    "LOOP_NUM_CHECKS": 10,
    "SKIP_LOOP": False,
}

HYPER_PARAMETERS_PROCESSING = {
    "CRATE_DISTANCE_DISCOUNT_FACTOR": 0.9,
    "COIN_DISTANCE_DISCOUNT_FACTOR": 0.9,
    "VISITED_GAIN": 0.05,
    "VISITED_MAX": 1,
    "SONAR_RANGE": 5,
    "SONAR_BAD_THRESHOLD": 0.2
}

GAME_REWARDS = {
    e.COIN_COLLECTED: 1,
    #e.KILLED_OPPONENT: 5,
    #e.CRATE_DESTROYED: 0.01,    #for now just a small reward that should encourage the agent to get out of the starting area and discover coins
    e.GOT_KILLED: -1,          #strong penalty for death (this includes suicide)
    #e.KILLED_SELF: -5,          #additional penalty for suicide
    e.INVALID_ACTION: -0.1,    #invalid actions are bad, but should probably not be discouraged too much
    #E_MOVED_TOWARDS_COIN: 0.1,
    #E_MOVED_AWAY_FROM_COIN: -0.11,
    #e.BOMB_DROPPED: -0.2,
    #e.WAITED: -0.05,
    E_DANGER_EXCEEDED: -5,
    E_DROPPED_BOMB_BAD: -0.5
}

GAME_REWARD_FACTORS = {
    E_MOVED_TOWARDS_COIN: 2,
    E_MOVED_AWAY_FROM_COIN: 2.2,
    E_MOVED_TOWARDS_CRATE: 1,
    E_MOVED_AWAY_FROM_CRATE: 1.1,
    E_DROPPED_BOMB_NEAR_CRATE: 1, #this is multiplied with number of crates in bomb range / 10
    E_VISITED_PENALTY: 0.25
}