import events as e
from .additional_definitions import *
from ._dqn_presets import *
from ._reward_presets import *

HYPER_PARAMETERS = {
    "USE_CUDA": True,
    "PLOT_PREPROCESSING": True,
    "DEBUG_TRAINING_RESULT": False,
    "MODEL_ARCHITECTURE": MODEL_ARCHITECTURE_DQN_TYPE_L2_FULL,
}

HYPER_PARAMETERS_TRAIN = {
    "EXPLOIT_SYMMETRY": False,
    "USE_8_BATCHES": False,
    "REPLAY_BUFFER_CAPACITY": 250000,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "GAMMA": 0.99,
    "TARGET_STEPS": 1000,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "EPOCH_START_EPSILON_DECAY": 2,
    "EPOCH_STOP_EPSILON_DECAY": 10,
    "EPOCH_LENGTH_TRAINING": 250,
    "EPOCH_LENGTH_VALIDATION": 75,
    "LOOP_THRESHOLD": 3,
    "LOOP_NUM_CHECKS": 10,
    "SKIP_LOOP": False,
    "INSERT_EVENTS": EVENTS_ALL
}

HYPER_PARAMETERS_PROCESSING = {
    "CRATE_DISTANCE_DISCOUNT_FACTOR": 0.9,
    "COIN_DISTANCE_DISCOUNT_FACTOR": 0.9,
    "VISITED_GAIN": 0.05,
    "VISITED_MAX": 1,
    "SONAR_RANGE": 5,
    "SONAR_BAD_THRESHOLD": 0.2,
    "SONAR_GOOD_THRESHOLD": 0.9,
    "USE_EXPLOSION_MAP": False
}

GAME_REWARDS = GAME_REWARDS_PROCESSED

GAME_REWARD_FACTORS = GAME_REWARD_FACTORS_PROCESSED