import events as e
from .additional_definitions import *
from ._model_presets import *
from ._reward_presets import *

HYPER_PARAMETERS = {
    "USE_CUDA": True,
    "PLOT_PREPROCESSING": True,
    "DEBUG_TRAINING_RESULT": False,
    "FORCE_LOAD_AGENT": False,
    # "LOAD_AGENT_PATH": "saved_agents/191707_first_working_agent/agent_236.pt",
    # "LOAD_AGENT_PATH": "saved_agents/201309_agent_begins_to_learn_to_destroy_crates_using_coin_agent/agent_57.pt",
    # "LOAD_AGENT_PATH": "saved_agents/201342_slowly_increase_destroyed_crates/agent_73.pt",
    # "LOAD_AGENT_PATH": "saved_agents/201606_30_crates_at_epoch_31/agent_31.pt",
    #"LOAD_AGENT_PATH": "saved_agents/agent.pt",
    "LOAD_AGENT_PATH": "agent.pt",
    "MODEL_ARCHITECTURE": MODEL_ARCHITECTURE_LINEAR,
}

HYPER_PARAMETERS_TRAIN = {
    "EXPLOIT_SYMMETRY": False,
    "USE_8_BATCHES": False,
    "REPLAY_BUFFER_CAPACITY": 250000,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "GAMMA": 0.99,
    "TARGET_STEPS": 1000,
    "EPSILON_START": 0.5,
    "EPSILON_END": 0.1,
    "EPOCH_START_EPSILON_DECAY": 1,
    "EPOCH_STOP_EPSILON_DECAY": 8,
    "EPOCH_LENGTH_TRAINING": 250,
    "EPOCH_LENGTH_VALIDATION": 75,
    "LOOP_THRESHOLD": 3,
    "LOOP_NUM_CHECKS": 10,
    "SKIP_LOOP": False,
    "DISCOUNT_FACTOR": 0.2,
}

HYPER_PARAMETERS_PROCESSING = {
    "CRATE_DISTANCE_DISCOUNT_FACTOR": 0.9,
    "COIN_DISTANCE_DISCOUNT_FACTOR": 0.9,
    "VISITED_GAIN": 0.05,
    "VISITED_MAX": 1,
    "SONAR_RANGE": 5,
    "SONAR_BAD_THRESHOLD": 0.2,
    "SONAR_GOOD_THRESHOLD": 0.9,
    "USE_EXPLOSION_MAP": False,
}

GAME_REWARDS = GAME_REWARDS_PROCESSED

GAME_REWARD_FACTORS = GAME_REWARD_FACTORS_PROCESSED