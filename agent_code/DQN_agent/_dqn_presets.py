from .dqn import *
from .additional_definitions import *
from .bomberman import NUM_ACTIONS

MODEL_ARCHITECTURE_DQN_TYPE_L2_SMALL = {
    "model_type": DQN_TYPE_L2,
    "process_type": PROCESS_LINEAR_SMALL,
    "dim_input": [5],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 16,
    "dim_layer_full_2": 16,
}
MODEL_ARCHITECTURE_DQN_TYPE_L2 = {
    "model_type": DQN_TYPE_L2,
    "process_type": PROCESS_LINEAR,
    "dim_input": [21],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 128,
    "dim_layer_full_2": 128,
}
MODEL_ARCHITECTURE_DQN_TYPE_L2_FULL = {
    "model_type": DQN_TYPE_L2,
    "process_type": PROCESS_LINEAR_FULL,
    "dim_input": [26],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 128,
    "dim_layer_full_2": 128,
}
MODEL_ARCHITECTURE_DQN_TYPE_L3 = {
    "model_type": DQN_TYPE_L3,
    "process_type": PROCESS_LINEAR,
    "dim_input": [21],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 128,
    "dim_layer_full_2": 128,
    "dim_layer_full_3": 128,
}
MODEL_ARCHITECTURE_DQN_TYPE_C1L2 = {
    "model_type": DQN_TYPE_C1L2,
    "process_type": PROCESS_CONVOLUTION,
    "dim_input": [2, 17, 17],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 16,
    "dim_layer_full_2": 16,
}
MODEL_ARCHITECTURE_DQN_TYPE_C3L1 = {
    "model_type": DQN_TYPE_C3L1,
    "process_type": PROCESS_CONVOLUTION,
    "dim_input": [2, 17, 17],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 256,
}
MODEL_ARCHITECTURE_DQN_TYPE_C3L2 = {
    "model_type": DQN_TYPE_C3L2,
    "process_type": PROCESS_CONVOLUTION,
    "dim_input": [2, 17, 17],
    "dim_output": NUM_ACTIONS,
    "dim_layer_full_1": 16,
    "dim_layer_full_2": 16,
}