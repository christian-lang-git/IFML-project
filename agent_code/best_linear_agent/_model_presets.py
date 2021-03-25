from .additional_definitions import *
from .bomberman import NUM_ACTIONS

MODEL_TYPE_LINEAR = "MODEL_TYPE_LINEAR"
MODEL_TYPE_TODO ="MODEL_TYPE_TODO"

MODEL_ARCHITECTURE_LINEAR = {
    "model_type": MODEL_TYPE_LINEAR,
    "process_type": PROCESS_LINEAR_FULL,
}

MODEL_ARCHITECTURE_TODO = {
    "model_type": MODEL_TYPE_TODO,
    "process_type": PROCESS_TODO,
}