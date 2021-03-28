import numpy as np
import torch as T
from .transition_creator import *

class ReplayBuffer():
    """
    The ReplayBuffer handles storage and sampling of transitions for experience replay.
    Implementation as cyclic buffer.
    """

    def __init__(self, capacity, feature_dim, device, exploit_symmetry):
        """
        :param capacity: The number of transitions to be stored
        :feature_dim: dimensions of the feature array
        :device: the used device (cpu or gpu)
        :exploit_symmetry: whether each transition should be transformed to 8 transitions
        """
        #store parameters
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.device = device
        self.exploit_symmetry = exploit_symmetry
        #generate buffers
        self.buffer_old_state = np.zeros((capacity, *feature_dim), dtype=np.float32)
        self.buffer_new_state = np.zeros((capacity, *feature_dim), dtype=np.float32)
        self.buffer_action_index = np.zeros(capacity, dtype=np.int32)
        self.buffer_reward = np.zeros(capacity, dtype=np.float32)
        self.buffer_termination_flag = np.zeros(capacity, dtype=np.bool)
        #additional data
        self.count = 0# the number of transitions stored. This can be bigger than the capacity.

    def check_for_loop(self, old_state, action_index, loop_num_checks):
        """
        check if the game state and action pairing is identical to one of the last prairings
        :param old_state: np array
        :param action_index: integer representation of action
        :param loop_num_checks: number of old pairs to check
        """
        #check if there is no element in the buffer
        if self.count == 0:
            return False

        num_checks = min(loop_num_checks, self.count)        
        for i in range(num_checks):
            var = self.count-1-i
            #print("var", var)
            if var < 0:
                break

            index = var % self.capacity
            _old_state = self.buffer_old_state[index]
            _action_index = self.buffer_action_index[index]
            if (_old_state==old_state).all() and _action_index == action_index:
                return True
        return False
            

    def store(self, old_state, action_index, reward, new_state, termination_flag):  
        """
        Stores a transition
        :param old_state: np array
        :param action_index: integer representation of action
        :param reward: reward for this transition
        :param new_state: np array
        :param termination_flag: whether this is the last turn
        """
        if self.exploit_symmetry and not termination_flag:
            transition_list = generate_8_transitions(old_state=old_state, action_index=action_index, new_state=new_state)
            for i in range(8):
                index = self.count % self.capacity
                self.buffer_old_state[index] = transition_list[i][0]
                self.buffer_action_index[index] = transition_list[i][1]
                self.buffer_reward[index] = reward
                self.buffer_new_state[index] = transition_list[i][2]
                self.buffer_termination_flag[index] = termination_flag
                self.count += 1
        else:
            index = self.count % self.capacity
            self.buffer_old_state[index] = old_state
            self.buffer_action_index[index] = action_index
            self.buffer_reward[index] = reward
            self.buffer_new_state[index] = new_state
            self.buffer_termination_flag[index] = termination_flag
            self.count += 1

    def sample(self, batch_size):
        """
        samples transitions uniformly
        :param batch_size: number of transitions smapled
        """
        if self.count < batch_size:
            return None, None, None, None, None

        #select the sample range. (usually this is capacity, but if the buffer is not full yet, it is count.)
        sample_max = min(self.capacity, self.count)
        #sample batch_size indices
        random_indices = np.random.choice(sample_max, batch_size, replace=False)

        #this is used for indexing --> no tensor
        batch_action_index = self.buffer_action_index[random_indices]
        batch_termination_flag = self.buffer_termination_flag[random_indices]
        #tensors for everything else
        batch_old_state = T.tensor(self.buffer_old_state[random_indices], device=self.device)        
        batch_reward = T.tensor(self.buffer_reward[random_indices], device=self.device)
        batch_new_state = T.tensor(self.buffer_new_state[random_indices], device=self.device)
        return batch_old_state, batch_action_index, batch_reward, batch_new_state, batch_termination_flag