import numpy as np

class LoopDetector():
    """
    The LoopDetector detects loops based on player position and action used.
    """
    def __init__(self, capacity):
        """
        :param capacity: The number of iterations per game
        """
        #store parameters
        self.capacity = capacity
        #generate buffers
        self.buffer = np.zeros((capacity, 3), dtype=np.float32)
        #additional data
        self.count = 0# the number of transitions stored. This can be bigger than the capacity.
    
    """
    """
    def store(self, player_position, action_index):          
        index = self.count % self.capacity
        self.buffer[index] = [*player_position, action_index]
        self.count += 1

        buffer_active = self.buffer[:self.count]

        indices = np.where(
            (buffer_active[:,0] == player_position[0]) & 
            (buffer_active[:,1] == player_position[1]) & 
            (buffer_active[:,2] == action_index)
            )[0]
        loop_count = len(indices)
        return loop_count
        #print("loop_count: ", loop_count)
        #print(buffer_active)

    def reset(self):
        self.count = 0

