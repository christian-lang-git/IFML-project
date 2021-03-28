import pickle
import numpy as np
from .bomberman import *


class EpochLogger():
    """
    The EpochLogger is used to gather data describing the performance and actions of the agent.
    This allows more robust comparison of agents at different stages of training.
    We use the term epoch to describe a sequence of multiple episodes.
    """

    def __init__(self, name, epoch_length, epsilon_start, epsilon_end, epoch_index_start_decay, epoch_index_stop_decay):
        self.name = name                    #the name of this logger
        self.epoch_length = epoch_length    #the number of episodes for each epoch in this logger
        self.epoch_index = -1               #epoch index
        self.total_round_index = 0          #total round index of this logger
        self.epsilon = epsilon_start        #epsilon-greedy parameter
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epoch_results = []
        self.epoch_index_start_decay = epoch_index_start_decay
        self.epoch_index_stop_decay = epoch_index_stop_decay
        self.round_index_start_decay = epoch_index_start_decay * epoch_length
        self.round_index_stop_decay = epoch_index_stop_decay * epoch_length
        self.number_rounds_decaying = self.round_index_stop_decay - self.round_index_start_decay  #the number of total rounds epsilon is decaying

    def start_epoch(self):
        """
        Sets EpochLogger to the initial state of an epoch and increments epoch index.
        This allows to recycle the same EpochLogger after each epoch.
        """
        self.epoch_index += 1
        print(f"[{self.name}] start epoch: {self.epoch_index}")
        self.current_round_index = 0
        self.round_loss = 0
        self.round_loss_count = 0
        self.array_action_count = np.zeros((self.epoch_length, NUM_ACTIONS), dtype=np.int32)
        self.array_invalid_count = np.zeros((self.epoch_length, NUM_ACTIONS), dtype=np.int32)
        self.array_reward = np.zeros((self.epoch_length), dtype=np.float32)
        self.array_score = np.zeros((self.epoch_length), dtype=np.float32)
        self.array_crates = np.zeros((self.epoch_length), dtype=np.int32)
        self.array_coins = np.zeros((self.epoch_length), dtype=np.int32)
        self.array_kills = np.zeros((self.epoch_length), dtype=np.int32)
        self.array_loop = np.zeros((self.epoch_length), dtype=np.int32)
        self.array_bomb_dropped = np.zeros((self.epoch_length), dtype=np.int32)
        self.array_bad_bomb = np.zeros((self.epoch_length), dtype=np.int32)
        self.array_loss = np.zeros((self.epoch_length), dtype=np.float32)

    def update_round(self, action_index, reward, invalid_action_flag, crates, coins, kills, is_loop, is_bomb_dropped, is_bad_bomb):
        #increment the count of the action index that was chosen
        self.array_action_count[self.current_round_index, action_index] += 1
        #increment the count of the action index that did result in an invalid action event
        if invalid_action_flag:
            self.array_invalid_count[self.current_round_index, action_index] += 1
        #increment loop count if loop was detected
        if is_loop:
            self.array_loop[self.current_round_index] += 1
        #increment bomb dropped count if bomb was dropped successfully
        if is_bomb_dropped:
            self.array_bomb_dropped[self.current_round_index] += 1
        #increment bad bomb count if bad bomb was detected
        if is_bad_bomb:
            self.array_bad_bomb[self.current_round_index] += 1
        #increase the rest
        self.array_reward[self.current_round_index] += reward
        self.array_crates[self.current_round_index] += crates
        self.array_coins[self.current_round_index] += coins
        self.array_kills[self.current_round_index] += kills

        
        
    def add_loss(self, turn_loss):
        self.round_loss += turn_loss
        self.round_loss_count += 1

    def finalize_round(self, score):
        #print(f"[{self.name}] finalize round: {self.current_round_index} with epsilon {self.epsilon}")
        #set the score of the current round
        self.array_score[self.current_round_index] = score
        #set the mean loss of the current round
        self.array_loss[self.current_round_index] = (self.round_loss / self.round_loss_count) if self.round_loss_count > 0 else 0
        #reset loss for next round
        self.round_loss = 0
        self.round_loss_count = 0
        #set index for next round
        self.current_round_index += 1
        self.total_round_index += 1
        #decay epsilon for next round only if decay is active (not during validation) 
        if self.number_rounds_decaying > 0:
            self.set_epsilon_for_round()

        


    def set_epsilon_for_round(self):
        #check if still in non decay phase
        if self.epoch_index < self.epoch_index_start_decay:
            return self.epsilon_start
        #check if more rounds than specified are played.
        #otherwise the LERP below will extrapolate and overshoot the target (epsilon_end)
        if self.epoch_index >= self.epoch_index_stop_decay:
            return self.epsilon_end
        #LERP    
        t = (self.total_round_index - self.round_index_start_decay) / (self.number_rounds_decaying-1)
        self.epsilon = (1-t) * self.epsilon_start + t * self.epsilon_end


    def try_finalize_epoch(self):
        #check for end of epoch
        if self.current_round_index < self.epoch_length:
            return False
        epoch_result = self.finalize_epoch()
        self.store_epoch_result(epoch_result)
        return True       

    def finalize_epoch(self):
        reward = np.mean(self.array_reward)
        score = np.mean(self.array_score)
        mean_crates = np.mean(self.array_crates)
        mean_coins = np.mean(self.array_coins)
        mean_kills = np.mean(self.array_kills)
        mean_loop = np.mean(self.array_loop)
        mean_bomb_dropped = np.mean(self.array_bomb_dropped)
        mean_bad_bomb = np.mean(self.array_bad_bomb)
        mean_loss = np.mean(self.array_loss)
        #calculate how often each action was used on average
        action_mean = np.mean(self.array_action_count, axis=0)
        invalid_mean = np.mean(self.array_invalid_count, axis=0)
        #combine all movement actions
        movement_mean = np.sum(action_mean[MOVEMENT_INDICES])
        invalid_movement_mean = np.sum(invalid_mean[MOVEMENT_INDICES])
        #extract non movement actions
        wait_mean = action_mean[INVERSE_ACTIONS[ACTION_WAIT]]
        invalid_wait_mean = invalid_mean[INVERSE_ACTIONS[ACTION_WAIT]]#this is probably useless
        bomb_mean = action_mean[INVERSE_ACTIONS[ACTION_BOMB]]
        invalid_bomb_mean = invalid_mean[INVERSE_ACTIONS[ACTION_BOMB]]   

        print(f"[{self.name}] finalize epoch: {self.epoch_index} score: {score} reward: {reward}, last epsilon:{self.epsilon}")
        return (reward, score, mean_crates, mean_coins, mean_kills, self.epsilon, movement_mean, wait_mean, bomb_mean, invalid_movement_mean, invalid_wait_mean, invalid_bomb_mean, mean_loss, mean_loop, mean_bomb_dropped, mean_bad_bomb)

    def store_epoch_result(self, epoch_result):
        self.epoch_results.append(epoch_result)
        with open(f"{self.name}_results.pt", "wb") as file:
            pickle.dump(self.epoch_results, file)