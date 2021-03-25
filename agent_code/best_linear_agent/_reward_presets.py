import events as e
from .additional_definitions import *

GAME_REWARDS_PROCESSED = {
    e.COIN_COLLECTED: 1,
    e.KILLED_OPPONENT: 5,
    e.GOT_KILLED: -.5,              #penalty for death (this includes suicide)
    e.INVALID_ACTION: -0.75,        #invalid actions are bad, but should probably not be discouraged too much
    E_DANGER_EXCEEDED: -.5,
    E_DROPPED_BOMB_BAD: -.5,
    E_DROPPED_BOMB_NEAR_ENEMY: 0.5,
}

GAME_REWARD_FACTORS_PROCESSED = {
    E_MOVED_TOWARDS_COIN: 2,
    E_MOVED_AWAY_FROM_COIN: 2.2,
    E_MOVED_TOWARDS_CRATE: 1.5,
    E_MOVED_AWAY_FROM_CRATE: 1.65,
    E_DROPPED_BOMB_NEAR_CRATE: 3,   #this is multiplied with number of crates in bomb range / 10
    E_VISITED_PENALTY: 0.1,
}

GAME_REWARDS_RAW = {
    e.COIN_COLLECTED: 0.1,
    e.KILLED_OPPONENT: 0.5,
    e.CRATE_DESTROYED: 0.01,        
    e.GOT_KILLED: -1,               #penalty for death (this includes suicide)
    e.INVALID_ACTION: -0.002,       
}

GAME_REWARD_FACTORS_RAW = {
    E_VISITED_PENALTY: 0.02,
}