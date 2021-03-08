import pickle
import math
import numpy as np
from collections import deque
from .dqn import *
from .bomberman import *
from ._parameters import HYPER_PARAMETERS_PROCESSING
from .additional_definitions import *

CRATE_DISTANCE_DISCOUNT_FACTOR = HYPER_PARAMETERS_PROCESSING["CRATE_DISTANCE_DISCOUNT_FACTOR"]
COIN_DISTANCE_DISCOUNT_FACTOR = HYPER_PARAMETERS_PROCESSING["COIN_DISTANCE_DISCOUNT_FACTOR"]

PRE_INDEX_PLAYERS = 0
PRE_INDEX_COIN_VALUES = 1
PRE_INDEX_DANGER_REPULSOR = 2
PRE_INDEX_BOMB_TIME_FIELD = 3
PRE_INDEX_SAFETY_TIME_FIELD = 4
PRE_INDEX_FIELD = 5
PRE_INDEX_CRATE_POTENTIAL_SCALED = 6
PRE_INDEX_CRATE_VALUE = 7

LINEAR_INDEX_BOMB_STATUS = 0
LINEAR_INDEX_COIN_VALUE_PLAYER = 1
LINEAR_INDEX_CRATE_VALUE_PLAYER = 6
LINEAR_INDEX_CRATE_POTENTIAL_PLAYER = 11
LINEAR_INDEX_DANGER_PLAYER = 16

LINEAR_LIST_PLAYER_INDICES = [
    LINEAR_INDEX_COIN_VALUE_PLAYER, 
    LINEAR_INDEX_CRATE_VALUE_PLAYER,
    LINEAR_INDEX_CRATE_POTENTIAL_PLAYER,
    LINEAR_INDEX_DANGER_PLAYER
    ]

def preprocess(game_state: dict, processing_cache: dict, plot: bool) -> np.array:
    #region extract game state
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    agent_data = game_state['self']
    bombs = game_state['bombs']
    coins = game_state['coins']
    enemies = game_state['others']
    a_x = agent_data[3][0]
    a_y = agent_data[3][1]
    bomb_ready = 1 if agent_data[2] else 0
    #endregion

    #print(enemies)

    #region generate arrays to store the features
    players = np.zeros_like(field, dtype=np.float32)
    f_bombs = -1 * np.ones_like(field, dtype=np.float32)
    f_coins = -1 * np.ones_like(field, dtype=np.float32)
    #endregion

    #region process data and write into arrays
    #player position
    players[a_x, a_y] = 1
    #enemy positions
    for enemy in enemies:
        e_x = enemy[3][0]
        e_y = enemy[3][1]
        players[e_x, e_y] = -1
        pass

    for bomb in bombs:
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        f_bombs[b_x,b_y] = bomb[1]

    for coin in coins:
        c_x = coin[0]
        c_y = coin[1]
        f_coins[c_x,c_y] = 1
    #endregion
        
    #region stack the arrays
    stacked = np.stack((field, players, explosion_map, f_bombs, f_coins))
    #endregion

    coin_attractor, crate_potential_scaled, crate_value = preprocess_coin_and_crates(field=field, coins=coins, processing_cache=processing_cache)

    bomb_time_field, bomb_set = preprocess_bomb_time_field(field=field, bombs=bombs)
    safety_time_field = preprocess_safety_time_field(field=field, bombs=bombs, bomb_time_field=bomb_time_field, bomb_set=bomb_set)
    danger_repulsor = preprocess_danger_repulsor(field=field, explosion_map=explosion_map, bombs=bombs, bomb_time_field=bomb_time_field, bomb_set=bomb_set, safety_time_field=safety_time_field)
    
    #print("preprocess")
    preprocessing_result = np.stack((players, coin_attractor, danger_repulsor, bomb_time_field, safety_time_field, field, crate_potential_scaled, crate_value))
    if plot:
        store_preprocessing_result(preprocessing_result)

    return preprocessing_result

def preprocess_coin_and_crates(field, coins, processing_cache):
    use_cache_crates = False
    use_cache_coins = False

    #check what needs to be recalculated
    if "field" in processing_cache:
        if np.array_equal(field, processing_cache["field"]):
            #thats everything needed for crates
            use_cache_crates = True
            #further checks for coins
            use_cache_coins = True
            dict_coins = {}
            for coin in processing_cache["coins"]:
                dict_coins[coin] = 1
            for coin in coins:
                dict_coins[coin] = 2 if coin in dict_coins.keys() else 1
            for val in dict_coins.values():
                if val == 1:
                    use_cache_coins = False


    #use old crate data or recalculate
    if use_cache_crates:
        crate_potential_scaled = processing_cache["crate_potential_scaled"]
        crate_value = processing_cache["crate_value"]
    else:
        #print("recalculate crates")
        crate_potential_scaled, crate_value = preprocess_crate_data(field=field, processing_cache=processing_cache)
        #set cache to new values
        processing_cache["field"] = field
        processing_cache["crate_potential_scaled"] = crate_potential_scaled
        processing_cache["crate_value"] = crate_value

    #use old coin data or recalculate
    if use_cache_coins:
        coin_attractor = processing_cache["coin_attractor"]
    else:
        #print("recalculate coins")
        coin_attractor = preprocess_coin_attractor(field=field, coins=coins, processing_cache=processing_cache)
        #set cache to new values
        processing_cache["coins"] = coins
        processing_cache["coin_attractor"] = coin_attractor

    return coin_attractor, crate_potential_scaled, crate_value
"""
Calculates the coin value of each free tile (no wall or crate)
"""
def preprocess_coin_attractor(field, coins, processing_cache):
    #data_changed = True
    coordinate_deque = deque()
    current_level_size = 0
    #new_cache_coin_field = -1 * np.ones_like(field, dtype=np.int32)
    for coin in coins:
        current_level_size += 1
        coordinate_deque.append(coin)
        #new_cache_coin_field[coin[0], coin[1]] = 1

    """
    #check cache
    if "coin_field" in processing_cache:
        data_changed = False
        #backwards check:
        old_cache_coin_field = processing_cache["coin_field"]
        for coin in coins:
            if old_cache_coin_field[coin[0], coin[1]] != 1:
                data_changed = True
        #forward check: 
        for coin in processing_cache["coins"]:
            if new_cache_coin_field[coin[0], coin[1]] != 1:
                data_changed = True

    #if the data has not changed, return cached data
    if not data_changed:
        #print("data has not changed")
        return processing_cache["coin_attractor"]
    
    #print("data changed")

    #update cache
    processing_cache["coin_field"] = new_cache_coin_field
    processing_cache["coins"] = coins
    """
    #calculate coin attractor
    level = 0
    level_value = 1
    distance_nearest_coin = -1 * np.ones_like(field, dtype=np.int32)
    coin_attractor = -1 * np.ones_like(field, dtype=np.float32) 
    while current_level_size > 0:
        next_level_size = 0
        while current_level_size > 0:
            current_level_size -= 1
            coords = coordinate_deque.popleft()
            not_visited = distance_nearest_coin[coords[0], coords[1]] < 0
            smaller = level < distance_nearest_coin[coords[0], coords[1]]
            if not_visited or smaller:
                distance_nearest_coin[coords[0], coords[1]] = level
                coin_attractor[coords[0], coords[1]] = level_value
                coords_left = (coords[0]-1, coords[1])
                coords_right = (coords[0]+1, coords[1])
                coords_up = (coords[0], coords[1]-1)
                coords_down = (coords[0], coords[1]+1)
                if check_free_tile(coords_left, field):
                    next_level_size += 1
                    coordinate_deque.append(coords_left)
                if check_free_tile(coords_right, field):
                    next_level_size += 1
                    coordinate_deque.append(coords_right)
                if check_free_tile(coords_up, field):
                    next_level_size += 1
                    coordinate_deque.append(coords_up)
                if check_free_tile(coords_down, field):
                    next_level_size += 1
                    coordinate_deque.append(coords_down)
        current_level_size = next_level_size
        level += 1
        level_value *= COIN_DISTANCE_DISCOUNT_FACTOR

    coin_attractor_2 = np.zeros_like(field, dtype=np.float32)
    coin_attractor_2[coin_attractor > 0] = coin_attractor[coin_attractor > 0]
    coin_attractor_2[field < 0] = -1
    coin_attractor_2[field > 0] = -1

    #update cache and return
    #processing_cache["coin_attractor"] = coin_attractor_2
    return coin_attractor_2

"""
Calculates the danger value of each tile (no wall)
"""
def preprocess_danger_repulsor(field, explosion_map, bombs, bomb_time_field, bomb_set, safety_time_field):   
    danger_repulsor =  np.zeros_like(field, dtype=np.float32)
    for c in bomb_set:
        #danger_repulsor[c] = safety_time_field[c] - bomb_time_field[c]
        danger_repulsor[c] = np.divide((safety_time_field[c]+1), (bomb_time_field[c]+1))
        #danger_repulsor[c] = np.divide(safety_time_field[c] - bomb_time_field[c], 3)
    
    danger_repulsor = np.clip(danger_repulsor + explosion_map, a_min=0, a_max=1)
    return danger_repulsor

"""
Calculates the minimum bomb time for each tile (no wall)
(turns until explosion)
"""
def preprocess_bomb_time_field(field, bombs):
    x_size = field.shape[0]  
    y_size = field.shape[1]

    bomb_time_field = -1 * np.ones_like(field)
    bomb_set = set()
    for bomb in bombs:
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        c = bomb[0]
        bomb_time = bomb[1]        
        #set bomb time at bomb position
        bomb_time_field[b_x,b_y] = bomb_time
        bomb_set.add(c)

    for bomb in bombs:
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        c = bomb[0]
        bomb_time = bomb[1]        
        #propagate bomb time in positive x direction
        for i in range(BOMB_RANGE):
            e_x = b_x+i
            c = (e_x, b_y)
            if e_x >= x_size:
                break
            if field[c] == -1:
                break
            #if this tile was already visited, use the minimum value
            if bomb_time_field[c] >= 0:
                bomb_time_field[c] = min(bomb_time_field[c], bomb_time)
            #otherwise we can simply use the current value
            #since this tile was visited for the first time, it is added to the list
            else:
                bomb_time_field[c] = bomb_time
                bomb_set.add(c)           
        #propagate bomb time in negative x direction
        for i in range(BOMB_RANGE):
            e_x = b_x-i
            c = (e_x, b_y)
            if e_x < 0:
                break
            if field[c] == -1:
                break
            #if this tile was already visited, use the minimum value
            if bomb_time_field[c] >= 0:
                bomb_time_field[c] = min(bomb_time_field[c], bomb_time)
            #otherwise we can simply use the current value
            #since this tile was visited for the first time, it is added to the list
            else:
                bomb_time_field[c] = bomb_time
                bomb_set.add(c)
        #propagate bomb time in positive y direction
        for i in range(BOMB_RANGE):
            e_y = b_y+i
            c = (b_x, e_y)
            if e_y >= y_size:
                break
            if field[c] == -1:
                break
            #if this tile was already visited, use the minimum value
            if bomb_time_field[c] >= 0:
                bomb_time_field[c] = min(bomb_time_field[c], bomb_time)
            #otherwise we can simply use the current value
            #since this tile was visited for the first time, it is added to the list
            else:
                bomb_time_field[c] = bomb_time
                bomb_set.add(c)
        #propagate bomb time in negative y direction
        for i in range(BOMB_RANGE):
            e_y = b_y-i
            c = (b_x, e_y)
            if e_y >= y_size:
                break
            if field[c] == -1:
                break
            #if this tile was already visited, use the minimum value
            if bomb_time_field[c] >= 0:
                bomb_time_field[c] = min(bomb_time_field[c], bomb_time)
            #otherwise we can simply use the current value
            #since this tile was visited for the first time, it is added to the list
            else:
                bomb_time_field[c] = bomb_time
                bomb_set.add(c)
    
    return bomb_time_field, bomb_set

"""
Calculates the minimum safety time for each tile (no wall)
(turns required to reach safe tile)
"""
def preprocess_safety_time_field(field, bombs, bomb_time_field, bomb_set):
    x_size = field.shape[0]  
    y_size = field.shape[1]

    safety_time_field = -1 * np.ones_like(field)
    safety_time_field_2 = -1 * np.ones_like(field)

    for c in bomb_set:
        safety_time_field[c] = 0

    for i in range(BOMB_POWER):
        safety_time_field_new = safety_time_field_2 if i % 2 == 0 else safety_time_field
        safety_time_field_old = safety_time_field if i % 2 == 0 else safety_time_field_2
        for c in bomb_set: 
            safety_time_field_new[c] = 1 + get_smallest_free_neighbor(field, c, safety_time_field_old)

    #for c in bomb_set:        
    #    safety_time_field_2[c] = 1 + get_smallest_free_neighbor(field, c, safety_time_field)

    #for c in bomb_set:        
    #    safety_time_field[c] = 1 + get_smallest_free_neighbor(field, c, safety_time_field_2)

    #for c in bomb_set:        
    #    safety_time_field_2[c] = 1 + get_smallest_free_neighbor(field, c, safety_time_field)
        
    return safety_time_field_new    

def get_smallest_free_neighbor(field, coords, array):
    coords_left = (coords[0]-1, coords[1])
    coords_right = (coords[0]+1, coords[1])
    coords_up = (coords[0], coords[1]-1)
    coords_down = (coords[0], coords[1]+1)

    free_list = []
    if check_free_tile(coords_left, field):
        free_list.append(array[coords_left])
    if check_free_tile(coords_right, field):
        free_list.append(array[coords_right])
    if check_free_tile(coords_up, field):
        free_list.append(array[coords_up])
    if check_free_tile(coords_down, field):
        free_list.append(array[coords_down])

    if len(free_list) == 0:
        return -1
    m = min(free_list)
    return m
    #return m if m >= 0 else 0

"""
Calculates the crate potential and crate value of each tile (no crate or wall)
"""
def preprocess_crate_data(field, processing_cache):
    crate_potential = preprocess_crate_potential(field=field)
    #crate_potential_scaled = crate_potential.astype(dtype=np.float32)
    crate_potential_scaled = np.divide(crate_potential,10)
    #crate_potential_scaled = -1 * np.ones_like(field, dtype=np.float32)
    crate_potential_scaled_2 = np.zeros_like(field, dtype=np.float32)
    crate_potential_scaled_2[field < 0] = -1
    crate_potential_scaled_2[field > 0] = -1
    crate_potential_scaled_2[crate_potential_scaled > 0] = crate_potential_scaled[crate_potential_scaled > 0]
    crate_value = preprocess_crate_value(field=field, crate_potential=crate_potential)
    crate_value_2 = np.zeros_like(field, dtype=np.float32)
    crate_value_2[field < 0] = -1
    crate_value_2[field > 0] = -1
    crate_value_2[crate_value > 0] = crate_value[crate_value > 0]
    return crate_potential_scaled_2, crate_value_2

"""
Calculates the crate potential of each tile (no crate or wall)
"""
def preprocess_crate_potential(field):   
    x_size = field.shape[0]  
    y_size = field.shape[1]
    crate_potential = np.zeros_like(field, dtype=np.int32)
    for x in range(x_size):
        for y in range(y_size):
            #check for crate or wall
            if field[x, y] < -0.1 or field[x, y] > 0.1:
                continue

            #propagate potential bomb in positive x direction
            for i in range(BOMB_RANGE):
                e_x = x+i
                c = (e_x, y)
                if e_x >= x_size:
                    break
                if field[c] == -1:
                    break
                #check for crate
                if field[c] == 1:
                    crate_potential[x, y] += 1
            #propagate potential bomb in negative x direction
            for i in range(BOMB_RANGE):
                e_x = x-i
                c = (e_x, y)
                if e_x >= x_size:
                    break
                if field[c] == -1:
                    break
                #check for crate
                if field[c] == 1:
                    crate_potential[x, y] += 1
            #propagate potential bomb in positive y direction
            for i in range(BOMB_RANGE):
                e_y = y+i
                c = (x, e_y)
                if e_y >= y_size:
                    break
                if field[c] == -1:
                    break
                #check for crate
                if field[c] == 1:
                    crate_potential[x, y] += 1
            #propagate potential bomb in negative y direction
            for i in range(BOMB_RANGE):
                e_y = y-i
                c = (x, e_y)
                if e_y >= y_size:
                    break
                if field[c] == -1:
                    break
                #check for crate
                if field[c] == 1:
                    crate_potential[x, y] += 1

    #TODO this is just a temporary fix to discourage bombs at the corners
    crate_potential[1,1] = 0
    crate_potential[1,y_size-2] = 0
    crate_potential[x_size-2,1] = 0
    crate_potential[x_size-2,y_size-2] = 0

    return crate_potential

"""
Calculates the crate value of each free tile (no crate or wall)
"""
def preprocess_crate_value(field, crate_potential):
    #crate_value = crate_potential.astype(dtype=np.float32)
    #crate_value = np.divide(crate_value,10)
    crate_value = -1 * np.ones_like(crate_potential, dtype=np.float32)
    max_count = np.max(crate_potential)
    #print(max_count)
    
    #the deque used to propagate the potential
    coordinate_deque = deque()

    #start with the highest crate count
    crate_count = max_count
    #for each crate count, propagate value if greater
    while crate_count > 0:
        level_value = np.divide(crate_count, 10)
        #print("crate_count", crate_count, "level_value", level_value)

        #find all coordinates where the potential is equal to the current crate count
        filter_array = np.where(crate_potential==crate_count)
        coord_list = list(zip(*filter_array))
        #fill deque with all starting coordinates
        current_level_size = 0
        for c in coord_list:
            current_level_size += 1
            coordinate_deque.append(c)

        while current_level_size > 0:
            next_level_size = 0
            while current_level_size > 0:
                current_level_size -= 1
                coords = coordinate_deque.popleft()
                not_visited = crate_value[coords[0], coords[1]] < 0
                smaller = crate_value[coords[0], coords[1]] < level_value
                if not_visited or smaller:
                    crate_value[coords[0], coords[1]] = level_value
                    coords_left = (coords[0]-1, coords[1])
                    coords_right = (coords[0]+1, coords[1])
                    coords_up = (coords[0], coords[1]-1)
                    coords_down = (coords[0], coords[1]+1)
                    if check_free_tile(coords_left, field):
                        next_level_size += 1
                        coordinate_deque.append(coords_left)
                    if check_free_tile(coords_right, field):
                        next_level_size += 1
                        coordinate_deque.append(coords_right)
                    if check_free_tile(coords_up, field):
                        next_level_size += 1
                        coordinate_deque.append(coords_up)
                    if check_free_tile(coords_down, field):
                        next_level_size += 1
                        coordinate_deque.append(coords_down)
            current_level_size = next_level_size
            level_value *= CRATE_DISTANCE_DISCOUNT_FACTOR

        #TODO set crate count for next iteration
        crate_count-=1
    
    #crate_value[coord_array] = 1
    return crate_value


"""
Check if the tile is free (no crate or wall)
"""
def check_free_tile(coords, field):
    return field[coords[0], coords[1]] > -0.1 and field[coords[0], coords[1]] < 0.1

def store_preprocessing_result(preprocessing_result):
    with open("preprocessing_results.pt", "wb") as file:
        pickle.dump(preprocessing_result, file)

def process(game_state: dict, preprocessing_result, process_type) -> np.array:
    switch = {
        PROCESS_LINEAR: process_linear,
        PROCESS_LINEAR_SMALL: process_linear_small,
        PROCESS_CONVOLUTION: process_convolution,
    }
    func = switch.get(process_type, "invalid_function")
    #print("process: ", process_type, func)
    return func(game_state, preprocessing_result)

def process_linear(game_state: dict, preprocessing_result) -> np.array:
    #print("process_linear")
    #print("A")
    #extract agent data
    agent_data = game_state['self']
    agent_coords = agent_data[3]
    a_x = agent_coords[0]
    a_y = agent_coords[1]
    bomb_status = agent_data[2]

    coords_left = (agent_coords[0]-1, agent_coords[1])
    coords_right = (agent_coords[0]+1, agent_coords[1])
    coords_up = (agent_coords[0], agent_coords[1]-1)
    coords_down = (agent_coords[0], agent_coords[1]+1)

    coin_values = preprocessing_result[PRE_INDEX_COIN_VALUES]
    coin_value_player = coin_values[agent_coords]
    coin_value_left = coin_values[coords_left]
    coin_value_right = coin_values[coords_right]
    coin_value_up = coin_values[coords_up]
    coin_value_down = coin_values[coords_down]

    crate_values = preprocessing_result[PRE_INDEX_CRATE_VALUE]
    crate_value_player = crate_values[agent_coords]
    crate_value_left = crate_values[coords_left]
    crate_value_right = crate_values[coords_right]
    crate_value_up = crate_values[coords_up]
    crate_value_down = crate_values[coords_down]

    crate_potentials = preprocessing_result[PRE_INDEX_CRATE_POTENTIAL_SCALED]
    crate_potential_player = crate_potentials[agent_coords]
    crate_potential_left = crate_potentials[coords_left]
    crate_potential_right = crate_potentials[coords_right]
    crate_potential_up = crate_potentials[coords_up]
    crate_potential_down = crate_potentials[coords_down]

    danger_values = preprocessing_result[PRE_INDEX_DANGER_REPULSOR]
    danger_value_player = danger_values[agent_coords]
    danger_value_left = danger_values[coords_left]
    danger_value_right = danger_values[coords_right]
    danger_value_up = danger_values[coords_up]
    danger_value_down = danger_values[coords_down]

    return np.array([
        bomb_status,
        coin_value_player, 
        coin_value_left, 
        coin_value_right, 
        coin_value_up, 
        coin_value_down,
        crate_value_player,
        crate_value_left,
        crate_value_right,
        crate_value_up,
        crate_value_down,
        crate_potential_player,
        crate_potential_left,
        crate_potential_right,
        crate_potential_up,
        crate_potential_down,
        danger_value_player,
        danger_value_left,
        danger_value_right,
        danger_value_up,
        danger_value_down,
        ])

def process_linear_small(game_state: dict, preprocessing_result) -> np.array:
    #print("process_linear")
    #print("A")
    #extract agent data
    agent_data = game_state['self']
    agent_coords = agent_data[3]
    a_x = agent_coords[0]
    a_y = agent_coords[1]
    bomb_status = agent_data[2]

    coords_left = (agent_coords[0]-1, agent_coords[1])
    coords_right = (agent_coords[0]+1, agent_coords[1])
    coords_up = (agent_coords[0], agent_coords[1]-1)
    coords_down = (agent_coords[0], agent_coords[1]+1)

    coin_values = preprocessing_result[PRE_INDEX_COIN_VALUES]
    coin_value_player = coin_values[agent_coords]
    coin_value_left = coin_values[coords_left]
    coin_value_right = coin_values[coords_right]
    coin_value_up = coin_values[coords_up]
    coin_value_down = coin_values[coords_down]

    return np.array([        
        coin_value_player, 
        coin_value_left, 
        coin_value_right, 
        coin_value_up, 
        coin_value_down,
        ])

def process_convolution(game_state: dict, preprocessing_result) -> np.array:
    #print("process_convolution")
    return np.stack((preprocessing_result[PRE_INDEX_PLAYERS], preprocessing_result[PRE_INDEX_COIN_VALUES]))

