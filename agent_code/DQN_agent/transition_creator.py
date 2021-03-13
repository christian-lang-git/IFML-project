from .preprocessing import *

def generate_8_transitions(old_state, action_index, new_state):
    """
    Generates 8 transitions via rotation and mirroring.
    Reward and termination flag are not included since they are constant for all 8 transitions.
    """    
    #the original transition
    original = [old_state, action_index, new_state]
    #counter clockwise rotations
    rot_90 = rotate_90_linear(original)
    rot_180 = rotate_90_linear(rot_90)
    rot_270 = rotate_90_linear(rot_180)
    #mirroring
    original_m = mirror_linear(original)
    rot_90_m = mirror_linear(rot_90)
    rot_180_m = mirror_linear(rot_180)
    rot_270_m = mirror_linear(rot_270)
    #generate and return list of all 8 transitions
    transition_list = [
        original,
        rot_90,
        rot_180,
        rot_270,
        original_m,
        rot_90_m,
        rot_180_m,
        rot_270_m
    ]
    return transition_list

def rotate_90_linear(transition):
    """
    Rotates a transition for processing type PROCESS_LINEAR counter clockwise.
    Reward and termination flag are not included since they are constant for all 8 transitions.
    """
    #extract components of transition
    old_features = transition[0]
    action_index = transition[1]
    new_features = transition[2]
    #apply rotation to components of transition
    old_features_rot = rotate_90_linear_features(old_features)
    action_index_rot = rotate_90_action(action_index)
    new_features_rot = rotate_90_linear_features(new_features)
    #return transition as list
    return [old_features_rot, action_index_rot, new_features_rot]


def rotate_90_action(action_index):
    """
    Rotates an action counter clockwise.
    """
    action = ACTIONS[action_index]
    if action == ACTION_LEFT:
        action_rot = ACTION_DOWN
    elif action == ACTION_DOWN:
        action_rot = ACTION_RIGHT
    elif action == ACTION_RIGHT:
        action_rot = ACTION_UP
    elif action == ACTION_UP:
        action_rot = ACTION_LEFT
    else:
        return action_index
    action_rot_index = INVERSE_ACTIONS[action_rot]
    return action_rot_index

def rotate_90_linear_features(features):
    """
    Rotates features for processing type PROCESS_LINEAR counter clockwise.
    """
    features_rot = np.copy(features)

    try:
        for offset in LINEAR_LIST_PLAYER_INDICES:        
            #the data in the linear processing is always in the order player=0, left=1, right=2, up=3, down=4
            #left=1 element stores up=3
            features_rot[offset+1] = features[offset+3]
            #right=2 element stores down=4
            features_rot[offset+2] = features[offset+4]
            #up=3 element stores right=2
            features_rot[offset+3] = features[offset+2]
            #down=4 element stores left=1
            features_rot[offset+4] = features[offset+1]
    except:
        print("ERROR: ", features)

    return features_rot


def mirror_linear(transition):
    """
    Mirrors a transition along the horizontal axis (vertical flip) for processing type PROCESS_LINEAR.
    Reward and termination flag are not included since they are constant for all 8 transitions.
    """
    #extract components of transition
    old_features = transition[0]
    action_index = transition[1]
    new_features = transition[2]
    #apply mirroring to components of transition
    old_features_m = mirror_linear_features(old_features)
    action_index_m = mirror_action(action_index)
    new_features_m = mirror_linear_features(new_features)
    #return transition as list
    return [old_features_m, action_index_m, new_features_m]

def mirror_action(action_index):
    """
    Mirrors an action along the horizontal axis (vertical flip).
    """
    action = ACTIONS[action_index]
    if action == ACTION_UP:
        action_m = ACTION_DOWN
    elif action == ACTION_DOWN:
        action_m = ACTION_UP
    else:
        return action_index
    action_m_index = INVERSE_ACTIONS[action_m]
    return action_m_index

def mirror_linear_features(features):
    """
    Mirrors features along the horizontal axis (vertical flip) for processing type PROCESS_LINEAR.
    """    
    features_m = np.copy(features)

    for offset in LINEAR_LIST_PLAYER_INDICES:        
        #the data in the linear processing is always in the order player=0, left=1, right=2, up=3, down=4
        #up=3 element stores down=4
        features_m[offset+3] = features[offset+4]
        #down=4 element stores up=3
        features_m[offset+4] = features[offset+3]

    return features_m