import os
import pickle
import torch as T
import torch.nn as nn
import torch.nn.functional as F

DQN_TYPE_L2 = "DQN_TYPE_L2"
DQN_TYPE_L3 = "DQN_TYPE_L3"
DQN_TYPE_C1L2 = "DQN_TYPE_C1L2"
DQN_TYPE_C3L1 = "DQN_TYPE_C3L1"
DQN_TYPE_C3L2 = "DQN_TYPE_C3L2"

"""
Deep Q Learning Neural Network class with 2 hidden layers
"""
class DQN(nn.Module):
    """
    :param input_dims: TODO
    """
    def __init__(self, model_architecture):#TODO maybe rename to n_in, n_h1, n_h2, n_out
    #def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):#TODO maybe rename to n_in, n_h1, n_h2, n_out        
        super(DQN, self).__init__()   
        print("__init__ DQN") 
        print("input: ", model_architecture["dim_input"])         
        #store parameters
        self.dim_input = model_architecture["dim_input"]
        self.dim_output = model_architecture["dim_output"]        

    def forward(self, state):
        """
        Text
        :param state: the state TODO.
        :return actions: estimated value of each action. TODO: value is probably the wrong word here.
        """
        pass

    """
    Copies the state (theta) of the provided other DQN.
    Since python does not support constructor overloading, this seems more readable than passing *args to the constructor
    :param other: the DQN whose state (theta) should be copied into this DQN
    """
    def copy_from(self, other):
        theta = other.state_dict()
        self.load_state_dict(theta)

    def save(self, path):
        print(f"save agent: {path}")
        with open(path, "wb") as file:
            pickle.dump(self.state_dict(), file)

    def load(self, path):
        print(f"load agent: {path}")
        if not os.path.isfile(path):
            print(f"could not find file: {path}")
            return

        try:
            with open(path, "rb") as file:
                theta = pickle.load(file)
                self.load_state_dict(theta)
        except:
            print(f"error loading agent: {path}")

class DQN_L2(DQN):
    def __init__(self, model_architecture):
        super(DQN_L2, self).__init__(model_architecture)        
        print("__init__ DQN_L2")
        #store parameters
        self.dim_layer_full_1 = model_architecture["dim_layer_full_1"]
        self.dim_layer_full_2 = model_architecture["dim_layer_full_2"]
        #setup layers
        self.layer_full_1 = nn.Linear(*self.dim_input, self.dim_layer_full_1)
        self.layer_full_2 = nn.Linear(self.dim_layer_full_1, self.dim_layer_full_2)
        self.layer_full_3 = nn.Linear(self.dim_layer_full_2, self.dim_output)

    def forward(self, state):       
        #print("forward DQN_L2")
        x = F.relu(self.layer_full_1(state))
        #print(x.size())
        x = F.relu(self.layer_full_2(x))
        #print(x.size())
        actions = self.layer_full_3(x)
        #print(actions.size())
        return actions

class DQN_L3(DQN):
    def __init__(self, model_architecture):
        super(DQN_L3, self).__init__(model_architecture)        
        print("__init__ DQN_L3")
        #store parameters
        self.dim_layer_full_1 = model_architecture["dim_layer_full_1"]
        self.dim_layer_full_2 = model_architecture["dim_layer_full_2"]
        self.dim_layer_full_3 = model_architecture["dim_layer_full_3"]
        #setup layers
        self.layer_full_1 = nn.Linear(*self.dim_input, self.dim_layer_full_1)
        self.layer_full_2 = nn.Linear(self.dim_layer_full_1, self.dim_layer_full_2)
        self.layer_full_3 = nn.Linear(self.dim_layer_full_2, self.dim_layer_full_3)
        self.layer_full_4 = nn.Linear(self.dim_layer_full_3, self.dim_output)

    def forward(self, state):       
        #print("forward DQN_L2")
        x = F.relu(self.layer_full_1(state))
        #print(x.size())
        x = F.relu(self.layer_full_2(x))
        #print(x.size())
        x = F.relu(self.layer_full_3(x))
        #print(x.size())
        actions = self.layer_full_4(x)
        #print(actions.size())
        return actions

class DQN_C1L2(DQN):
    def __init__(self, model_architecture):
        super(DQN_C1L2, self).__init__(model_architecture)        
        print("__init__ DQN_C1L2")
        #store parameters
        self.dim_layer_full_1 = model_architecture["dim_layer_full_1"]
        self.dim_layer_full_2 = model_architecture["dim_layer_full_2"]
        #setup layers
        channels = 6
        self.layer_conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1)#17x17 --> 13x13
        self.layer_full_1 = nn.Linear(channels*13*13, self.dim_layer_full_1)
        self.layer_full_2 = nn.Linear(self.dim_layer_full_1, self.dim_layer_full_2)
        self.layer_full_3 = nn.Linear(self.dim_layer_full_2, self.dim_output)

    def forward(self, state):       
        #print("forward DQN_C1L2")
        #print("forward:", state.size())
        x = F.relu(self.layer_conv_1(state))
        #tensor needs to be flattened to 2d to work with fully connected layer
        #first dimension for batch, second for the flattened part
        x = T.flatten(x, start_dim=1)
        #print(x.size())
        x = F.relu(self.layer_full_1(x))
        #print(x.size())
        x = F.relu(self.layer_full_2(x))
        #print(x.size())
        actions = self.layer_full_3(x)
        #print(actions.size())
        return actions

class DQN_C3L1(DQN):
    def __init__(self, model_architecture):
        super(DQN_C3L1, self).__init__(model_architecture)        
        print("__init__ DQN_C3L1")
        #store parameters
        self.dim_layer_full_1 = model_architecture["dim_layer_full_1"]
        #setup layers
        channels = 6
        self.layer_conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1)#17x17 --> 13x13
        self.layer_conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)#13x13 --> 11x11
        self.layer_conv_3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)#13x13 --> 9x9
        self.layer_full_1 = nn.Linear(channels*9*9, self.dim_layer_full_1)
        self.layer_full_2 = nn.Linear(self.dim_layer_full_1, self.dim_output)

    def forward(self, state):       
        #print("forward DQN_C1L2")
        #print("forward:", state.size())
        x = F.relu(self.layer_conv_1(state))
        #print("A", x.size())
        x = F.relu(self.layer_conv_2(x))
        #print("B", x.size())
        x = F.relu(self.layer_conv_3(x))
        #print("C", x.size())
        #tensor needs to be flattened to 2d to work with fully connected layer
        #first dimension for batch, second for the flattened part
        x = T.flatten(x, start_dim=1)
        #print("D", x.size())
        x = F.relu(self.layer_full_1(x))
        #print("F", x.size())
        actions = self.layer_full_2(x)
        #print(actions.size())
        return actions

class DQN_C3L2(DQN):
    def __init__(self, model_architecture):
        super(DQN_C3L2, self).__init__(model_architecture)        
        print("__init__ DQN_C3L2")
        #store parameters
        self.dim_layer_full_1 = model_architecture["dim_layer_full_1"]
        self.dim_layer_full_2 = model_architecture["dim_layer_full_2"]
        #setup layers
        channels = 6
        self.layer_conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1)#17x17 --> 13x13
        self.layer_conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)#13x13 --> 11x11
        self.layer_conv_3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)#13x13 --> 9x9
        self.layer_full_1 = nn.Linear(channels*9*9, self.dim_layer_full_1)
        self.layer_full_2 = nn.Linear(self.dim_layer_full_1, self.dim_layer_full_2)
        self.layer_full_3 = nn.Linear(self.dim_layer_full_2, self.dim_output)

    def forward(self, state):       
        #print("forward DQN_C1L2")
        #print("forward:", state.size())
        x = F.relu(self.layer_conv_1(state))
        #print("A", x.size())
        x = F.relu(self.layer_conv_2(x))
        #print("B", x.size())
        x = F.relu(self.layer_conv_3(x))
        #print("C", x.size())
        #tensor needs to be flattened to 2d to work with fully connected layer
        #first dimension for batch, second for the flattened part
        x = T.flatten(x, start_dim=1)
        #print("D", x.size())
        x = F.relu(self.layer_full_1(x))
        #print("E", x.size())
        x = F.relu(self.layer_full_2(x))
        #print("F", x.size())
        actions = self.layer_full_3(x)
        #print(actions.size())
        return actions


def Create_DQN(model_architecture): 
    """
    Creates the correct DQN depending on the model_type specified in model_architecture.
    The class is looked up from MODEL_DICT and model_architecture is passed on to the constructor.
    :param model_architecture: a dictionary containing the necessary variables to construct the DQN
    """
    MODEL_DICT = {
        DQN_TYPE_L2: DQN_L2, 
        DQN_TYPE_L3: DQN_L3, 
        DQN_TYPE_C1L2: DQN_C1L2,
        DQN_TYPE_C3L1: DQN_C3L1,
        DQN_TYPE_C3L2: DQN_C3L2,
        }
    return MODEL_DICT[model_architecture["model_type"]](model_architecture)

