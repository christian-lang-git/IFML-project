import os
import pickle
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math

DQN_TYPE_L2 = "DQN_TYPE_L2"
DQN_TYPE_L3 = "DQN_TYPE_L3"
DQN_TYPE_C1L2 = "DQN_TYPE_C1L2"
DQN_TYPE_C3L1 = "DQN_TYPE_C3L1"
DQN_TYPE_C3L2 = "DQN_TYPE_C3L2"
MODEL_TYPE_LINEAR = "MODEL_TYPE_LINEAR"
MODEL_TYPE_LINEAR_RAW ="MODEL_TYPE_LINEAR_RAW"

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

    def get_conv_out_size(self, input_size, kernel_size, padding, stride):
        """
        see https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer
        """
        return math.floor((input_size - kernel_size + 2 * padding ) / stride + 1)

    """
    Copies the state (theta) of the provided other DQN.
    Since python does not support constructor overloading, this seems more readable than passing *args to the constructor
    :param other: the DQN whose state (theta) should be copied into this DQN
    """
    def copy_from(self, other):
        theta = other.state_dict()
        self.load_state_dict(theta)

    def save(self, path):
        """
        wrapper for saving method
        """
        self.save_torch(path)

    def save_torch(self, path):
        """
        new saving method. resulting file can be loaded without CUDA.
        """
        print(f"save agent via torch: {path}")
        with open(path, "wb") as file:
            T.save(self.state_dict(), file)

    def save_pickle(self, path):
        """
        deprecated.
        old saving method, resulting file can not be loaded without CUDA.
        """
        print(f"save agent via pickle: {path}")
        with open(path, "wb") as file:
            pickle.dump(self.state_dict(), file)

    def load(self, path, device):
        """
        wrapper for saving method.
        first tries to load via torch. (new files)
        if this does not work, tries to load via pickle (old files).
        """
        print(f"load agent: {path}")
        if not os.path.isfile(path):
            print(f"could not find file: {path}")
            return

        try:
            print("try loading agent via torch...")
            self.load_torch(path, device)   
            print("loaded agent successfully via torch")           
        except Exception as e_torch:
            print("error loading agent via torch")
            print("try loading agent via pickle...")
            try:
                self.load_pickle(path)   
                print("loaded agent successfully via pickle")   
                print("WARNING: submission will not work. (pickle loading without CUDA)")
                print("replacing file...")
                self.save(path)        
                print("file replaced successfully")     
            except Exception as e_pickle:
                print("error loading agent via pickle")
                print("output torch error:")
                print(e_torch)
                print("output pickle error:")
                print(e_pickle)

    def load_torch(self, path, device):
        """
        new loading method. file can be loaded without CUDA.
        """
        with open(path, "rb") as file:
            theta = T.load(file, map_location=device)
            self.load_state_dict(theta)

    def load_pickle(self, path):
        """
        deprecated.
        old loading method. file can not be loaded without CUDA.
        """
        print("load agent via pickle")
        with open(path, "rb") as file:
            theta = pickle.load(file)
            self.load_state_dict(theta)

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
        channels = self.dim_input[0]
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
        self.conv_1 = model_architecture["conv_1"]
        self.conv_2 = model_architecture["conv_2"]
        self.conv_3 = model_architecture["conv_3"]
        #setup layers
        in_channels = self.dim_input[0]
        self.layer_conv_1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=self.conv_1["channels"], 
            kernel_size=self.conv_1["kernel_size"], 
            stride=self.conv_1["stride"], 
            padding=self.conv_1["padding"])
        self.layer_conv_2 = nn.Conv2d(
            in_channels=self.conv_1["channels"], 
            out_channels=self.conv_2["channels"], 
            kernel_size=self.conv_2["kernel_size"], 
            stride=self.conv_2["stride"], 
            padding=self.conv_2["padding"])
        self.layer_conv_3 = nn.Conv2d(
            in_channels=self.conv_2["channels"], 
            out_channels=self.conv_3["channels"], 
            kernel_size=self.conv_3["kernel_size"], 
            stride=self.conv_3["stride"], 
            padding=self.conv_3["padding"])

        size_after_conv_1 = self.get_conv_out_size(
            input_size=self.dim_input[1], 
            kernel_size=self.conv_1["kernel_size"],
            padding=self.conv_1["padding"],
            stride=self.conv_1["stride"])
        size_after_conv_2 = self.get_conv_out_size(
            input_size=size_after_conv_1, 
            kernel_size=self.conv_2["kernel_size"],
            padding=self.conv_2["padding"],
            stride=self.conv_2["stride"])
        size_after_conv_3 = self.get_conv_out_size(
            input_size=size_after_conv_2, 
            kernel_size=self.conv_3["kernel_size"],
            padding=self.conv_3["padding"],
            stride=self.conv_3["stride"])     

        flattened = self.conv_3["channels"]*size_after_conv_3*size_after_conv_3

        print("size_after_conv_1", size_after_conv_1)
        print("size_after_conv_2", size_after_conv_2)
        print("size_after_conv_3", size_after_conv_3)
        print("flattened", flattened)

        self.layer_full_1 = nn.Linear(flattened, self.dim_layer_full_1)
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
        channels = self.dim_input[0]
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

