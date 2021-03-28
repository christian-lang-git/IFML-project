import copy

class Cache():

    def __init__(self, name):       
        print("__init__ Cache")     
        #store parameters
        self.name = name
        self.data_old = {}
        self.data_new = {}
        self.turn_index_old = -1

    def advance_turn(self):
        self.turn_index_old += 1
        #print("advance_turn to: ", self.turn_index_old)   
        self.data_old = copy.deepcopy(self.data_new)
        self.data_new = {}

    def reset(self):
        #print("reset cache")  
        self.turn_index_old = -1
        self.data_old = {}
        self.data_new = {}

    def get_data(self, process_type, turn_index):
        #try to get data with old turn index
        if turn_index == self.turn_index_old:
            #check if data is available for this turn index
            if process_type in self.data_old:
                return True, self.data_old[process_type]
            #turn found, but no data available
            else:                
                return False, None

        #try to get data with new turn index
        elif turn_index == self.turn_index_old + 1:
            #check if data is available for this turn index
            if process_type in self.data_new:
                return True, self.data_new[process_type]
            #turn found, but no data available
            else:                
                return False, None

        #turn index does not match --> advance turn
        else:
            self.advance_turn()
            return False, None            

    def set_data(self, process_type, turn_index, data):
        if turn_index == self.turn_index_old + 1:
            self.data_new[process_type] = data