import copy

class Cache():

    def __init__(self):       
        print("__init__ Cache")     
        #store parameters
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

    def get_data(self, process_type, turn_index):
        #print("try to get data from turn: ", turn_index, "(", process_type, ")")
        if turn_index == self.turn_index_old:
            #print("index matches old")
            if process_type in self.data_old:
                #print("has data for ", process_type)
                return True, self.data_old[process_type]
            else:                
                #print("has no data for ", process_type)
                return False, None
        elif turn_index == self.turn_index_old + 1:
            #print("index matches new")
            if process_type in self.data_new:
                #print("has data for ", process_type)
                return True, self.data_new[process_type]
            else:                
                #print("has no data for ", process_type)
                return False, None
        else:
            #print("index error")
            self.advance_turn()
            return False, None
            #if turn_index == self.turn_index_old:
            #    print("index matches old")
            #elif turn_index == self.turn_index_old + 1:
            #    print("index matches new")
            

    def set_data(self, process_type, turn_index, data):
        #print("try to set data for turn: ", turn_index, "(", process_type, ")")
        if turn_index == self.turn_index_old + 1:
            #print("index matches new")
            self.data_new[process_type] = data
        else:
            #this will probably not happen
            print("set_data: index error")
        #return self.data_old[process_type]