import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
plot_active = False

#reward, score, self.epsilon, movement_mean, wait_mean, bomb_mean, invalid_movement_mean, invalid_wait_mean, invalid_bomb_mean
INDEX_REWARD = 0
INDEX_SCORE = 1
INDEX_CRATES = 2
INDEX_COINS = 3
INDEX_KILLS = 4
INDEX_EPSILON = 5
INDEX_MOVEMENT = 6
INDEX_WAIT = 7
INDEX_BOMB = 8
INDEX_INVALID_MOVEMENT = 9
INDEX_INVALID_WAIT = 10
INDEX_INVALID_BOMB = 11
INDEX_LOSS = 12
INDEX_LOOP = 13
INDEX_BOMB_DROPPED = 14
INDEX_BAD_BOMB = 15

COLOR_TRAIN = "lightgray"
COLOR_VALIDATE = "black"

COLOR_BOMB = "red"
COLOR_WAIT = "khaki"
COLOR_MOVEMENT = "cornflowerblue"

size_training = 0
size_validation = 0

def load(file):
    with open(file, "rb") as file:
        data = pickle.load(file)
        return data

def make_percentage(lists, index_a, index_b):
    a = lists[index_a]
    b = lists[index_b]
    a = np.array(a)
    b = np.array(b)
    c = np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)
    return c * 100

def update_plot(data_training, data_validation, fig, axs):
    global size_training
    global size_validation
    #transform data into list of lists
    lists_training = (list(map(list, zip(*data_training))))
    lists_validation = (list(map(list, zip(*data_validation))))

    #check if data changed
    if size_training == len(data_training) and size_validation == len(data_validation):
        return        
    #print("data changed")
    size_training = len(data_training)
    size_validation = len(data_validation)

    list_training_invalid_movement_percentage = make_percentage(lists_training, INDEX_INVALID_MOVEMENT, INDEX_MOVEMENT)
    list_training_invalid_wait_percentage = make_percentage(lists_training, INDEX_INVALID_WAIT, INDEX_WAIT)
    list_training_invalid_bomb_percentage = make_percentage(lists_training, INDEX_INVALID_BOMB, INDEX_BOMB)

    list_validation_invalid_movement_percentage = make_percentage(lists_validation, INDEX_INVALID_MOVEMENT, INDEX_MOVEMENT)
    list_validation_invalid_wait_percentage = make_percentage(lists_validation, INDEX_INVALID_WAIT, INDEX_WAIT)
    list_validation_invalid_bomb_percentage = make_percentage(lists_validation, INDEX_INVALID_BOMB, INDEX_BOMB)
    
    list_training_bad_bomb_percentage = make_percentage(lists_training, INDEX_BAD_BOMB, INDEX_BOMB)
    list_validation_bad_bomb_percentage = make_percentage(lists_validation, INDEX_BAD_BOMB, INDEX_BOMB_DROPPED)

    """
    print("----------------")
    print(lists_training[0])
    print(lists_training[1])
    print(lists_training[2])
    print(lists_training[3])
    print(lists_training[4])
    print(lists_training[5])
    print("")
    print(lists_validation[0])
    print(lists_validation[1])
    print(lists_validation[2])
    print(lists_validation[3])
    print(lists_validation[4])
    print(lists_validation[5])
    """

    for ax in fig.axes:
        ax.cla()
    
    #plot both together col 0 left
    axs[0, 0].set(title="mean reward", xlabel="epoch", ylabel="mean reward")
    axs[0, 0].plot(lists_training[INDEX_REWARD], COLOR_TRAIN)
    axs[0, 0].plot(lists_validation[INDEX_REWARD], COLOR_VALIDATE)

    axs[1, 0].set(title="mean score", xlabel="epoch", ylabel="mean score")
    axs[1, 0].plot(lists_training[INDEX_SCORE], COLOR_TRAIN)
    axs[1, 0].plot(lists_validation[INDEX_SCORE], COLOR_VALIDATE)

    axs[2, 0].set(title="last epsilon", xlabel="epoch", ylabel="last epsilon")
    axs[2, 0].plot(lists_training[INDEX_EPSILON], COLOR_TRAIN)
    axs[2, 0].plot(lists_validation[INDEX_EPSILON], COLOR_VALIDATE)

    #plot both together col 1 center left
    axs[0, 1].set(title="mean crates", xlabel="epoch", ylabel="mean crates")
    axs[0, 1].plot(lists_training[INDEX_CRATES], COLOR_TRAIN)
    axs[0, 1].plot(lists_validation[INDEX_CRATES], COLOR_VALIDATE)

    axs[1, 1].set(title="mean coins", xlabel="epoch", ylabel="mean coins")
    axs[1, 1].plot(lists_training[INDEX_COINS], COLOR_TRAIN)
    axs[1, 1].plot(lists_validation[INDEX_COINS], COLOR_VALIDATE)

    #axs[2, 1].set(title="mean kills", xlabel="epoch", ylabel="last kills")
    #axs[2, 1].plot(lists_training[INDEX_KILLS], COLOR_TRAIN)
    #axs[2, 1].plot(lists_validation[INDEX_KILLS], COLOR_VALIDATE)
    axs[2, 1].set(title="mean loss", xlabel="epoch", ylabel="mean loss")
    axs[2, 1].plot(lists_training[INDEX_LOSS], COLOR_TRAIN)
    axs[2, 1].plot(lists_validation[INDEX_LOSS], COLOR_VALIDATE)

    #plot training col 2 center right
    axs[0, 2].set(title="training: mean actions", xlabel="epoch", ylabel="mean count")
    axs[0, 2].plot(lists_training[INDEX_BOMB], COLOR_BOMB, label="bomb")
    axs[0, 2].plot(lists_training[INDEX_WAIT], COLOR_WAIT, label="wait")
    axs[0, 2].plot(lists_training[INDEX_MOVEMENT], COLOR_MOVEMENT, label="move")
    axs[0, 2].legend(loc="best")

    #axs[1, 2].set(title="training: mean invalid actions", xlabel="epoch", ylabel="mean count")
    #axs[1, 2].plot(lists_training[INDEX_INVALID_BOMB], COLOR_BOMB, label="bomb")
    #axs[1, 2].plot(lists_training[INDEX_INVALID_WAIT], COLOR_WAIT, label="wait")
    #axs[1, 2].plot(lists_training[INDEX_INVALID_MOVEMENT], COLOR_MOVEMENT, label="move")
    #axs[1, 2].legend(loc="best")

    axs[1, 2].set(title="training: mean invalid actions %", xlabel="epoch", ylabel="mean %")
    axs[1, 2].plot(list_training_invalid_bomb_percentage, COLOR_BOMB, label="bomb")
    axs[1, 2].plot(list_training_invalid_wait_percentage, COLOR_WAIT, label="wait")
    axs[1, 2].plot(list_training_invalid_movement_percentage, COLOR_MOVEMENT, label="move")
    axs[1, 2].legend(loc="best")

    axs[2, 2].set(title="bad bomb %", xlabel="epoch", ylabel="bad bomb %")
    axs[2, 2].plot(list_training_bad_bomb_percentage, COLOR_TRAIN)
    axs[2, 2].plot(list_validation_bad_bomb_percentage, COLOR_VALIDATE)

    #plot validation col 3 right
    axs[0, 3].set(title="validation: mean actions", xlabel="epoch", ylabel="mean count")
    axs[0, 3].plot(lists_validation[INDEX_BOMB], COLOR_BOMB, label="bomb")
    axs[0, 3].plot(lists_validation[INDEX_WAIT], COLOR_WAIT, label="wait")
    axs[0, 3].plot(lists_validation[INDEX_MOVEMENT], COLOR_MOVEMENT, label="move")
    axs[0, 3].legend(loc="best")

    #axs[1, 3].set(title="validation: mean invalid actions", xlabel="epoch", ylabel="mean count")
    #axs[1, 3].plot(lists_validation[INDEX_INVALID_BOMB], COLOR_BOMB, label="bomb")
    #axs[1, 3].plot(lists_validation[INDEX_INVALID_WAIT], COLOR_WAIT, label="wait")
    #axs[1, 3].plot(lists_validation[INDEX_INVALID_MOVEMENT], COLOR_MOVEMENT, label="move")
    #axs[1, 3].legend(loc="best")

    axs[1, 3].set(title="validation: mean invalid actions %", xlabel="epoch", ylabel="mean %")
    axs[1, 3].plot(list_validation_invalid_bomb_percentage, COLOR_BOMB, label="bomb")
    axs[1, 3].plot(list_validation_invalid_wait_percentage, COLOR_WAIT, label="wait")
    axs[1, 3].plot(list_validation_invalid_movement_percentage, COLOR_MOVEMENT, label="move")
    axs[1, 3].legend(loc="best")

    axs[2, 3].set(title="mean loops", xlabel="epoch", ylabel="mean loops")
    axs[2, 3].plot(lists_training[INDEX_LOOP], COLOR_TRAIN)
    axs[2, 3].plot(lists_validation[INDEX_LOOP], COLOR_VALIDATE)

    plt.tight_layout()


if __name__ == "__main__":    
    plt.ion()
    plt.show()
    i = 0
    fig, axs = plt.subplots(3, 4)
    plt.tight_layout()
    while True:
        data_training = load("Training_results.pt")
        data_validation = load("Validation_results.pt")
        update_plot(data_training, data_validation, fig, axs)
        plt.pause(1)
    plt.ioff()
    plt.show()