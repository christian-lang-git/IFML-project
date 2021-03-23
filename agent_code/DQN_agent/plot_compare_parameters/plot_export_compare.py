import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
plot_active = False

LOAD_PREFIX_LIST = ["lr001", "lr0001", "lr00001"]
LABELS = ["lr=1e-03", "lr=1e-04", "lr=1e-05"]
COLOR_LIST = ["red", "forestgreen", "cornflowerblue"]

PREFIX = "compare_lr"
LEGEND_LOC = "best"
VALIDATION_LOSS = False

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

COLOR_TRAIN = "black"
COLOR_VALIDATE = "forestgreen"

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
    return [make_percentage_single(entry, index_a, index_b) for entry in lists]

def make_percentage_single(lists, index_a, index_b):
    a = lists[index_a]
    b = lists[index_b]
    a = np.array(a)
    b = np.array(b)
    c = np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)
    return c * 100

def save_plots(data_training, data_validation):
    #transform data into list of lists
    lists_training = [(list(map(list, zip(*entry)))) for entry in data_training]
    lists_validation = [(list(map(list, zip(*entry)))) for entry in data_validation]

    list_training_invalid_movement_percentage = make_percentage(lists_training, INDEX_INVALID_MOVEMENT, INDEX_MOVEMENT)
    list_training_invalid_wait_percentage = make_percentage(lists_training, INDEX_INVALID_WAIT, INDEX_WAIT)
    list_training_invalid_bomb_percentage = make_percentage(lists_training, INDEX_INVALID_BOMB, INDEX_BOMB)

    list_validation_invalid_movement_percentage = make_percentage(lists_validation, INDEX_INVALID_MOVEMENT, INDEX_MOVEMENT)
    list_validation_invalid_wait_percentage = make_percentage(lists_validation, INDEX_INVALID_WAIT, INDEX_WAIT)
    list_validation_invalid_bomb_percentage = make_percentage(lists_validation, INDEX_INVALID_BOMB, INDEX_BOMB)
    
    list_training_bad_bomb_percentage = make_percentage(lists_training, INDEX_BAD_BOMB, INDEX_BOMB)
    list_validation_bad_bomb_percentage = make_percentage(lists_validation, INDEX_BAD_BOMB, INDEX_BOMB_DROPPED)


    #general
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="training: mean reward", xlabel="epoch", ylabel="mean reward")
    for i, label in enumerate(LABELS):
        ax.plot(lists_training[i][INDEX_REWARD], COLOR_LIST[i], label=label)
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_training_reward.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="validation: mean reward", xlabel="epoch", ylabel="mean reward")
    for i, label in enumerate(LABELS):
        ax.plot(lists_validation[i][INDEX_REWARD], COLOR_LIST[i], label=label)
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_validation_reward.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="training: mean score", xlabel="epoch", ylabel="mean score")
    for i, label in enumerate(LABELS):
        ax.plot(lists_training[i][INDEX_SCORE], COLOR_LIST[i], label=label)
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_training_score.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="validation: mean score", xlabel="epoch", ylabel="mean score")
    for i, label in enumerate(LABELS):
        ax.plot(lists_validation[i][INDEX_SCORE], COLOR_LIST[i], label=label)
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_validation_score.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="validation: invalid movement %", xlabel="epoch", ylabel="mean %")
    for i, label in enumerate(LABELS):
        ax.plot(list_validation_invalid_movement_percentage[i], COLOR_LIST[i], label=label)
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_validation_invalid_movement.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="validation: invalid bomb %", xlabel="epoch", ylabel="mean %")
    for i, label in enumerate(LABELS):
        ax.plot(list_validation_invalid_bomb_percentage[i], COLOR_LIST[i], label=label)
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_validation_invalid_bomb.pdf", format="pdf")
    plt.close()

    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="mean score", xlabel="epoch", ylabel="mean score")
    ax.plot(lists_training[INDEX_SCORE], COLOR_TRAIN, label="train")
    ax.plot(lists_validation[INDEX_SCORE], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_score.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="last epsilon", xlabel="epoch", ylabel="last epsilon")
    ax.plot(lists_training[INDEX_EPSILON], COLOR_TRAIN, label="train")
    ax.plot(lists_validation[INDEX_EPSILON], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_epsilon.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="mean crates", xlabel="epoch", ylabel="mean crates")
    ax.plot(lists_training[INDEX_CRATES], COLOR_TRAIN, label="train")
    ax.plot(lists_validation[INDEX_CRATES], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_crates.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="mean coins", xlabel="epoch", ylabel="mean coins")
    ax.plot(lists_training[INDEX_COINS], COLOR_TRAIN, label="train")
    ax.plot(lists_validation[INDEX_COINS], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_coins.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="mean kills", xlabel="epoch", ylabel="mean kills")
    ax.plot(lists_training[INDEX_KILLS], COLOR_TRAIN, label="train")
    ax.plot(lists_validation[INDEX_KILLS], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_kills.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="mean loss", xlabel="epoch", ylabel="mean loss")
    ax.plot(lists_training[INDEX_LOSS], COLOR_TRAIN, label="train")
    if VALIDATION_LOSS:
        ax.plot(lists_validation[INDEX_LOSS], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_loss.pdf", format="pdf")
    plt.close()

    #actions
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="training mean actions", xlabel="epoch", ylabel="mean count")
    ax.plot(lists_training[INDEX_BOMB], COLOR_BOMB, label="bomb")
    ax.plot(lists_training[INDEX_WAIT], COLOR_WAIT, label="wait")
    ax.plot(lists_training[INDEX_MOVEMENT], COLOR_MOVEMENT, label="move")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_training_actions.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="training mean invalid actions %", xlabel="epoch", ylabel="mean %")
    ax.plot(list_training_invalid_bomb_percentage, COLOR_BOMB, label="bomb")
    #ax.plot(list_training_invalid_wait_percentage, COLOR_WAIT, label="wait")
    ax.plot(list_training_invalid_movement_percentage, COLOR_MOVEMENT, label="move")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_training_invalid_actions.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="validation mean actions", xlabel="epoch", ylabel="mean count")
    ax.plot(lists_validation[INDEX_BOMB], COLOR_BOMB, label="bomb")
    ax.plot(lists_validation[INDEX_WAIT], COLOR_WAIT, label="wait")
    ax.plot(lists_validation[INDEX_MOVEMENT], COLOR_MOVEMENT, label="move")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_validation_actions.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="validation mean invalid actions %", xlabel="epoch", ylabel="mean %")
    ax.plot(list_validation_invalid_bomb_percentage, COLOR_BOMB, label="bomb")
    #ax.plot(list_validation_invalid_wait_percentage, COLOR_WAIT, label="wait")
    ax.plot(list_validation_invalid_movement_percentage, COLOR_MOVEMENT, label="move")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_validation_invalid_actions.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="bad bomb %", xlabel="epoch", ylabel="bad bomb %")
    ax.plot(list_training_bad_bomb_percentage, COLOR_TRAIN, label="train")
    ax.plot(list_validation_bad_bomb_percentage, COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_bad_bomb.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="mean loops", xlabel="epoch", ylabel="mean loops")
    ax.plot(lists_training[INDEX_LOOP], COLOR_TRAIN, label="train")
    ax.plot(lists_validation[INDEX_LOOP], COLOR_VALIDATE, label="validate")
    ax.legend(loc=LEGEND_LOC)
    plt.savefig(fname=PREFIX+"_plot_loops.pdf", format="pdf")
    plt.close()
    """

if __name__ == "__main__":    
    data_training = [load(prefix+"_Training_results.pt") for prefix in LOAD_PREFIX_LIST]
    data_validation = [load(prefix+"_Validation_results.pt") for prefix in LOAD_PREFIX_LIST]
    save_plots(data_training, data_validation)
       
