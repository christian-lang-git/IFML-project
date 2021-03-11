import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

PRE_INDEX_PLAYERS = 0
PRE_INDEX_COIN_VALUES = 1
PRE_INDEX_DANGER_REPULSOR = 2
PRE_INDEX_BOMB_TIME_FIELD = 3
PRE_INDEX_SAFETY_TIME_FIELD = 4
PRE_INDEX_FIELD = 5
PRE_INDEX_CRATE_POTENTIAL_SCALED = 6
PRE_INDEX_CRATE_VALUE = 7
PRE_INDEX_VISITED = 8
PRE_INDEX_VISITED_PENALTY = 9
PRE_INDEX_SONAR = 10

rows = 3
cols = 4

def load(file):
    with open(file, "rb") as file:
        data = pickle.load(file)
        return data

def update_plot(data, fig):
    fig.clf()
    
    ax = fig.add_subplot(rows,cols,1)
    ax.set(title="players")
    img = plt.imshow(data[PRE_INDEX_PLAYERS].T, interpolation = 'nearest')
    plt.clim(vmin=-1, vmax=1)
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(rows,cols,2)
    ax.set(title="sonar")
    img = plt.imshow(data[PRE_INDEX_SONAR].T, interpolation = 'nearest')
    plt.clim(vmin=-1, vmax=1)
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(rows,cols,3)
    ax.set(title="coin value")
    img = plt.imshow(data[PRE_INDEX_COIN_VALUES].T, interpolation = 'nearest')
    plt.clim(vmin=-0.1, vmax=1)
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(rows,cols,4)
    ax.set(title="visited penalty")
    img = plt.imshow(data[PRE_INDEX_VISITED_PENALTY].T, interpolation = 'nearest')
    plt.clim(vmin=-1, vmax=1)
    fig.colorbar(img, ax=ax)

    #row 2
    ax = fig.add_subplot(rows,cols,5)
    ax.set(title="field")
    img = plt.imshow(data[PRE_INDEX_FIELD].T, interpolation = 'nearest')
    plt.clim(vmin=-0.1, vmax=1)
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(rows,cols,6)
    ax.set(title="crate value")
    img = plt.imshow(data[PRE_INDEX_CRATE_VALUE].T, interpolation = 'nearest')
    plt.clim(vmin=-0.1, vmax=1)
    fig.colorbar(img, ax=ax) 

    ax = fig.add_subplot(rows,cols,7)
    ax.set(title="crate potential")
    img = plt.imshow(data[PRE_INDEX_CRATE_POTENTIAL_SCALED].T, interpolation = 'nearest')
    plt.clim(vmin=-0.1, vmax=1)
    fig.colorbar(img, ax=ax) 

    ax = fig.add_subplot(rows,cols,8)
    ax.set(title="visited")
    img = plt.imshow(data[PRE_INDEX_VISITED].T, interpolation = 'nearest')
    plt.clim(vmin=-0.1, vmax=1)
    fig.colorbar(img, ax=ax)

    #row 3
    ax = fig.add_subplot(rows,cols,9)
    ax.set(title="danger repulsor")
    img = plt.imshow(data[PRE_INDEX_DANGER_REPULSOR].T, interpolation = 'nearest')
    plt.clim(vmin=0, vmax=1)
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(rows,cols,10)
    ax.set(title="bomb time")
    img = plt.imshow(data[PRE_INDEX_BOMB_TIME_FIELD].T, interpolation = 'nearest')
    plt.clim(vmin=-1, vmax=4)
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(rows,cols,11)
    ax.set(title="safety time")
    img = plt.imshow(data[PRE_INDEX_SAFETY_TIME_FIELD].T, interpolation = 'nearest')
    plt.clim(vmin=-1, vmax=4)
    fig.colorbar(img, ax=ax)

    #axs[1, 0].set(title="coin value")
    #axs[1, 0].imshow(data[INDEX_COIN_VALUES], interpolation = 'nearest')

    #axs[2, 0].set(title="danger value")
    #axs[2, 0].imshow(data[INDEX_DANGER_VALUES], interpolation = 'nearest')

    #fig = plt.figure(figsize = (10, 10))
    #plt.imshow(array2D, interpolation = 'nearest')
    #plt.gray()
    #plt.axis('off')
    #fig.tight_layout()
    #plt.show()

    plt.tight_layout()

if __name__ == "__main__":    
    print("test")
    plt.ion()
    plt.show()
    i = 0
    #fig, axs = plt.subplots(3, 4)
    fig = plt.figure()
    #for ax in fig.axes:
    #    data = load("preprocessing_results.pt")
    #    img = ax.imshow(data[INDEX_PLAYERS], interpolation = 'nearest')
    #    fig.colorbar(img, ax=ax)

    plt.tight_layout()
    while True:
        data = load("preprocessing_results.pt")
        update_plot(data, fig)
        plt.pause(1)
    plt.ioff()
    plt.show()