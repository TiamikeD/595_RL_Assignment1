import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sympy


def plot3d(anchor_points = [], target_point = [], position_initial_estimate = [], iterator_points = []): # KEITH

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    blue_diamond, red_x, green_dot, black_dot = 0,0,0,0

    if(anchor_points.any()):
        for point in anchor_points:
            ax.scatter(point[0],point[1], point[2], c="blue", marker="D")
            blue_diamond = mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Anchor Point')


    if(target_point):
        ax.scatter(target_point[0], target_point[1], target_point[2], c="red", marker="X")
        red_x = mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label='Target')

    if(position_initial_estimate.any()):
        ax.scatter(position_initial_estimate[0], position_initial_estimate[1], position_initial_estimate[2], c="black")
        black_dot = mlines.Line2D([], [], color='black', marker='.',linestyle='None', markersize=10, label='Initial estimate')

    if(iterator_points.any()):
        for iterator in iterator_points:
            print(iterator)
            ax.scatter(iterator[0], iterator[1], iterator[2], c="green")
            green_dot = mlines.Line2D([], [], color='green', marker='.',linestyle='None', markersize=10, label='Iterator')




    plt.legend(handles=[blue_diamond, red_x, green_dot, black_dot])


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

    return
