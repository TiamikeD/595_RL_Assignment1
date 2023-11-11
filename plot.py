import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def plot_distance_error_vs_steps(distance_errors, steps, epsilon):
    return plot2d(steps, distance_errors, f"distance error vs steps e={epsilon}")


def plot_gdop_vs_steps(gdops, steps, epsilon):
    return plot2d(steps, gdops, f"gdops vs steps e={epsilon}")

def plot_reward_vs_steps(rewards, steps, epsilon):
    return plot2d(steps, rewards, f"rewards vs steps e={epsilon}")


def plot2d(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, label=title, color='blue')

    ax.set_xlabel('steps')
    ax.legend()
    plt.show()
    return
