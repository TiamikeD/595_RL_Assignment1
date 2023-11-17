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


def plot_episodic_results(steps, distance_errors, gdops, rewards, epsilon):
    x = steps
    y1 = distance_errors
    y2 = gdops
    y3 = rewards

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(x, y1, label=f'epsilon={epsilon}', color='blue')
    ax1.set_title(f'Distance Errors')
    ax1.legend()

    ax2.plot(x, y2, label=f'epsilon={epsilon}', color='green')
    ax2.set_title('Gdop')
    ax2.legend()
    ax3.plot(x, y3, label=f'epsilon={epsilon}', color='red')
    ax3.set_title('Rewards')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f"results_{epsilon}")
    return 


def plot_results(results):
    x = results[0][0]

    y1 = results[0][1]
    z1 = results[1][1]
    
    y2 = results[0][2]
    z2 = results[1][2]

    y3 = results[0][3]
    z3 = results[1][3]

    epsilon = results[0][4]
    epsilon1 = results[1][4]
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(x, y1, label=f'epsilon={epsilon}', color='blue')
    ax1.plot(x, z1, label=f'epsilon={epsilon1}', color='red')
    ax1.set_title(f'Distance Errors')
    ax1.legend()

    ax2.plot(x, y2, label=f'epsilon={epsilon}', color='green')
    ax2.plot(x, z2, label=f"epsilon={epsilon1}", color="blue")
    ax2.set_title('Gdop')
    ax2.legend()

    ax3.plot(x, y3, label=f'epsilon={epsilon}', color='red')
    ax3.plot(x, z3, label=f"epsilon={epsilon1}", color='green')
    ax3.set_title('Rewards')
    ax3.legend()

    plt.tight_layout()
    plt.savefig("results")



