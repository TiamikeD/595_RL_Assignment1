import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sympy


def plot3d(anchor_points = [], target_point = [], position_initial_estimate = [], iterator_points = [], centroid = []): # KEITH

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
        ax.scatter(iterator_points[0], iterator_points[1], iterator_points[2], c="green")
        green_dot = mlines.Line2D([], [], color='green', marker='.',linestyle='None', markersize=10, label='Iterator')

    if(centroid.any()):
        print(centroid)
        ax.scatter(centroid[0], centroid[1], centroid[2], c="orange")
        orange_dot = mlines.Line2D([], [], color='orange', marker='.',linestyle='None', markersize=10, label='centroid')




    plt.legend(handles=[blue_diamond, red_x, green_dot, black_dot, orange_dot])


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

    return


def get_jacobian_matrix(selected_positions, position_estimate):
        FA = (selected_positions[0][0] - position_estimate[0])**2 + (selected_positions[0][1] - position_estimate[1])**2 + (selected_positions[0][2] - position_estimate[2])**2
        GA = (selected_positions[1][0] - position_estimate[0])**2 + (selected_positions[1][1] - position_estimate[1])**2 + (selected_positions[1][2] - position_estimate[2])**2
        HA = (selected_positions[2][0] - position_estimate[0])**2 + (selected_positions[2][1] - position_estimate[1])**2 + (selected_positions[2][2] - position_estimate[2])**2

        fa = - 1 / (FA**(0.5)) * (selected_positions[0][0] - position_estimate[0])
        fb = - 1 / (FA**(0.5)) * (selected_positions[1][0] - position_estimate[1])
        fc = - 1 / (FA**(0.5)) * (selected_positions[2][0] - position_estimate[2])

        ga = - 1 / (GA**(0.5)) * (selected_positions[0][1] - position_estimate[0])
        gb = - 1 / (GA**(0.5)) * (selected_positions[1][1] - position_estimate[1])
        gc = - 1 / (GA**(0.5)) * (selected_positions[2][1] - position_estimate[1])

        ha = - 1 / (HA**(0.5)) * (selected_positions[0][2] - position_estimate[0])
        hb = - 1 / (HA**(0.5)) * (selected_positions[1][2] - position_estimate[0])
        hc = - 1 / (HA**(0.5)) * (selected_positions[2][2] - position_estimate[0])

        return np.array([[fa, fb, fc],[ga, gb, gc],[ha, hb, hc]])

def get_action_index(recs, POSSIBLE_ACTIONS): # KEITH
    for action_index, action in enumerate(POSSIBLE_ACTIONS):
        if((recs == action).all()):
            return action_index
    return -1

def get_residual_from_pseudoranges(selected_anchor_positions, pseudoranges):
    SOME_CONST_FLOAT: float = 100
    #print(f"pseudoranges\t{pseudoranges}")
    #print()
    #print(f"selected_anchor_positions\t{selected_anchor_positions}")
    #print()
    #for i in selected_anchor_positions:
    #    print(i(0))
    return np.array([[SOME_CONST_FLOAT],[SOME_CONST_FLOAT],[SOME_CONST_FLOAT]])

def get_new_deltas_to_calculate_new_position(jacobian, pseudoranges, selected_anchor_positions):
    #print("*********************************************************************************************")


    #print(f"jacobian\t{jacobian}")
    #print()
    #print(f"position_estimate\t{pseudoranges}")
    #print()
    jacobian_transpose = jacobian.T
    #print(f"jacobian_transpose\t{jacobian_transpose}")
    #print()
    jacobian_dot_jacobian_transpose = np.dot(jacobian.T, jacobian)
    #print(f"jacobian_dot_jacobian_transpose\t{jacobian_dot_jacobian_transpose}")
    #print()
    inverse_matrix = np.linalg.inv(jacobian_dot_jacobian_transpose)
    #print(f"inverse\t{inverse_matrix}")
    #print()
    multiply_me_with_residual = np.dot(inverse_matrix, jacobian_transpose)
    #print(f"multiply_me_with_residual\t{multiply_me_with_residual}")
    #print()
    #residual = np.array( [ [pseudoranges[0]], [pseudoranges[1]], [pseudoranges[2]] ] )
    residual = get_residual_from_pseudoranges(selected_anchor_positions, pseudoranges)
    #print(f"residual\t{residual}")
    #print()
    return_me = np.dot(multiply_me_with_residual, residual)
    #print(f"return_me\t{return_me}")
    #print()
    #print("*********************************************************************************************")
    return np.matrix.flatten(return_me)

def choose_anchors(anchors, anchor_positions, epsilon, num_anchors_to_choose):
    explore = np.random.binomial(1, epsilon) # Decide to explore or not
    #print(explore)
    '''
    if explore == 1: # Choose 3 random anchors (Explore)
    else: # Choose most promising anchors (Exploit)
        Rank anchors by their reward
        Choose the three highest ranked anchors
    recs = np.sort(recs)'''

    return


def choose_random_anchors(anchors, anchor_positions, epsilon, num_anchors_to_choose): # TIAMIKE
    return np.random.choice(anchors, size=num_anchors_to_choose, replace=False)


def calculate_q_value(reward, previous_q_value, action_count):
    # Qn = (R1+R2+...+Rn) / (n-1)
    # When recalculating, use the existing Q values instead of recalculating:
    # Q(n+1) = (1/n)(Rn - Qn)   (pg. 31 of textbook)
    Q = (1 / action_count) * (reward - previous_q_value)
    return Q


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def calculate_gdop(jacobian):
    G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
    return gdop


def calculate_reward(gdop):
    return np.sqrt(10/3) / gdop if gdop > 0 else 0

def red_string_for_print(change_me_to_red)->str:
    return f"\033[31m{change_me_to_red}\033[0m"

def print_exception(message, exception)->None:
    print(red_string_for_print(message))
    print(f"\t{exception}")
    return

