import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def get_action_index(recs): # KEITH
    for action_index, action in enumerate(POSSIBLE_ACTIONS):
        if((recs == action).all()):
            return action_index
    return -1

def get_residual_row(selected_positions, current_position):
    dx = (selected_positions[0] - current_position[0]) ** 2
    dy = (selected_positions[1] - current_position[1]) ** 2
    dz = (selected_positions[2] - current_position[2]) ** 2

    return (dx + dy + dz)**0.5

def get_residual_matrix(selected_positions, current_position, pseudoranges):
    residual_row_f = get_residual_row(selected_positions[0], current_position) - pseudoranges[0]
    residual_row_g = get_residual_row(selected_positions[1], current_position) - pseudoranges[1]
    residual_row_h = get_residual_row(selected_positions[2], current_position) - pseudoranges[2]
    residual_matrix = np.array( [[residual_row_f], [residual_row_g], [residual_row_h]] )
    return residual_matrix

def get_new_deltas_to_calculate_new_position(jacobian, residual):
    jacobian_dot_jacobian_transpose = np.dot(jacobian.T, jacobian)
    inverse_matrix = np.linalg.inv(jacobian_dot_jacobian_transpose)
    multiply_me_with_residual = np.dot(inverse_matrix, jacobian.T)
    return_me = np.dot(multiply_me_with_residual, residual)

    return  np.matrix.flatten(return_me)

def get_index_of_highest_reward_action(Q_values):
    max_value = max(Q_values)
    return Q_values.index(max_value)

def check_if_explore_or_exploit(epsilon):
    return np.random.binomial(1, epsilon)


def choose_anchors_for_exploring(anchors):
        return POSSIBLE_ACTIONS[np.random.choice(anchor_labels)]

def choose_anchors_for_exploiting(Q_values):
    index_of_highest = get_index_of_highest_reward_action(Q_values)
    return POSSIBLE_ACTIONS[index_of_highest]


def calculate_q_value(reward, prev_q, action_count):
    Q = (1 / (action_count+1)) * (reward - prev_q)
    return Q

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def calculate_gdop(jacobian):
    G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
    return gdop

def calculate_reward(gdop):
    return np.sqrt(10/3) / gdop if gdop > 0 else 0

def calculate_jacobian(selected_positions, position_estimate):
    FA = (selected_positions[0][0] - position_estimate[0])**2 + (selected_positions[0][1] - position_estimate[1])**2 + (selected_positions[0][2] - position_estimate[2])**2
    GA = (selected_positions[1][0] - position_estimate[0])**2 + (selected_positions[1][1] - position_estimate[1])**2 + (selected_positions[1][2] - position_estimate[2])**2
    HA = (selected_positions[2][0] - position_estimate[0])**2 + (selected_positions[2][1] - position_estimate[1])**2 + (selected_positions[2][2] - position_estimate[2])**2

    fa = - (FA**(-0.5)) * (selected_positions[0][0] - position_estimate[0])
    fb = - (FA**(-0.5)) * (selected_positions[0][1] - position_estimate[1])
    fc = - (FA**(-0.5)) * (selected_positions[0][2] - position_estimate[2])

    ga = - (GA**(-0.5)) * (selected_positions[1][0] - position_estimate[0])
    gb = - (GA**(-0.5)) * (selected_positions[1][1] - position_estimate[1])
    gc = - (GA**(-0.5)) * (selected_positions[1][2] - position_estimate[2])

    ha = - (HA**(-0.5)) * (selected_positions[2][0] - position_estimate[0])
    hb = - (HA**(-0.5)) * (selected_positions[2][1] - position_estimate[1])
    hc = - (HA**(-0.5)) * (selected_positions[2][2] - position_estimate[2])

    return np.array([[fa, fb, fc],[ga, gb, gc],[ha, hb, hc]])


def get_possible_actions():
    return [np.array([0,1,2]),np.array([0,1,3]),np.array([0,1,4]),np.array([0,2,3]),np.array([0,2,4]),np.array([0,3,4]),np.array([1,2,3]),np.array([1,3,4]),np.array([2,3,4]),np.array([1,2,4])]


def get_anchor_positions():
    return np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)


POSSIBLE_ACTIONS = get_possible_actions()
anchor_labels = [0,1,2,3,4]
