# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sympy
import bandit as bd

np.random.seed(19680801)


POSSIBLE_ACTIONS = [np.array([0,1,2]),np.array([0,1,3]),np.array([0,1,4]),np.array([0,2,3]),np.array([0,2,4]),np.array([0,3,4]),np.array([1,2,3]),np.array([1,3,4]),np.array([2,3,4]),np.array([1,2,4])]
NUMBER_OF_ANCHOR_NODES = 5
TOTAL_NUMBER_OF_STEPS_FOR_BANDIT_ALGORITHM = 100000
NUMBER_OF_ANCHORS_TO_SELECT_FOR_BANDIT_ALGORITHM = 3
ANCHOR_POSITIONS = np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)
ANCHOR_LABELS = [0,1,2,3,4]
TARGET_POSITION = [10, 35, 0.1]
EPSILON_VALUES = [0.01, 0.3]
CENTROID = np.mean(ANCHOR_POSITIONS, axis=0)
INITIAL_ESTIMATE_OF_THE_POSITION = CENTROID



for epsilon in EPSILON_VALUES:
    position_estimate = INITIAL_ESTIMATE_OF_THE_POSITION.copy()
    current_position = INITIAL_ESTIMATE_OF_THE_POSITION.copy()
    all_positions = [position_estimate]
    action_count = 0
    previous_q_value = [0,0,0,0,0,0,0,0,0,0,0]
    q_value = [0,0,0,0,0,0,0,0,0,0,0]
    reward_ra = 0


    for i in range(TOTAL_NUMBER_OF_STEPS_FOR_BANDIT_ALGORITHM):
        chosen_anchors = bd.choose_anchors(ANCHOR_LABELS, ANCHOR_POSITIONS, epsilon, NUMBER_OF_ANCHORS_TO_SELECT_FOR_BANDIT_ALGORITHM)
        selected_positions = [ANCHOR_POSITIONS[index] for index in chosen_anchors]
        pseudoranges = [bd.euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(NUMBER_OF_ANCHORS_TO_SELECT_FOR_BANDIT_ALGORITHM)]
        jacobian_matrix = bd.get_jacobian_matrix(selected_positions, position_estimate)
        gdop = bd.calculate_gdop(jacobian_matrix)
        reward_ra = bd.calculate_reward(gdop)
        action_count += 1
        current_action_index = bd.get_action_index(chosen_anchors, POSSIBLE_ACTIONS)
        q_value[current_action_index] = bd.calculate_q_value(reward_ra, previous_q_value[current_action_index], action_count)
        previous_q_value[current_action_index] = q_value[current_action_index]
        new_position_deltas = bd.get_new_deltas_to_calculate_new_position(jacobian_matrix, pseudoranges)
        position_estimate = position_estimate + new_position_deltas
        all_positions.append(position_estimate)









iterator_points = np.array([0,0,0])
for i in range(10):
    iterator_points = np.vstack([iterator_points, np.array([i,i,i])])
