import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import bandit as bd
import plot as bdplot
import math

np.random.seed(19680801)


num_anchor_nodes = 5
total_steps = 10000
num_anchors_to_choose = 3

POSSIBLE_ACTIONS = bd.POSSIBLE_ACTIONS
anchor_positions = bd.get_anchor_positions()
anchor_labels = bd.anchor_labels

target_position = [10, 35, 0.1]
epsilons = [0.01,0.3]

centroid = np.mean(anchor_positions, axis=0)
position_initial_estimate = centroid


for epsilon in epsilons:
    selected_positions = []
    distance_errors = []
    gdops = []
    rewards = []
    first_psuedorange = []

    position_estimate = position_initial_estimate.copy()
    current_position = position_initial_estimate.copy()
    all_positions = [position_estimate]

    prev_q = [0,0,0,0,0,0,0,0,0,0]
    Q_of_a = [0,0,0,0,0,0,0,0,0,0]
    R_of_a = 0

    for i in range(total_steps):
        selected_positions = []
        distance_errors.append(bd.euclidean_distance(position_estimate, target_position))

        explore = bd.check_if_explore_or_exploit(epsilon)

        if(0 == i):
            explore = True

        exploit = not explore

        if(explore):
            chosen_anchors = bd.choose_anchors_for_exploring(anchor_labels)
        if(exploit):
            chosen_anchors = bd.choose_anchors_for_exploiting(Q_of_a)


        for index in chosen_anchors:
            selected_positions.append(anchor_positions[index])

        if (0 == i):
            first_psuedorange = [bd.euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(num_anchors_to_choose)]

        pseudoranges = [bd.euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(num_anchors_to_choose)]

        jacobian_matrix = bd.calculate_jacobian(selected_positions, position_estimate)

        gdop = bd.calculate_gdop(jacobian_matrix)
        gdops.append(gdop)

        R_of_a = bd.calculate_reward(gdop)
        rewards.append(R_of_a)

        current_action_index = bd.get_action_index(chosen_anchors)

        Q_of_a[current_action_index] = bd.calculate_q_value(R_of_a, prev_q[current_action_index], i)
        prev_q[current_action_index] = Q_of_a[current_action_index]
        residual = bd.get_residual_matrix(selected_positions, position_estimate, pseudoranges)

        new_position_deltas = bd.get_new_deltas_to_calculate_new_position(jacobian_matrix, residual)
        position_estimate = position_estimate - new_position_deltas
        all_positions.append(position_estimate)






    steps = [i for i in range(total_steps)]

    #bdplot.plot_gdop_vs_steps(gdops, steps, epsilon)
    #bdplot.plot_reward_vs_steps(rewards, steps, epsilon)
    #bdplot.plot_distance_error_vs_steps(distance_errors, steps, epsilon)

    bdplot.plot_results(steps, distance_errors, gdops, rewards, epsilon)
