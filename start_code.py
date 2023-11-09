# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import bandit as bd
import plot as bdplot
import math

np.random.seed(19680801)


num_anchor_nodes = 5
total_steps = 100000
num_anchors_to_choose = 3

POSSIBLE_ACTIONS = [np.array([0,1,2]),np.array([0,1,3]),np.array([0,1,4]),np.array([0,2,3]),np.array([0,2,4]),np.array([0,3,4]),np.array([1,2,3]),np.array([1,3,4]),np.array([2,3,4]),np.array([1,2,4])]

def get_action_index(recs): # KEITH
    for action_index, action in enumerate(POSSIBLE_ACTIONS):
        #print("Action_index:\t" + str(action_index))
        #print("Action:\t" + str(action))
        #print("recs:\t"+str(recs))
        if((recs == action).all()):
            return action_index
    return -1

def get_residual_row(selected_positions, current_position):
    #calculate the three elements under the square root
    # print(f"anchor_position[0]: {selected_positions[0]}")
    #print(f"pseudorange[0]: {pseudorange[0]}")
    # print(f"anchor_position[1]: {selected_positions[1]}")
    #print(f"pseudorange[1]: {pseudorange[1]}")
    # print(f"anchor_position[2]: {selected_positions[2]}")
    #print(f"pseudorange[2]: {pseudorange[2]}")

    dx = (selected_positions[0] - current_position[0]) ** 2
    dy = (selected_positions[1] - current_position[1]) ** 2
    dz = (selected_positions[2] - current_position[2]) ** 2
    # print(f"dx: {dx}")
    # print(f"dy: {dy}")
    # print(f"dz: {dz}")
    # print("distance to anchor: " + str((dx + dy + dz)**0.5))

    #take the square root of the sum of the three elements
    return (dx + dy + dz)**0.5

def get_residual_matrix(selected_positions, current_position, pseudoranges):
    residual_row_f = get_residual_row(selected_positions[0], current_position) - pseudoranges[0]
    residual_row_g = get_residual_row(selected_positions[1], current_position) - pseudoranges[1]
    residual_row_h = get_residual_row(selected_positions[2], current_position) - pseudoranges[2]
    residual_matrix = np.array( [[residual_row_f], [residual_row_g], [residual_row_h]] )
    # print("Residual: " + str(residual_matrix))
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
        # return np.sort(np.random.choice(anchors, size=num_anchors_to_choose, replace=False))
        return POSSIBLE_ACTIONS[np.random.choice(anchor_labels)]

def choose_anchors_for_exploiting(Q_values):
    index_of_highest = get_index_of_highest_reward_action(Q_values)
    # print("index of highest:" + str(index_of_highest))
    return POSSIBLE_ACTIONS[index_of_highest]


def calculate_q_value(reward, prev_q, action_count):
    # Qn = (R1+R2+...+Rn) / (n-1)
    # When recalculating, use the existing Q values instead of recalculating:
    # Q(n+1) = (1/n)(Rn - Qn)   (pg. 31 of textbook)
    Q = (1 / (action_count+1)) * (reward - prev_q)
    return Q

### Step 1: Initialize the problem parameters.


# Initialize anchor node positions and target position
anchor_positions = np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)
anchor_labels = [0,1,2,3,4]

# Eirini says we're not supposed to use the target_position. We only use the pseudoranges.
target_position = [10, 35, 0.1]


# Define two epsilon values
# epsilons = [0.01, 0.3]
epsilons = [1.0]

# Calculate the centroid of anchor node positions
centroid = np.mean(anchor_positions, axis=0)
position_initial_estimate = centroid # 10, 36, 6

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function to calculate GDOP (Geometric Dilution of Precision)
def calculate_gdop(jacobian):
    G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
    #print("GDOP: " + str(gdop))
    return gdop

# Function to calculate reward based on GDOP
def calculate_reward(gdop):
    return np.sqrt(10/3) / gdop if gdop > 0 else 0



### Step 2: Implement the Bandit Algorithm.

# Loop through the epsilon values
for epsilon in epsilons:
    selected_positions = []
    distance_errors = []
    first_psuedorange = []
    # Initializing the 'position_stimate' to 'position_initial_estimate'
    # p(hat) ^ (ite)=0
    position_estimate = position_initial_estimate.copy()
    current_position = position_initial_estimate.copy()
    all_positions = [position_estimate]

    # Initialize Q-values for each epsilon
    # Tiamike: Also initializing R-values. These are single digit numbers
    # because each action has only one reward associated
    prev_q = [0,0,0,0,0,0,0,0,0,0]
    Q_of_a = [0,0,0,0,0,0,0,0,0,0]
    R_of_a = 0

    # Main loop for the epsilon-greedy bandit algorithm
    for i in range(total_steps):
        #print( "Iteration: " + str(i) )
        selected_positions = []
        distance_errors.append( euclidean_distance(position_estimate, target_position) )
        # Select three anchor nodes (action A)
        # Exploration: Choose random actions
        # Exploitation: Choose actions with highest Q-values

        explore = check_if_explore_or_exploit(epsilon)

        if(0 == i):
            explore = True

        exploit = not explore

        if(explore):
            print("exploring...")
            chosen_anchors = choose_anchors_for_exploring(anchor_labels)
        if(exploit):
            # print("exploiting...")
            chosen_anchors = choose_anchors_for_exploiting(Q_of_a)

        for index in chosen_anchors:
            selected_positions.append(anchor_positions[index])

        if (0 == i):
            first_psuedorange = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(num_anchors_to_choose)]

        # print("selected_positions: " + str(selected_positions))
        # print("Position_estimate:" + str(position_estimate))

        # selected_positions = [i for i in range(10)]

        # Code for determining pseudoranges
        # These pseudoranges are the 3 distances from the 3 anchors
        # i.e., the f, g, and h functions
        # They are the MEASURED distances so they have a small amount of noise.
        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(num_anchors_to_choose)]
        # print( "Pseudoranges: " + str(pseudoranges) )

        # Determine the 'jacobian' matrix based on the selected anchor nodes

        # A = ( Xa - X^t )^2 + ( Ya - Y^t )^2 + (Za - Z^t)^2
        # FA, GA, HA are the constants for function f, g and h

        FA = (selected_positions[0][0] - position_estimate[0])**2 + (selected_positions[0][1] - position_estimate[1])**2 + (selected_positions[0][2] - position_estimate[2])**2
        GA = (selected_positions[1][0] - position_estimate[0])**2 + (selected_positions[1][1] - position_estimate[1])**2 + (selected_positions[1][2] - position_estimate[2])**2
        HA = (selected_positions[2][0] - position_estimate[0])**2 + (selected_positions[2][1] - position_estimate[1])**2 + (selected_positions[2][2] - position_estimate[2])**2

        # fa, fb, fc are the first row of the jacobian matrix
        fa = - (FA**(-0.5)) * (selected_positions[0][0] - position_estimate[0])
        fb = - (FA**(-0.5)) * (selected_positions[0][1] - position_estimate[1])
        fc = - (FA**(-0.5)) * (selected_positions[0][2] - position_estimate[2])

        # ga, gb, gc are the second row of the jacobian matrix
        ga = - (GA**(-0.5)) * (selected_positions[1][0] - position_estimate[0])
        gb = - (GA**(-0.5)) * (selected_positions[1][1] - position_estimate[1])
        gc = - (GA**(-0.5)) * (selected_positions[1][2] - position_estimate[2])

        # ga, gb, gc are the third row of the jacobian matrix
        ha = - (HA**(-0.5)) * (selected_positions[2][0] - position_estimate[0])
        hb = - (HA**(-0.5)) * (selected_positions[2][1] - position_estimate[1])
        hc = - (HA**(-0.5)) * (selected_positions[2][2] - position_estimate[2])

        jacobian_matrix = np.array([[fa, fb, fc],
                                    [ga, gb, gc],
                                    [ha, hb, hc]])
        # print("Jacobian matrix: " + str(jacobian_matrix))

        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'
        gdop = calculate_gdop(jacobian_matrix)
        # print( "GDOP: " + str(gdop) )
        # Determine the 'reward' R(A) using the 'gdop' value
        R_of_a = calculate_reward(gdop)
        # print( "Reward: " + str(R_of_a) )

        current_action_index = get_action_index(chosen_anchors)
        # print("Get action index: " + str(current_action_index))

        # Update Q-values Q(A)
        Q_of_a[current_action_index] = calculate_q_value(R_of_a, prev_q[current_action_index], i)
        prev_q[current_action_index] = Q_of_a[current_action_index]
        # print("prev q: " + str(prev_q))
        # print("Q of a: " + str(Q_of_a))

        # Update position estimate
        # residual = get_residual_matrix(selected_positions, position_estimate, first_psuedorange)
        residual = get_residual_matrix(selected_positions, position_estimate, pseudoranges)

        new_position_deltas = get_new_deltas_to_calculate_new_position(jacobian_matrix, residual)
        # print("new position deltas: " + str(new_position_deltas))
        position_estimate = position_estimate - new_position_deltas
        # print("New position estimate: " + str(position_estimate))
        all_positions.append(position_estimate)
        # print(position_estimate)

        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'


    # Store GDOP values, rewards, Euclidean distance errors for each epsilon




### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon


# Plot Reward vs. Steps for each step and each epsilon


# Plot Distance Error vs. Steps for each step and each epsilon


# bdplot.plot3d(anchor_positions, target_position, position_initial_estimate, all_positions, centroid)

# print(distance_errors)

x_axis = [i for i in range(total_steps)]
bdplot.plot2d(x_axis, distance_errors)
