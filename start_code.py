# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sympy
import bandit as bd
import math

np.random.seed(19680801)


num_anchor_nodes = 5
total_steps = 100000
num_anchors_to_choose = 3

POSSIBLE_ACTIONS = [np.array([0,1,2]),np.array([0,1,3]),np.array([0,1,4]),np.array([0,2,3]),np.array([0,2,4]),np.array([0,3,4]),np.array([1,2,3]),np.array([1,3,4]),np.array([2,3,4]),np.array([1,2,4])]

def get_action_index(recs): # KEITH
    for action_index, action in enumerate(POSSIBLE_ACTIONS):
        if((recs == action).all()):
            return action_index
    return -1

def get_residual_row(anchor_position, pseudorange):
    #calculate the three elements under the square root
    print(f"anchor_position[0]: {anchor_position[0]}\n")
    print(f"\npseudorange[0]: {pseudorange[0]}\n")
    print(f"\nanchor_position[1]: {anchor_position[1]}\n")
    print(f"\npseudorange[1]: {pseudorange[1]}\n")
    print(f"\nanchor_position[2]: {anchor_position[2]}\n")
    print(f"\npseudorange[2]: {pseudorange[2]}\n")

    dx = (anchor_position[0] - pseudorange[0]) ** 2 
    dy = (anchor_position[1] - pseudorange[1]) ** 2 
    dz = (anchor_position[2] - pseudorange[2]) ** 2 
    print(f"\ndx: {dx}\n")
    print(f"\ndy: {dy}\n")
    print(f"\ndz: {dz}\n")

    #take the square root of the sum of the three elements
    return math.sqrt(dx + dy + dz)

def get_residual_matrix(anchor_positions, pseudoranges):
    residual_row_f = get_residual_row(anchor_positions[0], pseudoranges) - pseudoranges[0]
    residual_row_g = get_residual_row(anchor_positions[1], pseudoranges) - pseudoranges[1]
    residual_row_h = get_residual_row(anchor_positions[2], pseudoranges) - pseudoranges[2]
    residual_matrix = np.array( [[residual_row_f], [residual_row_g], [residual_row_h]] )
    return residual_matrix

def get_new_deltas_to_calculate_new_position(jacobian, anchor_positions, pseudoranges, residual):
    jacobian_transpose = jacobian.T
    jacobian_dot_jacobian_transpose = np.dot(jacobian.T, jacobian)
    inverse_matrix = np.linalg.inv(jacobian_dot_jacobian_transpose)

    multiply_me_with_residual = np.dot(inverse_matrix, jacobian_transpose)
    # residual = get_residual_matrix(anchor_positions, pseudoranges)
    print("Residual: " + str(residual))

    print(f"Transpose:\t\n{jacobian_transpose}\n\n  Dot Product:\t\n{jacobian_dot_jacobian_transpose}\n\n inverse_matrix\t{inverse_matrix}\n\n  multiply_me_with_residual\n{multiply_me_with_residual}\n")
    return_me = np.dot(multiply_me_with_residual, residual)

    return  np.matrix.flatten(return_me) #np.dot(np.dot(jacobian.T, np.linalg.inv(np.dot(jacobian.T, jacobian))), np.array([pseudorange for pseudorange in pseudoranges])) 

# where we left off: add delta to position function

def choose_anchors(anchors, anchor_positions, epsilon): # TIAMIKE
    # explore = np.random.binomial(1, epsilon) # Decide to explore or not
    # if explore: # Choose 3 random anchors (Explore)
    recs = np.random.choice(anchors, size=num_anchors_to_choose, replace=False)
    # else: # Choose most promising anchors (Exploit)
    #     # Rank anchors by their reward
    #     print("anchors: " + str(anchors))
    #     print("anchor_positions: " + str(anchor_positions))
    #     # Choose the three highest ranked anchors
    #     recs = np.random.choice(anchors, size=num_anchors_to_choose, replace=False)
        
    return recs

def calculate_q_value(reward, prev_q, action_count):
    # Qn = (R1+R2+...+Rn) / (n-1)
    # When recalculating, use the existing Q values instead of recalculating:
    # Q(n+1) = (1/n)(Rn - Qn)   (pg. 31 of textbook)
    Q = (1 / action_count) * (reward - prev_q)
    return Q

### Step 1: Initialize the problem parameters.


# Initialize anchor node positions and target position
anchor_positions = np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)
anchor_labels = [0,1,2,3,4]

# Eirini says we're not supposed to use the target_position. We only use the pseudoranges.
target_position = [10, 35, 0.1]


# Define two epsilon values
epsilons = [0.01, 0.3]

# Calculate the centroid of anchor node positions
centroid = np.mean(anchor_positions, axis=0)

# Set the initial position estimate as the centroid
position_initial_estimate = centroid # 10, 36, 6

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function to calculate GDOP (Geometric Dilution of Precision)
def calculate_gdop(jacobian):
    G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
    print("GDOP: " + str(gdop))
    return gdop

# Function to calculate reward based on GDOP
def calculate_reward(gdop):
    return np.sqrt(10/3) / gdop if gdop > 0 else 0



### Step 2: Implement the Bandit Algorithm.

# Loop through the epsilon values
for epsilon in epsilons:

    # Initializing the 'position_stimate' to 'position_initial_estimate'
    # p(hat) ^ (ite)=0
    position_estimate = position_initial_estimate.copy()
    current_position = position_initial_estimate.copy()
    all_positions = [position_estimate]


    # Initialize action counts for each epsilon
    action_count = 0

    # Initialize Q-values for each epsilon
    # Tiamike: Also initializing R-values. These are single digit numbers 
    # because each action has only one reward associated
    prev_q = [0,0,0,0,0,0,0,0,0,0,0]
    Q_of_a = [0,0,0,0,0,0,0,0,0,0,0]
    R_of_a = 0

    # Main loop for the epsilon-greedy bandit algorithm
    for i in range(total_steps):
        # print( "Iteration: " + str(i) )

        # Select three anchor nodes (action A)
        # Exploration: Choose random actions
        # Exploitation: Choose actions with highest Q-values
        # Right now, choose_anchors() selects the index of the anchor.
        chosen_anchors = choose_anchors(anchor_labels, anchor_positions, epsilon)
        print( "Chosen Anchors: " + str(chosen_anchors) )

        # Tiamike: I'm assuming selected_positions are the positions of the anchors that were chosen.
        selected_positions = []
        for index in chosen_anchors:
            selected_positions.append(anchor_positions[index])
        print("selected_positions: " + str(selected_positions))

        # selected_positions = [i for i in range(10)]

        # Code for determining pseudoranges
        # These pseudoranges are the 3 distances from the 3 anchors
        # i.e., the f, g, and h functions
        # They are the MEASURED distances so they have a small amount of noise.
        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(num_anchors_to_choose)]
        print( "Pseudoranges: " + str(pseudoranges) )
        print( "Initial position estimate: " + str(position_estimate) )

        # Determine the 'jacobian' matrix based on the selected anchor nodes

        # A = ( Xa - X^t )^2 + ( Ya - Y^t )^2 + (Za - Z^t)^2
        # FA, GA, HA are the constants for function f, g and h
        FA = (selected_positions[0][0] - position_estimate[0])**2 + (selected_positions[0][1] - position_estimate[1])**2 + (selected_positions[0][2] - position_estimate[2])**2
        # print("FA: " + str(FA))
        GA = (selected_positions[1][0] - position_estimate[0])**2 + (selected_positions[1][1] - position_estimate[1])**2 + (selected_positions[1][2] - position_estimate[2])**2
        HA = (selected_positions[2][0] - position_estimate[0])**2 + (selected_positions[2][1] - position_estimate[1])**2 + (selected_positions[2][2] - position_estimate[2])**2



        # fa, fb, fc are the first row of the jacobian matrix
        fa = - 1 / (FA**(0.5)) * (selected_positions[0][0] - position_estimate[0])
        # print("fa: " + str(fa))
        fb = - 1 / (FA**(0.5)) * (selected_positions[0][1] - position_estimate[1])
        fc = - 1 / (FA**(0.5)) * (selected_positions[0][2] - position_estimate[2])

        # ga, gb, gc are the second row of the jacobian matrix
        ga = - 1 / (GA**(0.5)) * (selected_positions[1][0] - position_estimate[0])
        gb = - 1 / (GA**(0.5)) * (selected_positions[1][1] - position_estimate[1])
        gc = - 1 / (GA**(0.5)) * (selected_positions[1][2] - position_estimate[1])

        # ga, gb, gc are the third row of the jacobian matrix
        ha = - 1 / (HA**(0.5)) * (selected_positions[2][0] - position_estimate[0])
        hb = - 1 / (HA**(0.5)) * (selected_positions[2][1] - position_estimate[0])
        hc = - 1 / (HA**(0.5)) * (selected_positions[2][2] - position_estimate[0])

        jacobian_matrix = np.array([[fa, fb, fc],
                                    [ga, gb, gc],
                                    [ha, hb, hc]])
        print("Jacobian matrix: " + str(jacobian_matrix))

        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'
        gdop = calculate_gdop(jacobian_matrix)
        # print( "GDOP: " + str(gdop) )
        # Determine the 'reward' R(A) using the 'gdop' value
        R_of_a = calculate_reward(gdop)
        # print( "Reward: " + str(R_of_a) )

        # Update action counts N(A)
        action_count += 1

        # Update Q-values Q(A)
        current_action_index = get_action_index(chosen_anchors)
        # print("Get action index: " + str(current_action_index))

        Q_of_a[current_action_index] = calculate_q_value(R_of_a, prev_q[current_action_index], action_count)
        prev_q[current_action_index] = Q_of_a[current_action_index]
        print("prev q: " + str(prev_q))
        print("Q of a: " + str(Q_of_a))

        # Update position estimate
        residual = get_residual_matrix(anchor_positions, pseudoranges)

        new_position_deltas = get_new_deltas_to_calculate_new_position(jacobian_matrix, selected_positions, pseudoranges, residual)
        print("new position deltas: " + str(new_position_deltas))
        position_estimate = position_estimate + new_position_deltas
        print("New position estimate: " + str(position_estimate))
        all_positions.append(position_estimate)
        print(f"\n\n\n DONE WITH ONE STEP {i} \n\n\n")


        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'


    # Store GDOP values, rewards, Euclidean distance errors for each epsilon




### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon


# Plot Reward vs. Steps for each step and each epsilon


# Plot Distance Error vs. Steps for each step and each epsilon
iterator_points = np.array([0,0,0])
for i in range(10):
    iterator_points = np.vstack([iterator_points, np.array([i,i,i])])

#bd.plot3d(anchor_positions, target_position, position_initial_estimate, iterator_points)