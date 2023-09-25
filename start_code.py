# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sympy
import bandit as bd

np.random.seed(19680801)

POSSIBLE_ACTIONS = ["012","013","014","023","024","034","123","134","234","124"]

def choose_anchors(anchors, epsilon): # TIAMIKE
    # explore = np.random.binomial(1, epsilon) # Decide to explore or not
    # if explore == 1: # Choose 3 random anchors (Explore)
    recs = np.random.choice(anchors, size=num_anchors_to_choose, replace=False)
    # else: # Choose most promising anchors (Exploit)
        # Rank anchors by their reward
        # Choose the three highest ranked anchors
        # recs = [1,2,3]
    return recs

def calculate_q_value(reward, prev_q, action_count=0):
    # Qn = (R1+R2+...+Rn) / (n-1)
    # When recalculating, use the existing Q values instead of recalculating:
    # Q(n+1) = (1/n)(Rn - Qn)   (pg. 31 of textbook)
    # Q = (1 / action_count) * reward * 
    return [0]

### Step 1: Initialize the problem parameters.

num_anchor_nodes = 5
total_steps = 100000
num_anchors_to_choose = 3

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
position_initial_estimate = centroid

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function to calculate GDOP (Geometric Dilution of Precision)
def calculate_gdop(jacobian):
    G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
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


    # Initialize action counts for each epsilon
    action_count = 0

    # Initialize Q-values for each epsilon
    # Tiamike: Also initializing R-values. These are single digit numbers 
    # because each action has only one reward associated
    prev_q = 0
    Q_of_a = 0
    R_of_a = 0

    # Main loop for the epsilon-greedy bandit algorithm
    for i in range(total_steps):
        # print( "Iteration: " + str(i) )

        # Select three anchor nodes (action A)
        # Exploration: Choose random actions
        # Exploitation: Choose actions with highest Q-values
        # Right now, choose_anchors() selects the index of the anchor.
        chosen_anchors = choose_anchors(anchor_labels, epsilon)
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
        fb = - 1 / (FA**(0.5)) * (selected_positions[1][0] - position_estimate[1])
        fc = - 1 / (FA**(0.5)) * (selected_positions[2][0] - position_estimate[2])

        # ga, gb, gc are the second row of the jacobian matrix
        ga = - 1 / (GA**(0.5)) * (selected_positions[0][1] - position_estimate[0])
        gb = - 1 / (GA**(0.5)) * (selected_positions[1][1] - position_estimate[1])
        gc = - 1 / (GA**(0.5)) * (selected_positions[2][1] - position_estimate[1])

        # ga, gb, gc are the third row of the jacobian matrix
        ha = - 1 / (HA**(0.5)) * (selected_positions[0][2] - position_estimate[0])
        hb = - 1 / (HA**(0.5)) * (selected_positions[1][2] - position_estimate[0])
        hc = - 1 / (HA**(0.5)) * (selected_positions[2][2] - position_estimate[0])

        jacobian_matrix = np.array([[fa, fb, fc],
                                    [ga, gb, gc],
                                    [ha, hb, hc]])
        print("Jacobian matrix: " + str(jacobian_matrix))

        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'
        gdop = calculate_gdop(jacobian_matrix)
        print( "GDOP: " + str(gdop) )
        # Determine the 'reward' R(A) using the 'gdop' value
        R_of_a = calculate_reward(gdop)
        print( "Reward: " + str(R_of_a) )

        # Update action counts N(A)
        action_count += 1

        # Update Q-values Q(A)
        Q_of_a = calculate_q_value(R_of_a, action_count)
        prev_q = Q_of_a

        # Update position estimate
        # position_estimate = 

        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'


    # Store GDOP values, rewards, Euclidean distance errors for each epsilon




### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon


# Plot Reward vs. Steps for each step and each epsilon


# Plot Distance Error vs. Steps for each step and each epsilon
iterator_points = np.array([0,0,0])
for i in range(10):
    iterator_points = np.vstack([iterator_points, np.array([i,i,i])])

bd.plot3d(anchor_positions, target_position, position_initial_estimate, iterator_points)