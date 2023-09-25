# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sympy

np.random.seed(19680801)

POSSIBLE_ACTIONS = ["012","013","014","023","024","034","123","134","234","124"]

def plot3d(anchor_points = [], target_point = [], iterator = []): # KEITH
    print(anchor_points)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for point in anchor_points:
        ax.scatter(point[0],point[1], point[2], c="blue", marker="D")

    ax.scatter(target_point[0], target_point[1], target_point[2], c="red", marker="X")

    ax.scatter(iterator[0], iterator[1], iterator[2], c="green")

    blue_diamond = mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Anchor Point')
    red_x = mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label='Target')
    green_dot = mlines.Line2D([], [], color='green', marker='.',linestyle='None', markersize=10, label='Iterator')

    plt.legend(handles=[blue_diamond, red_x, green_dot])


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

    return

def choose_anchors(anchors, epsilon): # TIAMIKE
    # explore = np.random.binomial(1, epsilon) # Decide to explore or not
    # if explore == 1: # Choose 3 random anchors (Explore)
    recs = np.random.choice(anchors, size=num_anchors_to_choose, replace=False)
    # else: # Choose most promising anchors (Exploit)
        # Rank anchors by their reward
        # Choose the three highest ranked anchors
        # recs = [1,2,3]
    return recs

def calculate_q_values(rewards = [], action_count=0):
    # Qn = (R1+R2+...+Rn) / (n-1)
    # When recalculating, use the existing Q values instead of recalculating:
    # Q(n+1) = (1/n)(Rn - Qn)   (pg. 31 of textbook)
    return [0,0,0,0,0]

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
    position_estimate = position_initial_estimate.copy()


    # Initialize action counts for each epsilon
    action_count = 0

    # Initialize Q-values for each epsilon
    # Tiamike: Also initializing R-values. 5 elements for 5 anchors.
    Q_of_a = [0,0,0,0,0]
    R_of_a = [0,0,0,0,0]

    # Main loop for the epsilon-greedy bandit algorithm
    for i in range(total_steps):
        # print( "Iteration: " + str(i) )

        # Select three anchor nodes (action A)
        # Exploration: Choose random actions
        # Exploitation: Choose actions with highest Q-values
        # Right now, choose_anchors() selects the index of the anchor.
        chosen_anchors = choose_anchors(anchor_labels, epsilon)
        print( "Chosen Anchors: " + str(chosen_anchors) )

# plot3d(anchor_positions, target_position, position_initial_estimate)

        # Tiamike: I'm assuming selected_positions are the positions of the anchors that were chosen.
        selected_positions = []
        for index in chosen_anchors:
            selected_positions.append(anchor_positions[index])
        print("selected_positions: " + str(selected_positions))

        # selected_positions = [i for i in range(10)]

        # Code for determining pseudoranges
        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(num_anchors_to_choose)]
        print("Pseudoranges: " + str(pseudoranges))

        # Determine the 'jacobian' matrix based on the selected anchor nodes


        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'


        # Determine the 'reward' R(A) using the 'gdop' value
        # R_of_a = 

        # Update action counts N(A)
        # action_count++

        # Update Q-values Q(A)
        # Q_of_a = 

        # Update position estimate
        # position_estimate = 

        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'


    # Store GDOP values, rewards, Euclidean distance errors for each epsilon




### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon


# Plot Reward vs. Steps for each step and each epsilon


# Plot Distance Error vs. Steps for each step and each epsilon
