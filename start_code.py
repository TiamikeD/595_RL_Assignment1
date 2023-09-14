# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

np.random.seed(19680801)

POSSIBLE_ACTIONS = ["012","013","014","023","024","034","123","134","234","124"]

def plot3d(anchor_points = [], target_point = [], iterator = []):
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


def get_q_values(rewards = [], action_count=0):

    return [0,0,0,0]

### Step 1: Initialize the problem parameters.

num_anchor_nodes = 5
total_steps = 100000

# Initialize anchor node positions and target position
anchor_positions = np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)
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

action_count = 0
rewards = [0,0,0,0]


for epsilon in epsilons:

    action_count = 0

    # Initializing the 'position_stimate' to 'position_initial_estimate'


    position_estimate = position_initial_estimate.copy()




    # Initialize action counts for each epsilon

    # Initialize Q-values for each epsilon

    q_values = get_q_values(rewards, action_count)

    # Main loop for the epsilon-greedy bandit algorithm

#    for i in range(total_steps):
#        print(i)




        # Select three anchor nodes (action A)



            # Exploration: Choose random actions


            # Exploitation: Choose actions with highest Q-values

        # Code for determining pseudoranges

plot3d(anchor_positions, target_position, position_initial_estimate)

#selected_positions = [i for i in range(10)]

#pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(3)]

        # Determine the 'jacobian' matrix based on the selected anchor nodes


        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'


        # Determine the 'reward' R(A) using the 'gdop' value


        # Update action counts N(A)


        # Update Q-values Q(A)


        # Update position estimate


        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'


    # Store GDOP values, rewards, Euclidean distance errors for each epsilon




### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon


# Plot Reward vs. Steps for each step and each epsilon


# Plot Distance Error vs. Steps for each step and each epsilon
