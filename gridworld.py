import numpy as np
import random
import matplotlib.pyplot as plt
import copy

state_values = np.zeros((5,5))

initial_state = (0,0)
terminal_state = (4,4)
wall = [(3,4),(3,3),(3,2)]

# state_values[terminal_state] = 1
state_values[wall[0]] = -1
state_values[wall[1]] = -1
state_values[wall[2]] = -1

all_actions = ['u','d','l','r']

episodes = 1000
episode = 0

epsilon = 0.5
gamma = 0.5

def ChooseAction(current_position):
    highest_expected_reward = 0
    if np.random.uniform(0,1) <= epsilon:
        action = np.random.choice(all_actions)
        return action
    else:
        for move in all_actions:
            # potential bug? If at a wall and surrounding  moves have less value, could
            # choose to move into the wall forever, going nowhere
            expected_reward = state_values[TakeAction(move, current_position)]
            if expected_reward >= highest_expected_reward:
                action = move
                highest_expected_reward = expected_reward
        return action

def TakeAction(action, current_position):
    curr_pos = list(current_position)
    # print(curr_pos)
    if action == 'u':
        curr_pos[1] += 1
    elif action == 'd':
        curr_pos[1] -= 1
    elif action == 'l':
        curr_pos[0] -= 1
    elif action == 'r':
        curr_pos[0] += 1

    if ( 0 <= curr_pos[0] <= 4) & ( 0 <= curr_pos[1] <= 4)  & ( tuple(curr_pos) not in wall ):
        return (tuple(curr_pos))
    else: # return the current position if the move is invalid
        return current_position

current_position = initial_state
states_visited = []
value_deltas = []
state_value_deltas = []

while episode < episodes:
    if current_position == terminal_state: # Reached end
        reward = 1
        episode += 1
        print(episode)
        current_position = initial_state # reset to starting point
        old_values = copy.deepcopy(state_values)
        for state in reversed(states_visited):
            # Perform value iteration
            reward = state_values[state] + gamma * (reward - state_values[state])
            # Rounding here does not make the result incorrect,
            # it just makes the difference between state values larger,
            # and thus its easier to see at a glance that value iteration works
            state_values[state] = round(reward, 2)
        new_values = copy.deepcopy(state_values)
        state_value_deltas.append(new_values - old_values)
        states_visited = []
    else: # take another action
        action = ChooseAction(current_position)
        previous_position = current_position
        current_position = TakeAction(action, current_position)
        if previous_position == current_position:
            pass # If we chose an illegal move, don't append it to visited states.
            # This makes is such that the value iteration doesn't give additional value
            # to states next to the wall.
        else:
            states_visited.append(current_position)

print(state_values)
print(state_value_deltas)

plt.figure()
for i in range(5):
    plt.plot(range(len(state_value_deltas)), [values[i] for values in state_value_deltas])

plt.xlabel("Iterations")
plt.ylabel("Change in value function")
plt.title("Change in value function vs. iterations")
plt.show()








