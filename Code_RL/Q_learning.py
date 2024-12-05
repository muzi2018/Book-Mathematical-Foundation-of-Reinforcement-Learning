import numpy as np

# Define the size of the grid (5x5)
grid_size = (5, 5)
start = (0, 0)  # Starting position of the agent
goal = (4, 4)   # Goal position of the agent

# Define the action space (up, down, left, right)
actions = ['up', 'down', 'left', 'right']
action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

# Define the Q-table (initializing with zeros)
# Q-table dimensions: [rows x cols x actions]
Q = np.zeros((grid_size[0], grid_size[1], len(actions)))

# Reward function
def reward(state):
    if state == goal:
        return 100  # Reward for reaching the goal
    return -1  # Penalty for each step

# Valid moves in the grid
def valid_moves(state):
    x, y = state
    moves = []
    if x > 0: moves.append((x - 1, y))  # Up
    if x < grid_size[0] - 1: moves.append((x + 1, y))  # Down
    if y > 0: moves.append((x, y - 1))  # Left
    if y < grid_size[1] - 1: moves.append((x, y + 1))  # Right
    return moves

# Off-line Q-learning dataset (Simulating collected transitions)
# The dataset will be a list of tuples: (state, action, reward, next_state)
# Here we simulate transitions as an example.
# (In practice, this dataset would come from experience data or a replay buffer)
dataset = [
    ((0, 0), 'right', -1, (0, 1)),
    ((0, 1), 'right', -1, (0, 2)),
    ((0, 2), 'right', -1, (0, 3)),
    ((0, 3), 'right', -1, (0, 4)),
    ((0, 4), 'down', -1, (1, 4)),
    ((1, 4), 'left', -1, (1, 3)),
    ((1, 3), 'down', -1, (2, 3)),
    ((2, 3), 'down', -1, (3, 3)),
    ((3, 3), 'down', -1, (4, 3)),
    ((4, 3), 'right', -1, (4, 4)),  # Reached goal
]

# Q-learning parameters
learning_rate = 0.1   # Alpha (learning rate)
discount_factor = 0.9  # Gamma (discount factor)

# Off-line Q-learning training loop
for state, action, r, next_state in dataset:
    action_idx = action_map[action]  # Get the index for the action
    
    # Update Q-value using the Q-learning formula
    Q[state[0], state[1], action_idx] += learning_rate * (r + discount_factor * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action_idx])

# Visualize the learned Q-values
print("Learned Q-values:")
for row in range(grid_size[0]):
    for col in range(grid_size[1]):
        print(f"State ({row}, {col}): {Q[row, col]}")

# Extract the best action for each state (max Q-value)
policy = np.argmax(Q, axis=2)

# Map actions back to their names (for easier interpretation)
policy_str = np.vectorize(lambda x: actions[x])(policy)

# Print the learned policy in a grid format
print("\nLearned Policy:")
for row in policy_str:
    print(' '.join(row))
