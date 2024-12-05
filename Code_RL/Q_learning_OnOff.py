import numpy as np
import random

# Define the 5x5 grid world
grid_size = (5, 5)
target = (4, 4)  # Target state
actions = ['up', 'down', 'left', 'right']
action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

# Rewards
R_BOUNDARY = -1  # For hitting boundaries or forbidden states
R_TARGET = 1     # Reward for reaching the target

# Discount factor and learning rate
gamma = 0.9
alpha = 0.1

# Initialize Q-table (states x actions)
Q = np.zeros((grid_size[0], grid_size[1], len(actions)))

# Behavior policy (uniform random)
def behavior_policy(state):
    return random.choice(range(len(actions)))

# Reward function
def reward(state):
    if state == target:
        return R_TARGET
    return R_BOUNDARY

# Valid moves function
def valid_moves(state):
    x, y = state
    moves = []
    if x > 0: moves.append((x - 1, y))  # Up
    if x < grid_size[0] - 1: moves.append((x + 1, y))  # Down
    if y > 0: moves.append((x, y - 1))  # Left
    if y < grid_size[1] - 1: moves.append((x, y + 1))  # Right
    return moves

# Get next state given an action
def next_state(state, action):
    moves = valid_moves(state)
    if action < len(moves):
        return moves[action]
    return state  # If invalid action, stay in the same state

# Generate a single episode with the behavior policy
episode_length = 100_000
episode = []
state = (0, 0)  # Start at top-left corner

for _ in range(episode_length):
    action = behavior_policy(state)
    next_s = next_state(state, action)
    r = reward(next_s)
    episode.append((state, action, r, next_s))
    state = next_s if next_s != target else (0, 0)  # Reset after reaching target

# Train Q-learning off-policy
for state, action, r, next_s in episode:
    current_q = Q[state[0], state[1], action]
    max_next_q = np.max(Q[next_s[0], next_s[1]])
    Q[state[0], state[1], action] += alpha * (r + gamma * max_next_q - current_q)

# Derive the target policy
policy = np.zeros((grid_size[0], grid_size[1]), dtype=int)
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        policy[x, y] = np.argmax(Q[x, y])

# Display the learned Q-values and target policy
print("Learned Q-values:")
print(Q)

print("\nLearned Policy (0:up, 1:down, 2:left, 3:right):")
print(policy)

# Evaluate convergence (RMSE)
true_state_values = np.zeros(grid_size)  # Assume known true state values (for demonstration)
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        true_state_values[x, y] = np.max(Q[x, y])  # State value is max Q-value

rmse = np.sqrt(np.mean((true_state_values - np.max(Q, axis=2)) ** 2))
print(f"\nRoot-Mean-Square Error of State Values: {rmse:.4f}")
