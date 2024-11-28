import numpy as np

# Define parameters
gamma = 0.9  # Discount factor
states = ['s1', 's2', 's3', 's4']
actions = ['up', 'down', 'left', 'right', 'stay']
tolerance = 1e-6  # Convergence tolerance
max_iterations = 1000  # Maximum number of iterations

# Define the reward function and transition probabilities for each action and state
# Example: r[s, a] and P[s, a, s'] should be defined based on the problem
# Here we use dummy values for illustration purposes

# Reward function for each state-action pair (example)
r = {
    's1': [-1, 0, -1, 1, 0],  # Example rewards for s1 for each action
    's2': [0, -1, 0, 1, 0],   # Example rewards for s2 for each action
    's3': [1, 0, -1, 0, 0],   # Example rewards for s3 for each action
    's4': [0, 0, 1, -1, 0]    # Example rewards for s4 for each action
}

# Transition probabilities (example: deterministic transitions)
P = {
    ('s1', 'up'): {'s1': 1.0},
    ('s1', 'right'): {'s2': 1.0},
    ('s2', 'down'): {'s1': 1.0},
    ('s2', 'right'): {'s4': 1.0},
    ('s3', 'left'): {'s1': 1.0},
    ('s4', 'left'): {'s2': 1.0}
    # Add other transitions as necessary
}

# Initialize the value function
v = {s: 0 for s in states}

# Iterative process for solving the BOE
iteration = 0
while iteration < max_iterations:
    delta = 0  # To check the change in value function
    for s in states:
        old_v = v[s]
        # Apply the Bellman update
        v[s] = max(
            [r[s][a] + gamma * sum(P.get((s, a), {}).get(s_prime, 0) * v[s_prime] for s_prime in states)
             for a in range(len(actions))]
        )
        delta = max(delta, abs(old_v - v[s]))

    # Check for convergence
    if delta < tolerance:
        break

    iteration += 1

print("Optimal value function:", v, " iteration:", iteration)
