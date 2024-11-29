import numpy as np
"""
    +---+---+---+
    | s1| s2| s3|
    +---+---+---+
    | s4| s5| s6|
    +---+---+---+
    | s7| s8| s9|
    +---+---+---+

"""
# Define the gridworld
grid = np.array([
    ['s1', 's2', 's3'],
    ['s4', 's5', 's6'],
    ['s7', 's8', 's9']
])

# Define actions
actions = ['up', 'down', 'left', 'right']

# Reward function
def reward(state):
    if state == 's9':  # Goal state
        return 10
    elif state == 's5':  # Danger state
        return -5
    else:
        return 0

# Simulate a trajectory
trajectory = [('s1', 'right', 's2', 0),
              ('s2', 'down', 's5', -5),
              ('s5', 'down', 's8', 0),
              ('s8', 'right', 's9', 10)]

# Calculate return
gamma = 0.9
G = sum(r * (gamma ** t) for t, (_, _, _, r) in enumerate(trajectory))
print(f"Discounted Return: {G}")
