import numpy as np
import matplotlib.pyplot as plt

# Define a simple environment (linear states for simplicity)
states = ['s1', 's2', 's3', 's4', 's5']
rewards = {'s1': 0, 's2': 0, 's3': 1, 's4': 0, 's5': 0}  # Reward at 's3'
discount_rate = 0.9
num_episodes = 1000

# Initialize value estimates for both methods
value_estimates_every_visit = {state: 0 for state in states}
value_estimates_first_visit = {state: 0 for state in states}

# To track returns for both methods
returns_every_visit = {state: [] for state in states}
returns_first_visit = {state: [] for state in states}

# Function to generate a random episode
def generate_episode():
    episode = []
    state = np.random.choice(states)
    while state != 's3':  # Terminal state with a reward
        next_state = np.random.choice(states)
        episode.append((state, rewards[state]))
        state = next_state
    episode.append(('s3', rewards['s3']))
    return episode

# Monte Carlo estimation for both Every-Visit and First-Visit methods
for _ in range(num_episodes):
    episode = generate_episode()
    G = 0  # Initialize return
    visited_states = set()

    for state, reward in reversed(episode):
        G = reward + discount_rate * G

        # Every-Visit MC
        returns_every_visit[state].append(G)
        value_estimates_every_visit[state] = np.mean(returns_every_visit[state])

        # First-Visit MC
        if state not in visited_states:
            returns_first_visit[state].append(G)
            value_estimates_first_visit[state] = np.mean(returns_first_visit[state])
            visited_states.add(state)

# Plot the value estimates for comparison
states_indices = range(len(states))
every_visit_values = [value_estimates_every_visit[s] for s in states]
first_visit_values = [value_estimates_first_visit[s] for s in states]

plt.figure(figsize=(8, 6))
plt.bar(states_indices, every_visit_values, alpha=0.6, label='Every-Visit MC', color='blue')
plt.bar(states_indices, first_visit_values, alpha=0.6, label='First-Visit MC', color='orange', width=0.4)

plt.xticks(states_indices, states)
plt.xlabel('States')
plt.ylabel('Value Estimates')
plt.title('Comparison of Every-Visit vs First-Visit Monte Carlo Value Estimates')
plt.legend()
plt.show()
