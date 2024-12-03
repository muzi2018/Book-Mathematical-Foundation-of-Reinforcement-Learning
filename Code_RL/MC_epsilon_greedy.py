import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 500
actions = ["up", "down", "left", "right"]  # Action space
states = ["A", "B", "C", "D"]  # State space

# Initialization
q = {state: {action: 0 for action in actions} for state in states}  # Action-value function
returns = {state: {action: 0 for action in actions} for state in states}  # Cumulative returns
num_visits = {state: {action: 0 for action in actions} for state in states}  # Visit counts

def generate_episode(policy):
    """Simulate an episode based on the current policy."""
    episode = []
    state = np.random.choice(states)  # Start in a random state
    for _ in range(np.random.randint(3, 6)):  # Random episode length
        action = np.random.choice(actions, p=list(policy[state].values()))
        reward = np.random.random() - 0.5  # Random reward in range [-0.5, 0.5]
        next_state = np.random.choice(states)  # Random transition
        episode.append((state, action, reward))
        state = next_state
    return episode

# Initialize a random policy
policy = {
    state: {action: 1 / len(actions) for action in actions} for state in states
}

# Tracking for visualization
q_values_over_time = []

# Main loop for MC epsilon-greedy
for _ in range(num_episodes):
    # Generate an episode
    episode = generate_episode(policy)
    g = 0  # Initialize return
    visited_state_actions = set()  # Track visited state-action pairs in the episode

    # Backward pass through the episode
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        g = gamma * g + reward  # Update return
        if (state, action) not in visited_state_actions:
            visited_state_actions.add((state, action))
            # Update returns and visit count
            returns[state][action] += g
            num_visits[state][action] += 1
            # Update q-value
            q[state][action] = returns[state][action] / num_visits[state][action]

    # Policy improvement
    for state in states:
        greedy_action = max(q[state], key=q[state].get)  # Action with max q-value
        for action in actions:
            if action == greedy_action:
                policy[state][action] = 1 - epsilon + (epsilon / len(actions))
            else:
                policy[state][action] = epsilon / len(actions)

    # Record q-values for visualization
    q_values_over_time.append({state: q[state].copy() for state in states})

# Plotting results
plt.figure(figsize=(10, 6))
for state in states:
    for action in actions:
        values = [q_t[state][action] for q_t in q_values_over_time]
        plt.plot(values, label=f"Q({state},{action})")
plt.xlabel("Episodes")
plt.ylabel("Q-value")
plt.title("Convergence of Q-values in MC $\epsilon$-Greedy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
