import numpy as np
import matplotlib.pyplot as plt

# Define a simple MDP
states = [0, 1]  # Example: two states
actions = [0, 1]  # Example: two actions
transition_probs = {
    0: {0: [(1, 0, 0)], 1: [(1, 1, 1)]},  # state 0 -> {action -> (prob, next_state, reward)}
    1: {0: [(1, 0, 0)], 1: [(1, 1, 1)]},  # state 1
}
gamma = 0.9  # Discount factor

def evaluate_policy(policy, V, theta=1e-6):
    """Policy Evaluation: Iteratively calculates state values for a given policy."""
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]
            V[s] = sum(p * (r + gamma * V[s_]) for p, s_, r in transition_probs[s][a])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration():
    """Policy Iteration: Iteratively improves the policy."""
    policy = [0 for _ in states]  # Initial policy (arbitrary)
    V = np.zeros(len(states))    # Initial state values
    history = []
    while True:
        V = evaluate_policy(policy, V)  # Evaluate current policy
        history.append(V.copy())
        policy_stable = True
        for s in states:
            old_action = policy[s]
            policy[s] = np.argmax([sum(p * (r + gamma * V[s_]) for p, s_, r in transition_probs[s][a]) for a in actions])
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return V, history

def value_iteration(theta=1e-6):
    """Value Iteration: Iteratively calculates optimal state values."""
    V = np.zeros(len(states))  # Initial state values
    history = []
    while True:
        delta = 0
        history.append(V.copy())
        for s in states:
            v = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_]) for p, s_, r in transition_probs[s][a]) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, history

def truncated_policy_iteration(truncation=3, theta=1e-6):
    """Truncated Policy Iteration: Combines elements of Policy Iteration and Value Iteration."""
    policy = [0 for _ in states]  # Initial policy
    V = np.zeros(len(states))    # Initial state values
    history = []
    while True:
        # Perform truncated policy evaluation
        for _ in range(truncation):
            delta = 0
            for s in states:
                v = V[s]
                a = policy[s]
                V[s] = sum(p * (r + gamma * V[s_]) for p, s_, r in transition_probs[s][a])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        history.append(V.copy())
        # Policy improvement
        policy_stable = True
        for s in states:
            old_action = policy[s]
            policy[s] = np.argmax([sum(p * (r + gamma * V[s_]) for p, s_, r in transition_probs[s][a]) for a in actions])
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return V, history

# Run the algorithms
v_policy, history_policy = policy_iteration()
v_value, history_value = value_iteration()
v_truncated, history_truncated = truncated_policy_iteration()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.max(history_policy, axis=1), label='Policy iteration', marker='o', linestyle='-')
plt.plot(np.max(history_value, axis=1), label='Value iteration', marker='D', linestyle='--')
plt.plot(np.max(history_truncated, axis=1), label='Truncated policy iteration', marker='s', linestyle='-.')
plt.axhline(y=np.max(v_value), color='r', label='Optimal state value (v*)', linestyle='-')
plt.xlabel('Iteration (k)')
plt.ylabel('Value Function Convergence (v_k)')
plt.title('Comparison of Policy Iteration, Value Iteration, and Truncated Policy Iteration')
plt.legend()
plt.grid()
plt.show()
