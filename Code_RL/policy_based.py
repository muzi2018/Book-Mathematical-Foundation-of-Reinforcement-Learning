import numpy as np

'''
9.2 Metrics for defining optimal policies
'''
'''
Metric 1: Average state value
'''
# # Example states, stationary distribution, and state-value function
# states = ['s1', 's2', 's3']  # States
# d = np.array([0.5, 0.3, 0.2])  # Stationary distribution d(S)
# v_pi = np.array([12, 8, 6])    # State-value function v_pi(S)

# # Draw samples based on the stationary distribution
# num_samples = 10000  # Number of samples
# sampled_states = np.random.choice(states, size=num_samples, p=d)

# # Map state names to indices
# state_indices = {state: i for i, state in enumerate(states)}

# # Compute the expected value using sampling
# sampled_values = [v_pi[state_indices[state]] for state in sampled_states]
# v_bar_pi = np.mean(sampled_values)

# print(f"Sampled expected value (v_bar_pi): {v_bar_pi:.4f}")


'''
stationary distribution
'''

# # Define states, actions, and other parameters
# states = [0, 1, 2]  # Example states
# actions = ["a1", "a2"]  # Example actions

# # Transition probabilities p(s' | s, a)
# # Shape: (num_states, num_states, num_actions)
# # p(s_j | s_i, a) = T[i, j, a]
# T = np.array([
#     [[0.8, 0.2], [0.1, 0.9], [0.0, 0.0]],  # Transitions from state 0
#     [[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]],  # Transitions from state 1
#     [[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]]   # Transitions from state 2
# ])

# # Policy probabilities pi(a | s)
# # Shape: (num_states, num_actions)
# pi = np.array([
#     [0.5, 0.5],  # Probabilities for state 0
#     [0.7, 0.3],  # Probabilities for state 1
#     [0.4, 0.6]   # Probabilities for state 2
# ])

# # Compute P_pi
# num_states = len(states)
# P_pi = np.zeros((num_states, num_states))

# for i in range(num_states):
#     for j in range(num_states):
#         # Sum over all actions
#         P_pi[i, j] = sum(pi[i, a] * T[i, j, a] for a in range(len(actions)))

# # Print the resulting P_pi
# print("State transition probability matrix P_pi:")
# print(P_pi)


'''
Suppose that an agent collects rewards
'''
# # Define the environment
# states = [0, 1, 2]  # Example states
# actions = [0, 1]  # Example actions

# # Transition probabilities p(s' | s, a)
# transition_probabilities = {
#     #s  a   p_s
#     0: {0: [0.7, 0.3, 0.0], 1: [0.2, 0.8, 0.0]},
#     1: {0: [0.1, 0.6, 0.3], 1: [0.0, 0.5, 0.5]},
#     2: {0: [0.0, 0.0, 1.0], 1: [0.0, 0.0, 1.0]}
# }
# # Reward function r(s, a)
# reward_function = {
#     #s  a  r
#     0: {0: 1, 1: 2},
#     1: {0: 0, 1: 3},
#     2: {0: 5, 1: 5}
# }

# # Policy probabilities pi(a | s)
# policy = {
#     #s  #a
#     0: [0.6, 0.4],  # Probabilities for actions at state 0
#     1: [0.5, 0.5],  # Probabilities for actions at state 1
#     2: [1.0, 0.0]   # Probabilities for actions at state 2
# }

# # Simulate an agent following the policy
# np.random.seed(42)  # For reproducibility
# num_episodes = 10
# horizon = 5  # Maximum steps per episode

# for episode in range(num_episodes):
#     state = np.random.choice(states)  # Start in a random state
#     total_reward = 0
#     print(f"Episode {episode + 1}:")
    
#     for step in range(horizon):
#         # Choose action based on policy
#         action = np.random.choice(actions, p=policy[state])
        
#         # Collect reward
#         reward = reward_function[state][action]
#         total_reward += reward
        
#         # Transition to next state
#         next_state_probs = transition_probabilities[state][action]
#         next_state = np.random.choice(states, p=next_state_probs)
        
#         print(f"  Step {step + 1}: State={state}, Action={action}, Reward={reward}, Next State={next_state}")
#         state = next_state
    
#     print(f"Total Reward: {total_reward}\n")

'''
metric average state value defining optimal policies
'''


