# import numpy as np

# '''
# step 1: policy evaluation
# '''
# step1 = False
# step2 = True
# if step1:
#     # Define parameters
#     gamma = 0.9  # Discount factor
#     threshold = 1e-4  # Convergence threshold
#     states = ['s1', 's2', 's3', 's4']  # State space
#     actions = ['a1', 'a2', 'a3', 'a4', 'a5']  # Action space

#     # Transition and reward model (simplified for one policy)
#     transition_model = {
#         's1': {'a3': [('s3', 1, 0)]},
#         's2': {'a3': [('s4', 1, 1)]},
#         's3': {'a2': [('s4', 1, 1)]},
#         's4': {'a5': [('s4', 1, 1)]},
#     }

#     # Define a fixed policy π
#     policy = {
#         's1': 'a3',
#         's2': 'a3',
#         's3': 'a2',
#         's4': 'a5',
#     }

#     # Initialize value function
#     v_pi = {state: 0 for state in states}
#     iteration =0
#     # Policy evaluation: Iterative computation of v_pi
#     while True:
#         delta = 0
#         v_next = v_pi.copy()
#         for s in states:
#             action = policy[s]
#             q_value = 0
#             for next_state, prob, reward in transition_model[s][action]:
#                 q_value += prob * (reward + gamma * v_pi[next_state])
#             delta = max(delta, abs(v_next[s] - q_value))
#             v_next[s] = q_value
#         v_pi = v_next
#         iteration += 1
#         if delta < threshold:
#             break

#     print(f"\nConverged after {iteration} iterations.")
#     # Print the evaluated state value function
#     print("Evaluated State Value Function:")
#     for s in states:
#         print(f"V({s}) = {v_pi[s]:.4f}")

# '''
# step 2: policy improvement
# '''

# if step2: 
#     # Define parameters
#     gamma = 0.9  # Discount factor
#     states = ['s1', 's2', 's3', 's4']  # State space
#     actions = ['a1', 'a2', 'a3', 'a4', 'a5']  # Action space

#     # Transition and reward model
#     transition_model = {
#         's1': {
#             'a1': [('s1', 1, -1)],
#             'a2': [('s2', 1, -1)],
#             'a3': [('s3', 1, 0)],
#             'a4': [('s1', 1, -1)],
#             'a5': [('s1', 1, 0)],
#         },
#         's2': {
#             'a1': [('s2', 1, -1)],
#             'a2': [('s2', 1, -1)],
#             'a3': [('s4', 1, 1)],
#             'a4': [('s1', 1, 0)],
#             'a5': [('s2', 1, -1)],
#         },
#         's3': {
#             'a1': [('s1', 1, 0)],
#             'a2': [('s4', 1, 1)],
#             'a3': [('s3', 1, -1)],
#             'a4': [('s3', 1, -1)],
#             'a5': [('s3', 1, 0)],
#         },
#         's4': {
#             'a1': [('s2', 1, -1)],
#             'a2': [('s4', 1, -1)],
#             'a3': [('s4', 1, -1)],
#             'a4': [('s3', 1, 0)],
#             'a5': [('s4', 1, 1)],
#         },
#     }

#     # Assume v_pi has been calculated
#     v_pi = {
#         's1': 0,
#         's2': 0.9,
#         's3': 1.8,
#         's4': 1.0,
#     }

#     # Policy improvement step
#     new_policy = {}

#     for s in states:
#         q_values = {}
#         for a in actions:
#             q_value = 0
#             for next_state, prob, reward in transition_model[s][a]:
#                 q_value += prob * (reward + gamma * v_pi[next_state])
#             q_values[a] = q_value
        
#         # Select the action with the maximum Q-value
#         best_action = max(q_values, key=q_values.get)
#         new_policy[s] = best_action

#     # Display the improved policy
#     print("Improved Policy:")
#     for s in states:
#         print(f"π({s}) = {new_policy[s]}")
        
#     # Function to calculate the value function for a given policy
#     def evaluate_policy(policy, transition_model, gamma, threshold=1e-4):
#         v = {s: 0 for s in states}  # Initialize value function
#         while True:
#             delta = 0
#             for s in states:
#                 action = policy[s]
#                 new_value = 0
#                 for next_state, prob, reward in transition_model[s][action]:
#                     new_value += prob * (reward + gamma * v[next_state])
#                 delta = max(delta, abs(v[s] - new_value))
#                 v[s] = new_value
#             if delta < threshold:
#                 break
#         return v

#     # Calculate the value function for the new policy
#     v_pi_new = evaluate_policy(new_policy, transition_model, gamma)

#     # Compare the new policy value with the old policy value
#     print("Comparing Policies:")
#     improved = False
#     for s in states:
#         old_value = v_pi[s]
#         new_value = v_pi_new[s]
#         print(f"State {s}: V_old = {old_value:.4f}, V_new = {new_value:.4f}")
#         if new_value > old_value:
#             improved = True

#     if improved:
#         print("\nThe new policy is better than the previous policy!")
#     else:
#         print("\nThe new policy is not better than the previous policy.")




import numpy as np

# Define the number of states and actions
num_states = 4  # Number of states
num_actions = 2  # Number of actions

# Create a 3D transition probability matrix P[state, action, next_state]
P = np.array([
    [[0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0]],  # Transitions from state 0 under action 0 and 1
    [[0.0, 0.9, 0.1, 0.0], [0.0, 0.8, 0.2, 0.0]],  # Transitions from state 1 under action 0 and 1
    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],  # Transitions from state 2 (absorbing)
    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],  # Transitions from state 3 (absorbing)
])

# Define the reward function for each state and action
r = np.array([
    [1, 0],  # Rewards for actions at state 0
    [0, 1],  # Rewards for actions at state 1
    [0, 0],  # Rewards for actions at state 2
    [0, 0],  # Rewards for actions at state 3
])

gamma = 0.9  # Discount factor

# Initialize policies
policy_k = np.random.choice(num_actions, size=num_states)  # Random policy
policy_k1 = np.zeros(num_states, dtype=int)  # Initializing the next policy

# Function for policy evaluation
def policy_evaluation(P, r, gamma, policy, tol=1e-6, max_iterations=1000):
    num_states = len(r)
    v = np.zeros(num_states)  # Initial value function
    
    for iteration in range(max_iterations):
        v_new = np.zeros(num_states)
        for state in range(num_states):
            action = policy[state]
            v_new[state] = r[state, action] + gamma * np.sum(P[state, action, :] * v)
        if np.linalg.norm(v_new - v, ord=np.inf) < tol:
            break
        v = v_new
    
    return v

# Function for policy improvement
def policy_improvement(P, r, gamma, v):
    num_states = len(r)
    new_policy = np.zeros(num_states, dtype=int)
    
    for state in range(num_states):
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            action_values[action] = r[state, action] + gamma * np.sum(P[state, action, :] * v)
        new_policy[state] = np.argmax(action_values)
    
    return new_policy

# Run policy iteration for 100 iterations
num_iterations = 100
for i in range(num_iterations):
    # Evaluate the current policy
    v_policy_k = policy_evaluation(P, r, gamma, policy_k)
    
    # Improve the policy based on the value function

    
    # Check the change in policy
    if np.array_equal(policy_k, policy_k1):
        print(f"Policy converged after {i+1} iterations.")
        break
    policy_k1 = policy_improvement(P, r, gamma, v_policy_k)
    
    v_policy_k1 = policy_evaluation(P, r, gamma, policy_k1)
    difference = v_policy_k - v_policy_k1
    print("Difference between v_{π_k} and v_{π_{k+1}}:", difference)
    # Update policy_k for the next iteration
    policy_k = policy_k1

# Print final policies and value functions
print("Final policy π_k:", policy_k)
print("Final policy π_{k+1}:", policy_k1)

v_policy_k = policy_evaluation(P, r, gamma, policy_k)
v_policy_k1 = policy_evaluation(P, r, gamma, policy_k1)

print("Value function v_{π_k}:", v_policy_k)
print("Value function v_{π_{k+1}}:", v_policy_k1)

exit()








## Proof of Lemma 4.1
import numpy as np

# Define parameters
gamma = 0.9  # Discount factor
num_states = 2  # Number of states
num_actions = 2  # Number of actions
iterations = 10  # Number of iterations for the simulation

# Transition probabilities and rewards
P = np.array([[0.8, 0.2],  # Transition probabilities for action 0
              [0.2, 0.8]])  # Transition probabilities for action 1
rewards = np.array([0, 1])  # Rewards for each state

# Initialize value functions
v_pi_k = np.zeros(num_states)
v_pi_k_plus_1 = np.zeros(num_states)

# Policy iteration process
for k in range(iterations):
    # Policy evaluation step
    v_pi_k_plus_1 = rewards + gamma * np.dot(P, v_pi_k)

    # Policy improvement step
    # Here we assume a simple policy improvement based on the max value
    v_pi_k = v_pi_k_plus_1

    # Print the values at each iteration
    print(f"Iteration {k + 1}: v_pi_k = {v_pi_k}")

    # Check for convergence
    if np.allclose(v_pi_k, v_pi_k_plus_1):
        print("Convergence reached.")
        break

# Final values
print("Final value function:", v_pi_k)