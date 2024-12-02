import numpy as np
 #Define the MDP parameters
S = [0, 1, 2] # State space
A = [0, 1] # Action space

gamma = 0.9 # Discount factor

# Reward probabilities: P_r[(s, a)] -> list of (reward, probability)
P_r = {
    (0, 0): [(1, 1.0)],
    (0, 1): [(0, 1.0)],
    (1, 0): [(2, 1.0)],
    (1, 1): [(0, 1.0)],
    (2, 0): [(1, 1.0)],
    (2, 1): [(2, 1.0)],
}


# Transition probabilities: P_s[s' | s, a]
P_s = {
    (0, 0): {0: 0.5, 1: 0.5},
    (0, 1): {0: 1.0},
    (1, 0): {2: 1.0},
    (1, 1): {1: 0.7, 0: 0.3},
    (2, 0): {0: 0.4, 2: 0.6},
    (2, 1): {2: 1.0},
}

def policy_iteration():
    # Initialize policy arbitrarily
    policy = {s: np.random.choice(A) for s in S}
    
    def evaluate_policy (policy):
        V = np.zeros(len(S))
        while True:
            delta = 0
            New_V = V.copy()
            for s in S:
                a = policy[s]
                # 
                for reward, prob_r in P_r.get((s, a), []):
                    aaa = prob_r * (reward) 
                # v_s = sum(
                #     prob_r * (reward) for reward, prob_r in P_r.get((s, a), [])
                # )
exit()



# import numpy as np

# # Define the number of states and actions
# num_states = 4  # Number of states
# num_actions = 2  # Number of actions

# # Create a 3D transition probability matrix P[state, action, next_state]
# P = np.array([
#     [[0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0]],  # Transitions from state 0 under action 0 and 1
#     [[0.0, 0.9, 0.1, 0.0], [0.0, 0.8, 0.2, 0.0]],  # Transitions from state 1 under action 0 and 1
#     [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],  # Transitions from state 2 (absorbing)
#     [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],  # Transitions from state 3 (absorbing)
# ])

# # Define the reward function for each state and action
# r = np.array([
#     [1, 0],  # Rewards for actions at state 0
#     [0, 1],  # Rewards for actions at state 1
#     [0, 0],  # Rewards for actions at state 2
#     [0, 0],  # Rewards for actions at state 3
# ])

# gamma = 0.9  # Discount factor

# # Initialize policies
# policy_k = np.random.choice(num_actions, size=num_states)  # Random policy
# policy_k1 = np.zeros(num_states, dtype=int)  # Initializing the next policy

# # Function for policy evaluation
# def policy_evaluation(P, r, gamma, policy, tol=1e-6, max_iterations=1000):
#     num_states = len(r)
#     v = np.zeros(num_states)  # Initial value function
#     v_history = np.zeros((max_iterations, num_states))
#     for iteration in range(max_iterations):
#         v_new = np.zeros(num_states)
#         for state in range(num_states):
#             action = policy[state]
#             v_new[state] = r[state, action] + gamma * np.sum(P[state, action, :] * v)
#         v_history[iteration] = v_new  # Store the current v_new in history
#         if np.linalg.norm(v_new - v, ord=np.inf) < tol:
#             v_history = v_history[:iteration + 1]  # Trim unused rows
#             break
#         v = v_new
    
#     return v, v_history

# # Function for policy improvement
# def policy_improvement(P, r, gamma, v):
#     num_states = len(r)
#     new_policy = np.zeros(num_states, dtype=int)
    
#     for state in range(num_states):
#         action_values = np.zeros(num_actions)
#         for action in range(num_actions):
#             action_values[action] = r[state, action] + gamma * np.sum(P[state, action, :] * v)
#         new_policy[state] = np.argmax(action_values)
    
#     return new_policy

# # Run policy iteration for 100 iterations
# num_iterations = 100
# for i in range(num_iterations):
#     # Evaluate the current policy
#     v_policy_k, v_policy_k_buff = policy_evaluation(P, r, gamma, policy_k)
    
#     # Improve the policy based on the value function

    
#     # Check the change in policy
#     if np.array_equal(policy_k, policy_k1):
#         print(f"Policy converged after {i+1} iterations.")
#         break
#     policy_k1 = policy_improvement(P, r, gamma, v_policy_k)
    
#     v_policy_k1, v_policy_k1_buff = policy_evaluation(P, r, gamma, policy_k1)
#     difference = v_policy_k_buff - v_policy_k1_buff
#     print("Difference between v_{π_k} and v_{π_{k+1}}:", difference)
#     # Update policy_k for the next iteration
#     policy_k = policy_k1

# # Print final policies and value functions
# print("Final policy π_k:", policy_k)
# print("Final policy π_{k+1}:", policy_k1)

# v_policy_k, _ = policy_evaluation(P, r, gamma, policy_k)
# v_policy_k1, _ = policy_evaluation(P, r, gamma, policy_k1)

# print("Value function v_{π_k}:", v_policy_k)
# print("Value function v_{π_{k+1}}:", v_policy_k1)

# exit()








# ## Proof of Lemma 4.1
# import numpy as np

# # Define parameters
# gamma = 0.9  # Discount factor
# num_states = 2  # Number of states
# num_actions = 2  # Number of actions
# iterations = 10  # Number of iterations for the simulation

# # Transition probabilities and rewards
# P = np.array([[0.8, 0.2],  # Transition probabilities for action 0
#               [0.2, 0.8]])  # Transition probabilities for action 1
# rewards = np.array([0, 1])  # Rewards for each state

# # Initialize value functions
# v_pi_k = np.zeros(num_states)
# v_pi_k_plus_1 = np.zeros(num_states)

# # Policy iteration process
# for k in range(iterations):
#     # Policy evaluation step
#     v_pi_k_plus_1 = rewards + gamma * np.dot(P, v_pi_k)

#     # Policy improvement step
#     # Here we assume a simple policy improvement based on the max value
#     v_pi_k = v_pi_k_plus_1

#     # Print the values at each iteration
#     print(f"Iteration {k + 1}: v_pi_k = {v_pi_k}")

#     # Check for convergence
#     if np.allclose(v_pi_k, v_pi_k_plus_1):
#         print("Convergence reached.")
#         break

# # Final values
# print("Final value function:", v_pi_k)