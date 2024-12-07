import sys
import os
from grid_world import GridWorld
import random
import numpy as np
import random
import matplotlib.pyplot as plt

#
if __name__ == "__main__":      
    env = GridWorld()

    # 1. Define the transition probability matrix P_pi^T to get the stationary distribution
    '''
    lim_{k \ righarrow inf} P_pi^k = 1_n * d_pi^T (8.8)
     ->
    d_k^T = d_0^T * P_pi^k (8.7)
    (8.8) -> (8.7)
    lim_{k \ righarrow inf} d_k^T = d_0^T * 1_n * d_pi^T  = d_pi^T (8.9)
    (8.9) means that the state distribution d_k converges to a constant value d_pi
    
    then lim_k d_k^T = lim_k d_{k-1}^T * P_pi
    ->
    d_pi^T = d_pi^T * P_pi (8.10) which is stationary distribution
    '''
    P_pi_T = np.array([
        [0.3, 0.1, 0.1, 0.0],
        [0.1, 0.3, 0.0, 0.1],
        [0.6, 0.0, 0.3, 0.1],
        [0.0, 0.6, 0.6, 0.8]
    ])
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P_pi_T)
    index = np.isclose(eigenvalues, 1)
    eigenvector_1 = eigenvectors[:, index].flatten().real  # Flatten and take real part
    eigenvector_1 /= eigenvector_1.sum()
    print("Eigenvector corresponding to eigenvalue 1 (normalized):")
    print(eigenvector_1)
    exit()
    
    
    
    

    # Parameters
    gamma = 0.9  # Discount factor
    epsilon = 0.5  # Exploration rate
    num_episodes = 1
    grid_size = 2  # 5x5 grid

    # aciton 
    # up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    actions = [(0,-1), (0, 1), (-1,0), (1,0)]
    num_actions = len(actions)
    '''
    reward setup
    rboundary = -1
    rforbidden = -1
    rstep = 0
    rtarget = 1
    '''
    # reward
    env.reward_boundary = -1
    env.reward_forbidden = -1
    env.reward_step = 0
    env.reward_target = 1
    # Create grid 
    # env, row->x, column->y
    env.env_size = (grid_size, grid_size)
    env.num_states = grid_size * grid_size
    env.forbidden_states = [(1, 0)]
    env.target_state = (1, 1)
    

    def epsilon_greedy_policy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(num_actions)
        else:
            return np.argmax(q[state[0], state[1]])

    
    state_values = np.zeros((grid_size, grid_size))
    q = np.zeros((grid_size, grid_size, num_actions)) 
    returns = np.zeros((grid_size, grid_size, num_actions))
    num_visits = np.zeros((grid_size, grid_size, num_actions))
    episode_len = 1000
    def generate_episode(policy, state_values, alpha=0.1):
        episode = []
        env.start_state = (np.random.randint(grid_size), np.random.randint(grid_size)) 
        env.reset()
        state = env.start_state  
        i = 0
        while i <= episode_len:  
            env.render(1)
            action_index = epsilon_greedy_policy(state, epsilon)
            action = actions[action_index]
            next_state, reward, done, info = env.step(action)
            # Update the state value function V(s) using temporal difference (TD) learning
            # V(s) <- V(s) + alpha * [reward + gamma * V(next_state) - V(s)]
            state_values[state[0], state[1]] += alpha * (
                reward + gamma * state_values[next_state[0], next_state[1]] - state_values[state[0], state[1]]
            )
            episode.append((state, action, reward))
            state = next_state        
        return episode, state_values

    # Initialize a random policy (uniform random for each state)
    policy = {
        (i, j): {action: 1 / num_actions for action in actions} 
        for i in range(grid_size) for j in range(grid_size)
    }
    for num_ in range(num_episodes):
        episode, state_values = generate_episode(policy, state_values)
        state_values_flat = state_values.flatten()
 
        policy_matrix = np.zeros((grid_size * grid_size, num_actions))
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         state = (i, j)
        #         for action, prob in policy[state].items():
        #             action_idx = action_to_index[action]
        #             state_index = i * grid_size + j
        #             policy_matrix[state_index, action_idx] = prob
        # g = 0 
        # visited_state_actions = set()
        # for t in reversed(range(len(episode))):
        #     state, action, reward = episode[t]
        #     g = gamma * g + reward
        #     if (state, action) not in visited_state_actions:
        #         visited_state_actions.add((state, action))
        #         returns[state[0], state[1], action] += g
        #         num_visits[state[0], state[1], action] += 1
        #         q[state[0], state[1], action] = returns[state[0], state[0], action] / num_visits[state[0], state[1], action]
        
        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         greed_action = np.argmax(q[i, j])
        #         for action in range(num_actions):
        #             if action == greed_action:
        #                 policy[(i, j)][actions[action]] = 1 - epsilon + (epsilon / num_actions)
        #         else:
        #             policy[(i, j)][actions[action]] = epsilon / num_actions
        
        # Record q-values for visualization
        # q_values_over_time.append(q.copy())

        
        
# # Plotting results
# plt.figure(figsize=(10, 6))
# for i in range(grid_size):
#     for j in range(grid_size):
#         for action in range(num_actions):
#             values = [q_t[i, j, action] for q_t in q_values_over_time]  # Plot the first Q-value
#             plt.plot(values, label=f"Q({i},{j},{actions[action]})")
# plt.xlabel("Episodes")
# plt.ylabel("Q-value")
# plt.title("Convergence of Q-values in MC $\epsilon$-Greedy")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# # plt.tight_layout()
# plt.show()
        
    
    
    