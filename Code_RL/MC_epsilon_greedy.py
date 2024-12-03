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
    
    # Parameters
    gamma = 0.9  # Discount factor
    epsilon = 0.2  # Exploration rate
    num_episodes = 10
    grid_size = 5  # 5x5 grid

    ## aciton 
    ### up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    actions = [(0,-1), (0, 1), (-1,0), (1,0)]
    num_actions = len(actions)
    '''
    reward setup
    rforbidden = -1
    rtarget = 1
    rstep = 0
    '''
    ## reward
    env.reward_boundary = -1
    env.reward_forbidden = -10
    env.reward_step = 0
    env.reward_target = 1
    
    # Create grid 
    ## env, row->x, column->y
    env.env_size = (grid_size, grid_size)
    env.num_states = grid_size * grid_size
    env.start_state = (np.random.randint(grid_size), np.random.randint(grid_size)) 
    env.forbidden_states = [(1, 1),(2,1),(2,2),(1,3),(1,4),(3,3)]
    env.target_state = (2, 3)
    env.reset()
    # Generate all state coordinates
    # states = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    # states = np.array(states)
    # values = np.zeros(env.num_states)
    # state_values = np.array([[s[0], s[1], v] for s, v in zip(states, values)])
    state_values = np.zeros((grid_size, grid_size))  # A grid for values

    # Q-value initialization
    q = np.zeros((grid_size, grid_size, num_actions)) 
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         q[i,j] = set(actions)
    # # Example: printing the actions for each state
    # for state in q:
    #     print(f"Actions for state {state}: {q[state]}")
    
    returns = np.zeros((grid_size, grid_size, num_actions))  # Cumulative returns
    num_visits = np.zeros((grid_size, grid_size, num_actions))  # Visit counts

    # Function to choose action based on epsilon-greedy policy
    def epsilon_greedy_policy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(num_actions)  # Explore
        else:
            # print("epsilon_greedy_policy: ", np.argmax(q[state[0], state[1]]))
            return np.argmax(q[state[0], state[1]])  # Exploit
    
    def generate_episode(policy, state_values, alpha=0.1):
        '''
        Generate an episode using the given policy and update state values.

        Args:
            policy: The policy being used for action selection.
            state_values: A dictionary or array holding the state values V(s).
            alpha: Learning rate for updating state values.
        
        Returns:
            episode: A list of (state, action, reward) tuples for the generated episode.
        '''
        episode = []
        state = (np.random.randint(grid_size), np.random.randint(grid_size))  # Start in a random state

        while state != env.target_state:  # Continue until the target state is reached
            # Select an action using the epsilon-greedy policy
            action_index = epsilon_greedy_policy(state, epsilon)
            action = actions[action_index]

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)
            
            # Update the state value function V(s) using temporal difference (TD) learning
            # V(s) <- V(s) + alpha * [reward + gamma * V(next_state) - V(s)]
            state_values[state[0], state[1]] += alpha * (
                reward + gamma * state_values[next_state[0], next_state[1]] - state_values[state[0], state[1]]
            )
            # Add the current transition to the episode
            episode.append((state, action, reward))

            # Move to the next state
            state = next_state

        return episode, state_values


    # Initialize a random policy (uniform random for each state)
    policy = {
        (i, j): {action: 1 / num_actions for action in actions} 
        for i in range(grid_size) for j in range(grid_size)
    }
    # Tracking for visualization
    q_values_over_time = []
    # Main loop for Monte Carlo epsilon-greedy
    for num_ in range(num_episodes):
        # Generate an episode
        print("episode is ", num_)
        episode, state_values = generate_episode(policy, state_values)
        state_values_flat = state_values.flatten()
 
        # Add policy     
        policy_matrix = np.zeros((grid_size * grid_size, num_actions))
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                # Extract the probability for each action from the policy
                for action, prob in policy[state].items():
                    action_idx = action_to_index[action]
                    # Find the corresponding row for the state and set the probability for that action
                    state_index = i * grid_size + j
                    policy_matrix[state_index, action_idx] = prob
                    
       # render
        env.render()
        env.add_state_values(state_values_flat)
        env.add_policy(policy_matrix)
        
        g = 0 # Initialize return
        visited_state_actions = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            g = gamma * g + reward
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                returns[state[0], state[1], action] += g
                num_visits[state[0], state[1], action] += 1
                q[state[0], state[1], action] = returns[state[0], state[0], action] / num_visits[state[0], state[1], action]
        
        for i in range(grid_size):
            for j in range(grid_size):
                greed_action = np.argmax(q[i, j])
                for action in range(num_actions):
                    if action == greed_action:
                        policy[(i, j)][actions[action]] = 1 - epsilon + (epsilon / num_actions)
                else:
                    policy[(i, j)][actions[action]] = epsilon / num_actions
        
        # Record q-values for visualization
        q_values_over_time.append(q.copy())
        
        
# Plotting results
plt.figure(figsize=(10, 6))
for i in range(grid_size):
    for j in range(grid_size):
        for action in range(num_actions):
            values = [q_t[i, j, action] for q_t in q_values_over_time]  # Plot the first Q-value
            plt.plot(values, label=f"Q({i},{j},{actions[action]})")
plt.xlabel("Episodes")
plt.ylabel("Q-value")
plt.title("Convergence of Q-values in MC $\epsilon$-Greedy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
        
    
    
    