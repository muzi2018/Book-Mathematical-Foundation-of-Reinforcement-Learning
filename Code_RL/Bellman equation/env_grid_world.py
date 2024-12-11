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
    grid_size = 5  # 5x5 grid
    num_episodes = 500
    
    # environment 
    ##action up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    actions = [(0,-1), (0, 1), (-1,0), (1,0)]
    num_actions = len(actions)

    ## reward
    '''
    reward setup:rboundary = -1; rforbidden = -1; rtarget = 1; rstep = 0
    '''
    env.reward_boundary = -1
    env.reward_forbidden = -1
    env.reward_target = 1
    env.reward_step = 0
    
    ## Create grid row->x, column->y
    env.env_size = (grid_size, grid_size)
    env.num_states = grid_size * grid_size
    env.forbidden_states = [(1, 1),(2,1),(2,2),(1,3),(1,4),(3,3)]
    env.target_state = (2, 3)
    
    state_values = np.zeros((grid_size, grid_size))  # A grid for values
    returns = np.zeros((grid_size, grid_size, num_actions))  # Cumulative returns
    
    def generate_episode(policy, state_values, alpha=0.1):
        episode = []
        env.start_state = (np.random.randint(grid_size), np.random.randint(grid_size)) 
        env.reset()
        state = env.start_state  # Start in a random state
        print("start_state' = ", state)
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
    
    
if __name__ == "__main__":
    policy = {
        (i, j): {action: 1 / num_actions for action in actions} 
        for i in range(grid_size) for j in range(grid_size)
    }
    for num_ in range(num_episodes):
        # Generate an episode
        print("episode is ", num_)
        episode, state_values = generate_episode(policy, state_values)


        

        
    
    
    