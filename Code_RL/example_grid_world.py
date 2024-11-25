import sys
import os
from grid_world import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    env.start_state = (2, 3)
    state = env.reset()               
    for t in range(1):
        env.render()
        action = random.choice(env.action_space)
        action = (0,1)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        # if done:
        #     break
    
    # Add policy
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    env.add_policy(policy_matrix)

    
    # Add state values
    values = np.random.uniform(0,10,(env.num_states,))
    print("values")
    print(values)
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=2)