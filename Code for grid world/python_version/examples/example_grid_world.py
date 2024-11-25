import sys
import os
src_path = os.path.abspath(os.path.join(os.getcwd(), "Code for grid world/python_version/src"))
sys.path.append(src_path)

print("sys.path = ", sys.path)
print(os.getcwd())

# from src.grid_world import GridWorld
from grid_world import GridWorld
import random
import numpy as np

exit()

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()               
    for t in range(1):
        env.render()
        action = random.choice(env.action_space)
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