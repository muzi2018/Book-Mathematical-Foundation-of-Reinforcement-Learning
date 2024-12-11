import sys
import os
from grid_world import GridWorld
import random
import numpy as np
import random
# 3 - Chapter 2 State Values and Bellman Equation
# 1. state value: which is defined as the average reward that an agent can obtain if it follows a given policy
# 2. Bellman equation which is an important tool for analyzing state values, in a nutshell, Bellman equation describes the relationships between the values of all states. By solving the Bellman equation , we can obtain the state values
# 3. policy evaluation: By solving the Bellman equation to obtain the state values.
# 4. action value

#
if __name__ == "__main__":      
    policy = {
        's1': {'up': 0.0,'down': 0.0,   'left': 0.0, 'right': 1.0, 'stay': 0.0},
        's2': {'up': 0.0,'down': 1.0,   'left': 0.0, 'right': 0.0, 'stay': 0.0},
        's3': {'up': 0.0,'down': 0.0,   'left': 0.0, 'right': 1.0, 'stay': 0.0},
        's4': {'up': 0.0,'down': 0.0,   'left': 0.0, 'right': 0.0, 'stay': 1.0}}
    G_t = 0
    G_t_total = 0
    gamma_ = 0.9
    # 
    A = np.array([
        [1, -gamma_, 0, 0],   
        [0, 1, 0, -gamma_],   
        [0, 0, 1, -gamma_],   
        [0, 0, 0, 1 - gamma_] 
        ])
    b = np.array([-1,  1,   1,   1 ])
    v_pi = np.linalg.solve(A, b)
    
    for i, value in enumerate(v_pi, 1):
        print(f"v_pi(s{i}) = {value:.2f}")
    v_pis = {
        's1': v_pi[0],
        's2': v_pi[1],
        's3': v_pi[2],
        's4': v_pi[3],
    }
    q_pi = {
        'q_pi(s1, a1)': -1 + gamma_ * v_pis['s1'],  # q_pi(s1, a1) = -1 + gamma * v_pi(s1)
        'q_pi(s1, a2)': -1 + gamma_ * v_pis['s2'],  # q_pi(s1, a2) = -1 + gamma * v_pi(s2)
        'q_pi(s1, a3)': 0  + gamma_ * v_pis['s3'],   # q_pi(s1, a3) = 0 + gamma * v_pi(s3)
        'q_pi(s1, a4)': -1 + gamma_ * v_pis['s1'],  # q_pi(s1, a4) = -1 + gamma * v_pi(s1)
        'q_pi(s1, a5)': 0  + gamma_ * v_pis['s1'],   # q_pi(s1, a5) = 0 + gamma * v_pi(s1)
    }
    for action, value in q_pi.items():
        print(f"{action} = {value:.2f}")

    exit()
    
    
    
    ###
    # 1. without Bellman equation, Use an iterative approach to get the state value
    #    it can only converge after many iterations about the trajectory
    ###
    # Iterative approach to approximate state values
    state_values = {state: 0 for state in states.keys()}  # Initialize state values to 0
    gamma_ = 0.9  # Discount factor
    num_iterations = 100  # Number of iterations for convergence
    num_traj = 100
    for i in range(num_traj):
        env.reset()
        G_t = 0
        for t in range(num_iterations):  # Iteratively update state values
            # env.render()
            for state_name, state_coords in states.items():
                if env.agent_state == state_coords:
                    # Choose action based on the policy's probabilities
                    actions_list = list(policy[state_name].keys())
                    probobilities = list(policy[state_name].values())
                    chosen_action =  random.choices(actions_list, probobilities)[0]
                    action_probability = probobilities[actions_list.index(chosen_action)]
                    next_state, reward, done, info = env.step(actions[chosen_action])
                    # G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... discounted return along a trajectory
                    G_t += reward * (gamma_ ** t)
                    if(next_state == (0,0)):
                        print(f"Step: {t}, State: {next_state}, Action: {chosen_action}, "
                            f"Probability: {action_probability:.4f}, Reward: {reward}, G_t: {G_t:.6f}")
                    break
        print(f"G_t:  {G_t}")
        G_t_total += G_t
        print(f"Trajectory {i + 1} completed. G_t: {G_t:.6f}")               
      # Average state value
    avg_state_value = G_t_total / num_traj
    print(f"Average State Value: {avg_state_value:.6f}")
    # env.render(animation_interval=7) 
    
    # ###
    # # 2. with Bellman equation, sometimes we could get the analytical solution, it is easier to get 
    # #    the state value.
    # ###
    # # for s1
    # G_t = gamma_ / (1 - gamma_)
    
    