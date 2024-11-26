import sys
import os
from grid_world import GridWorld
import random
import numpy as np

# 3 - Chapter 2 State Values and Bellman Equation
# 1. state value: which is defined as the average reward that an agent can obtain if it follows a given policy
# 2. Bellman equation which is an important tool for analyzing state values, in a nutshell, Bellman equation describes the relationships between the values of all states. By solving the Bellman equation , we can obtain the state values
# 3. policy evaluation: By solving the Bellman equation to obtain the state values.
# 4. action value

#
if __name__ == "__main__":  
    # 2.4 Bellman equation
    
    # 2.5 Examples for illustrating the Bellman equation
    env = GridWorld()
    ## aciton 
    ### up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    (up_, down_, left_, right_, stay_) = (env.action_space[2], env.action_space[0], env.action_space[3], env.action_space[1], env.action_space[4])

    # ## state
    (s1, s2, s3, s4) = ((0, 0), (1, 0), (0, 1), (1, 1))
    ## reward
    env.reward_forbidden = -1
    env.reward_step = 0
    env.reward_target = 1
    # ## env, row->x, column->y
    env.env_size = (2, 2)
    env.num_states = 4
    env.start_state = s2
    env.forbidden_states = [(1, 0)]
    env.target_state = (1, 1)
    env.reset()
    ## Policy    
    policy = [
        (s1, down_),
        (s2, down_),
        (s3, right_),
        (s4, stay_)
    ]
    ## state value
    G_t = 0
    gamma_ = 0.9
    
    # Bellman equation 
    ## v_{\pi}(s_1) = 0 + \gammar v_{\pi}(s_1)
    ## v_{\pi}(s_2) = 1 + \gammar v_{\pi}(s_4)
    ## v_{\pi}(s_3) = 1 + \gammar v_{\pi}(s_4)
    ## v_{\pi}(s_4) = 1 + \gammar v_{\pi}(s_4)
    


    ###
    # 1. without Bellman equation, Use an iterative approach to get the state value
    #    it can only converge after many iterations.
    ###
    for t in range(200): # converge number > 200
        env.render()
        for state, action in policy:
            if env.agent_state == state:
                next_state, reward, done, info = env.step(action)
                # G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... discounted return along a trajectory
                G_t += reward * (gamma_ ** t)
                print(f"Step: {t}, Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")
                break
    print(f"State value: {G_t}")
    env.render(animation_interval=7) 
    
    ###
    # 2. with Bellman equation, sometimes we could get the analytical solution, it is easier to get 
    #    the state value.
    ###
    # for s1
    G_t = gamma_ / (1 - gamma_)
    
    