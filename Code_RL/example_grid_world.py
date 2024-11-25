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
    # 2.1 Movivating example 1: Why are returns important
    ## firt policy 
    # up_ = (-1, 0)
    # down_ = (1, 0)
    # left_ = (0, -1)
    # right_ = (0, 1)
    # stay_ = (0, 0)
    # action = random.choice(env.action_space)
    # next_state, reward, done, info= env.step(action)
    
    # 2.2 Motivating example 2: How to calculate returns?
    # 2.3 State values
    ## When both the policy and the system model are deterministic, starting from a state always leads to the same trjectory. In this case, the return obtained starting from a state is equal to the value of that state.
    ## By contrast, when either the policy or the system model is stochastic, starting from the same state may generate different trjectories. In this case, the returns of different trajectories are different, and the state value is the maen of these returns.
    
    # 2.4 Bellman equation
    # G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... discounted return along a trajectory
    
    # 2.5 Examples for illustrating the Bellman equation
    env = GridWorld()
    ## aciton
    (up_, down_, left_, right_, stay_) = (env.action_space[3], env.action_space[1], env.action_space[2], env.action_space[0], env.action_space[4])

    # ## state
    (s1, s2, s3, s4) = ((0, 0), (1, 0), (0, 1), (1, 1))
    # ## env, row->x, column->y
    env.env_size = (2,2)
    env.num_states = 4
    env.start_state = (1,0)
    env.forbidden_states = [(1,0)]
    env.target_state = (1,1)
    env.reset()
    ## Policy    
    policy = [
        (s1, down_),
        (s2, down_),
        (s3, right_),
        (s4, stay_)
    ]
    for t in range(4):
        env.render()
        for state, action in policy:
            if env.agent_state == state:
                next_state, reward, done, info = env.step(action)
                print(f"Step: {t}, Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")
    gamma_ = 0.1
    env.render(animation_interval=2) 
    print("Grid_world")