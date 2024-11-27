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
    # 2.5 Examples for illustrating the Bellman equation
    env = GridWorld()
    ## aciton 
    ### up_ = (0,-1); down_ = (0, 1); left_ = (-1, 0); right_ = (1, 0)
    actions = {
        'up': env.action_space[2],    # Action for moving up
        'down': env.action_space[0],   # Action for moving down
        'left': env.action_space[3],   # Action for moving left
        'right': env.action_space[1],  # Action for moving right
        'stay': env.action_space[4]     # Action for staying in place
    }

    ## state
    states = {
        's1': (0, 0),  # State s1 at coordinates (0, 0)
        's2': (1, 0),  # State s2 at coordinates (1, 0)
        's3': (0, 1),  # State s3 at coordinates (0, 1)
        's4': (1, 1)   # State s4 at coordinates (1, 1)
    }
    ## reward
    env.reward_forbidden = -1
    env.reward_step = 0
    env.reward_target = 1
    # ## env, row->x, column->y
    env.env_size = (2, 2)
    env.num_states = 4
    env.start_state = states['s1']
    env.forbidden_states = [(1, 0)]
    env.target_state = (1, 1)
    env.reset()
    ## Policy    
    policy = {
        's1': {
            'up': 0.0,    # Probability of taking action 'up' in state s1
            'down': 0.5,  # Probability of taking action 'down' in state s1
            'left': 0.0,  # Probability of taking action 'left' in state s1
            'right': 0.5, # Probability of taking action 'right' in state s1
            'stay': 0.0   # Probability of taking action 'stay' in state s1
        },
        's2': {
            'up': 0.0,    # Define probabilities for state s2 (example values)
            'down': 1.0,
            'left': 0.0,
            'right': 0.0,
            'stay': 0.0   # Example: only staying in state s2
        },
        's3': {
            'up': 0.0,    
            'down': 0.0,
            'left': 0.0,
            'right': 1.0,
            'stay': 0.0   # Example: only staying in state s3
        },
        's4': {
            'up': 0.0,    
            'down': 0.0,
            'left': 0.0,
            'right': 0.0,
            'stay': 1.0   # Example: only staying in state s4
        }
    }
    ## state value
    G_t = 0
    G_t_total = 0

    action_values = {state: {action: 0 for action in actions.keys()} for state in states.keys()}
    gamma_ = 0.9



    ###
    # 1. without Bellman equation, Use an iterative approach to get the state value
    #    it can only converge after many iterations about the trajectory
    ###
    # Iterative approach to approximate state values
    state_values = {state: 0 for state in states.keys()}  # Initialize state values to 0
    gamma_ = 0.9  # Discount factor
    num_iterations = 100  # Number of iterations for convergence
    num_traj = 10
    for i in range(num_traj):
        env.reset()
        for t in range(num_iterations):  # Iteratively update state values
            # env.render()
            for state_name, state_coords in states.items():
                if env.agent_state == state_coords:
                    # Choose action based on the policy's probabilities
                    actions_list = list(policy[state_name].keys())
                    probobilities = list(policy[state_name].values())
                    chosen_action =  random.choices(actions_list, probobilities)[0]
                    action_probability = probobilities[actions_list.index(chosen_action)]
                    # Execute chosen action
                    next_state, reward, done, info = env.step(actions[chosen_action])
                    
                    
                    # Calculate Q(s, a)
                    # Immediate reward + discounted future rewards based on policy
                    next_state_name = None
                    for name, coords in states.items():
                        if coords == next_state:
                            next_state_name = name
                            break
                    if next_state_name:
                        future_value = sum(policy[next_state_name][next_action] * action_values[next_state_name][next_action]
                                        for next_action in actions.keys())
                        action_values[state_name][chosen_action] = reward + gamma_ * future_value
                    break

# Print the action values
print("Action Values (Q):")
for state, action_value in action_values.items():
    print(f"State: {state}")
    for action, value in action_value.items():
        print(f"  Action: {action}, Q(s, a): {value/10.0:.6f}")
    
    