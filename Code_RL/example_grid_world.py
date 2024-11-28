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
            'down': 0.0,  # Probability of taking action 'down' in state s1
            'left': 0.0,  # Probability of taking action 'left' in state s1
            'right': 1.0, # Probability of taking action 'right' in state s1
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

    gamma_ = 0.9


    # Solves the Bellman equations using a system of linear equations
    ## Coefficients matrix for the system of equations
    A = np.array([
        [1, -gamma_, 0, 0],   # v_pi(s1) = -1 + gamma * v_pi(s2)
        [0, 1, 0, -gamma_],   # v_pi(s2) = +1 + gamma * v_pi(s4)
        [0, 0, 1, -gamma_],   # v_pi(s3) = +1 + gamma * v_pi(s4)
        [0, 0, 0, 1 - gamma_] # v_pi(s4) = +1 + gamma * v_pi(s4)
    ])

    # Right-hand side vector
    b = np.array([
        -1,  # Constant term for v_pi(s1)
        1,   # Constant term for v_pi(s2)
        1,   # Constant term for v_pi(s3)
        1    # Constant term for v_pi(s4)
    ])

    # Solve the system of equations
    v_pi = np.linalg.solve(A, b)

    # Display the results
    for i, value in enumerate(v_pi, 1):
        print(f"v_pi(s{i}) = {value:.2f}")


    # Known state values (v_pi) from previous calculations
    v_pis = {
        's1': v_pi[0],
        's2': v_pi[1],
        's3': v_pi[2],
        's4': v_pi[3],
    }

    # Calculate q_pi(s, a) values based on the given equations
    q_pi = {
        'q_pi(s1, a1)': -1 + gamma_ * v_pis['s1'],  # q_pi(s1, a1) = -1 + gamma * v_pi(s1)
        'q_pi(s1, a2)': -1 + gamma_ * v_pis['s2'],  # q_pi(s1, a2) = -1 + gamma * v_pi(s2)
        'q_pi(s1, a3)': 0  + gamma_ * v_pis['s3'],   # q_pi(s1, a3) = 0 + gamma * v_pi(s3)
        'q_pi(s1, a4)': -1 + gamma_ * v_pis['s1'],  # q_pi(s1, a4) = -1 + gamma * v_pi(s1)
        'q_pi(s1, a5)': 0  + gamma_ * v_pis['s1'],   # q_pi(s1, a5) = 0 + gamma * v_pi(s1)
    }

    # Print the calculated q_pi(s, a) values
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
    
    