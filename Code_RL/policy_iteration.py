import numpy as np

'''
step 1: policy evaluation
'''
step1 = False
step2 = True
if step1:
    # Define parameters
    gamma = 0.9  # Discount factor
    threshold = 1e-4  # Convergence threshold
    states = ['s1', 's2', 's3', 's4']  # State space
    actions = ['a1', 'a2', 'a3', 'a4', 'a5']  # Action space

    # Transition and reward model (simplified for one policy)
    transition_model = {
        's1': {'a3': [('s3', 1, 0)]},
        's2': {'a3': [('s4', 1, 1)]},
        's3': {'a2': [('s4', 1, 1)]},
        's4': {'a5': [('s4', 1, 1)]},
    }

    # Define a fixed policy π
    policy = {
        's1': 'a3',
        's2': 'a3',
        's3': 'a2',
        's4': 'a5',
    }

    # Initialize value function
    v_pi = {state: 0 for state in states}
    iteration =0
    # Policy evaluation: Iterative computation of v_pi
    while True:
        delta = 0
        v_next = v_pi.copy()
        for s in states:
            action = policy[s]
            q_value = 0
            for next_state, prob, reward in transition_model[s][action]:
                q_value += prob * (reward + gamma * v_pi[next_state])
            delta = max(delta, abs(v_next[s] - q_value))
            v_next[s] = q_value
        v_pi = v_next
        iteration += 1
        if delta < threshold:
            break

    print(f"\nConverged after {iteration} iterations.")
    # Print the evaluated state value function
    print("Evaluated State Value Function:")
    for s in states:
        print(f"V({s}) = {v_pi[s]:.4f}")

'''
step 2: policy improvement
'''

if step2: 
    # Define parameters
    gamma = 0.9  # Discount factor
    states = ['s1', 's2', 's3', 's4']  # State space
    actions = ['a1', 'a2', 'a3', 'a4', 'a5']  # Action space

    # Transition and reward model
    transition_model = {
        's1': {
            'a1': [('s1', 1, -1)],
            'a2': [('s2', 1, -1)],
            'a3': [('s3', 1, 0)],
            'a4': [('s1', 1, -1)],
            'a5': [('s1', 1, 0)],
        },
        's2': {
            'a1': [('s2', 1, -1)],
            'a2': [('s2', 1, -1)],
            'a3': [('s4', 1, 1)],
            'a4': [('s1', 1, 0)],
            'a5': [('s2', 1, -1)],
        },
        's3': {
            'a1': [('s1', 1, 0)],
            'a2': [('s4', 1, 1)],
            'a3': [('s3', 1, -1)],
            'a4': [('s3', 1, -1)],
            'a5': [('s3', 1, 0)],
        },
        's4': {
            'a1': [('s2', 1, -1)],
            'a2': [('s4', 1, -1)],
            'a3': [('s4', 1, -1)],
            'a4': [('s3', 1, 0)],
            'a5': [('s4', 1, 1)],
        },
    }

    # Assume v_pi has been calculated
    v_pi = {
        's1': 0,
        's2': 0.9,
        's3': 1.8,
        's4': 1.0,
    }

    # Policy improvement step
    new_policy = {}

    for s in states:
        q_values = {}
        for a in actions:
            q_value = 0
            for next_state, prob, reward in transition_model[s][a]:
                q_value += prob * (reward + gamma * v_pi[next_state])
            q_values[a] = q_value
        
        # Select the action with the maximum Q-value
        best_action = max(q_values, key=q_values.get)
        new_policy[s] = best_action

    # Display the improved policy
    print("Improved Policy:")
    for s in states:
        print(f"π({s}) = {new_policy[s]}")
        
    # Function to calculate the value function for a given policy
    def evaluate_policy(policy, transition_model, gamma, threshold=1e-4):
        v = {s: 0 for s in states}  # Initialize value function
        while True:
            delta = 0
            for s in states:
                action = policy[s]
                new_value = 0
                for next_state, prob, reward in transition_model[s][action]:
                    new_value += prob * (reward + gamma * v[next_state])
                delta = max(delta, abs(v[s] - new_value))
                v[s] = new_value
            if delta < threshold:
                break
        return v

    # Calculate the value function for the new policy
    v_pi_new = evaluate_policy(new_policy, transition_model, gamma)

    # Compare the new policy value with the old policy value
    print("Comparing Policies:")
    improved = False
    for s in states:
        old_value = v_pi[s]
        new_value = v_pi_new[s]
        print(f"State {s}: V_old = {old_value:.4f}, V_new = {new_value:.4f}")
        if new_value > old_value:
            improved = True

    if improved:
        print("\nThe new policy is better than the previous policy!")
    else:
        print("\nThe new policy is not better than the previous policy.")

