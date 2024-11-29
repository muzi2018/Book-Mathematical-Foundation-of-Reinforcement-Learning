import numpy as np

def value_iteration(states, actions, rewards, transition_prob, gamma, epsilon=1e-6):
    """
    Compute value function using the Bellman Optimality Equation.
    
    Parameters:
    - states: List of all states in the state space S.
    - actions: List of all actions in the action space A.
    - rewards: Function rewards(s, a, r) that gives reward probabilities p(r|s, a).
    - transition_prob: Function transition_prob(s, a, s') that gives p(s'|s, a).
    - gamma: Discount factor (0 <= gamma <= 1).
    - epsilon: Convergence threshold for value iteration.

    Returns:
    - v: Value function as a dictionary {state: value}.
    - policy: Optimal policy as a dictionary {state: action probabilities}.
    """
    # Initialize value function for all states
    v = {s: 0 for s in states}
    
    while True:
        delta = 0
        new_v = v.copy()
        
        # Update value function for each state
        for s in states:
            max_value = float('-inf')
            
            # Evaluate actions
            for a in actions:
                q_sa = 0
                
                # Compute Q(s, a)
                for r in rewards:
                    q_sa += rewards(s, a, r) * r
                for s_prime in states:
                    q_sa += gamma * transition_prob(s, a, s_prime) * v[s_prime]
                
                # Maximize over actions
                max_value = max(max_value, q_sa)
            
            # Update the value for state s
            new_v[s] = max_value
            delta = max(delta, abs(new_v[s] - v[s]))
        
        v = new_v
        
        # Check convergence
        if delta < epsilon:
            break
    
    # Derive the optimal policy
    policy = {}
    for s in states:
        action_values = []
        for a in actions:
            q_sa = 0
            for r in rewards:
                q_sa += rewards(s, a, r) * r
            for s_prime in states:
                q_sa += gamma * transition_prob(s, a, s_prime) * v[s_prime]
            action_values.append((a, q_sa))
        
        # Compute optimal action probabilities (greedy policy)
        best_action = max(action_values, key=lambda x: x[1])[0]
        policy[s] = {a: 1.0 if a == best_action else 0.0 for a in actions}
    
    return v, policy


# Example usage:
states = ["s1", "s2", "s3"]
actions = ["a1", "a2"]
rewards = lambda s, a, r: 1.0 if (s, a, r) in [("s1", "a1", 10), ("s2", "a2", 5)] else 0.0
transition_prob = lambda s, a, s_prime: 1.0 if (s, a, s_prime) in [("s1", "a1", "s2"), ("s2", "a2", "s3")] else 0.0
gamma = 0.9

v, policy = value_iteration(states, actions, rewards, transition_prob, gamma)
print("Value Function:", v)
print("Optimal Policy:", policy)
