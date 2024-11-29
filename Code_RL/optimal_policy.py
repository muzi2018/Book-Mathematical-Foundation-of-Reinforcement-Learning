def derive_optimal_policy(states, actions, transition_prob, rewards, v, gamma):
    """
    Derive the optimal policy given the value function v(s).

    Parameters:
    - states: List of states.
    - actions: List of actions.
    - transition_prob: Function P(s'|s, a), transition probabilities.
    - rewards: Function R(s, a, s'), rewards for transitions.
    - v: Value function as a dictionary {state: value}.
    - gamma: Discount factor.

    Returns:
    - policy: Optimal policy as a dictionary {state: best_action}.
    """
    policy = {}
    
    for s in states:
        best_action = None
        max_value = float('-inf')
        
        for a in actions:
            q_sa = 0  # Compute Q(s, a)
            for s_prime in states:
                q_sa += transition_prob(s, a, s_prime) * (
                    rewards(s, a, s_prime) + gamma * v[s_prime]
                )
            
            if q_sa > max_value:
                max_value = q_sa
                best_action = a
        
        policy[s] = best_action
    
    return policy


# Example usage:
states = ["s1", "s2", "s3"]
actions = ["a1", "a2"]
transition_prob = lambda s, a, s_prime: 1.0 if (s, a, s_prime) in [("s1", "a1", "s2"), ("s2", "a2", "s3")] else 0.0
rewards = lambda s, a, s_prime: 1.0 if (s, a, s_prime) in [("s1", "a1", "s2"), ("s2", "a2", "s3")] else 0.0
v = {"s1": 0.5, "s2": 1.0, "s3": 2.0}  # Example value function
gamma = 0.9

policy = derive_optimal_policy(states, actions, transition_prob, rewards, v, gamma)
print("Optimal Policy:", policy)
