import numpy as np

# Simplified Q-function for state-action pairs
def q_mu(s, a, theta):
    """Action-value function q_mu(s, a) as a function of state, action, and parameters theta."""
    return theta[0] * s[0] + theta[1] * s[1] + (theta[2] * a)

# Deterministic policy: action is determined by state
def mu(s, theta):
    """Deterministic policy mu(s, theta) selects an action based on state and theta."""
    return 0 if s[0] > s[1] else 1  # Example: action 0 (left) if x > y, else action 1 (right)

# Gradients of q_mu and mu
def grad_q_mu(s, a, theta):
    """Gradient of the Q-function with respect to theta."""
    grad = np.zeros_like(theta)
    grad[0] = s[0]  # Gradient w.r.t theta_0
    grad[1] = s[1]  # Gradient w.r.t theta_1
    grad[2] = a     # Gradient w.r.t theta_2
    return grad

def grad_mu(s, theta):
    """Gradient of the policy mu with respect to theta."""
    grad = np.zeros_like(theta)
    grad[0] = 0.1 if s[0] > s[1] else -0.1  # Simplified gradient based on state comparison
    return grad

# Now calculate the gradient of v_mu(s) with respect to theta
def grad_v_mu(s, theta):
    """Compute the gradient of v_mu(s) with respect to theta."""
    # Get the action chosen by the policy
    a = mu(s, theta)
    
    # First term: gradient of Q with respect to theta
    grad_q = grad_q_mu(s, a, theta)
    
    # Second term: gradient of policy with respect to theta
    grad_policy = grad_mu(s, theta)
    
    # Compute the gradient of v_mu(s)
    gradient_v_mu = grad_q + grad_policy * grad_q[2]  # We multiply by grad_q[2] because of the dependence on action
    
    return gradient_v_mu

# Example usage
s = np.array([2, 3])  # Example state (x, y)
theta = np.array([1.0, 0.5, -0.3])  # Example parameters for the policy

# Compute the gradient of v_mu(s) with respect to theta
grad_v_mu_value = grad_v_mu(s, theta)
print(f"Gradient of v_mu(s) with respect to theta: {grad_v_mu_value}")
