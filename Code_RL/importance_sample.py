import numpy as np

def importance_sampling(samples, p0, p1):
    """
    Estimate E[X ~ p0] using importance sampling with samples drawn from p1.
    
    Parameters:
    - samples: array-like, samples drawn from distribution p1
    - p0: function, probability density function of p0
    - p1: function, probability density function of p1
    
    Returns:
    - estimated_value: float, the estimated value of E[X ~ p0]
    """
    # Calculate importance weights
    weights = p0(samples) / p1(samples)
    
    # Avoid division by zero
    weights = np.where(p1(samples) == 0, 0, weights)
    
    # Calculate the weighted average
    estimated_value = np.sum(weights * samples) / np.sum(weights)
    
    return estimated_value

# Example usage
if __name__ == "__main__":
    # Define the probability density functions for p0 and p1
    def p0(x):
        # p0 is a uniform distribution over [0, 1]
        return np.where((x >= 0) & (x <= 1), 1.0, 0.0)

    def p1(x):
        # p1 is a normal distribution with mean 0.5 and std 0.1
        return (1 / (0.1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

    # Generate samples from p1
    num_samples = 10000
    samples = np.random.normal(0.5, 0.1, num_samples)

    # Estimate E[X ~ p0]
    estimated_value = importance_sampling(samples, p0, p1)
    print(f"Estimated E[X ~ p0]: {estimated_value:.4f}")
