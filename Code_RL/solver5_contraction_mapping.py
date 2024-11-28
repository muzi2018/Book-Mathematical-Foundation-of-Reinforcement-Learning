import numpy as np
import matplotlib.pyplot as plt

# Define the contraction mapping function f(x) = 0.5x
def f(x):
    return 0.5 * x

# Initialize the sequence with an initial guess
x_0 = 10  # Starting point
iterations = 50  # Number of iterations

# List to store the sequence
sequence = [x_0]

# Generate the sequence using the iterative process x_{k+1} = f(x_k)
for k in range(1, iterations + 1):
    x_next = f(sequence[-1])
    sequence.append(x_next)

# Plot the sequence
plt.plot(range(iterations + 1), sequence, marker='o', linestyle='-', label='x_k')
plt.axhline(0, color='red', linestyle='--', label='Fixed point x* = 0')
plt.xlabel('Iteration k')
plt.ylabel('x_k')
plt.title('Convergence of x_k to the Fixed Point x*')
plt.legend()
plt.grid()
plt.show()

# Check the convergence rate between successive points
convergence_rates = [abs(sequence[k] - sequence[k - 1]) for k in range(1, len(sequence))]

# Print convergence rates
print("Convergence rates between successive points:")
for k, rate in enumerate(convergence_rates, start=1):
    print(f"Iteration {k}: |x_{k+1} - x_{k}| = {rate:.6f}")

# Theoretical bound on convergence
gamma = 0.5  # Contraction constant
bound = (gamma**(iterations - 1)) * abs(x_0)

print(f"The theoretical bound on the error after {iterations} iterations: {bound:.6f}")

exit()
import numpy as np
import math
# Define the function f(x) as a contraction mapping
def f(x):
    return 0.5 * math.sin(x)


# Function to perform the iterative process to find the fixed point
def contraction_mapping_iteration(x0, tolerance=1e-6, max_iterations=100):
    x = x0
    for k in range(max_iterations):
        x_new = f(x)
        print(f"Iteration {k+1}: x = {x_new}")
        
        # Check for convergence based on the tolerance
        if abs(x_new - x) < tolerance:
            print(f"Converged to fixed point x* = {x_new} after {k+1} iterations")
            return x_new
        
        x = x_new

    print("Maximum iterations reached without convergence")
    return x

# Initial guess
x0 = 10  # Initial guess for the fixed point

# Perform the iterative process
fixed_point = contraction_mapping_iteration(x0)

# Print the final result
print(f"Final solution: x* = {fixed_point}")
