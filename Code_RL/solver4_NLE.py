# Fixed-point iteration for solving x^2 - 4x + 3 = 0
def fixed_point_iteration(initial_guess, tolerance, max_iterations):
    def g(x):
        return (x**2 + 3) / 4  # Iterative function

    x = initial_guess
    for i in range(max_iterations):
        x_next = g(x)
        print(f"Iteration {i + 1}: x = {x_next}")
        # Check for convergence
        if abs(x_next - x) < tolerance:
            print("Converged!")
            return x_next
        x = x_next
    print("Max iterations reached without convergence.")
    return x

# Parameters
initial_guess = 2.5  # Starting point for iteration
tolerance = 1e-6     # Convergence criterion
max_iterations = 50

# Solve
solution = fixed_point_iteration(initial_guess, tolerance, max_iterations)
print(f"Solution: x = {solution}")
