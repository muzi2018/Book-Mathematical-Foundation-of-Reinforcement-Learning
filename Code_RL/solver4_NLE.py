# Define the function that represents the equation to solve: x = 2x - 1
def func(x):
    return 2 * x - 1

# Define a simple iterative method to solve the equation
def solve_nonlinear_equation(initial_guess, tolerance=1e-6, max_iterations=1000):
    x = initial_guess
    for i in range(max_iterations):
        new_x = func(x)
        
        # Check if the difference between new_x and x is below the tolerance level
        if abs(new_x - x) < tolerance:
            print(f"Converged to solution x = {new_x} after {i+1} iterations")
            return new_x
        x = new_x
    
    print("Maximum iterations reached without convergence")
    return x

# Initial guess for x
initial_guess = 0.0

# Solve the equation
solution = solve_nonlinear_equation(initial_guess)

# Print the result
print(f"Final solution: x = {solution}")
