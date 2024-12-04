# import numpy as np
# import matplotlib.pyplot as plt
# '''
# find root
# '''
# # Define the objective function J(w)
# def J(w):
#     return w**2 - 4*w + 4

# # Define the gradient (first derivative) of the objective function g(w)
# def g(w):
#     return 2*w - 4

# # Gradient Descent to find the root of g(w) = 0
# def gradient_descent(learning_rate=0.1, initial_guess=0.0, tolerance=1e-6, max_iter=1000):
#     w = initial_guess
#     for i in range(max_iter):
#         grad = g(w)  # Compute the gradient at w
#         w = w - learning_rate * grad  # Update w using gradient descent
        
#         # Stop if the gradient is small enough (indicating we are near a minimum)
#         if np.abs(grad) < tolerance:
#             break
    
#     return w

# # Find the root using gradient descent
# root = gradient_descent(learning_rate=0.1, initial_guess=0.0)

# print(f"The root of the equation g(w) = 0 is approximately: {root}")

# # Plotting the objective function and the gradient descent path
# w_vals = np.linspace(-2, 6, 400)
# J_vals = J(w_vals)

# plt.figure(figsize=(8, 6))
# plt.plot(w_vals, J_vals, label="Objective Function J(w)", color='b')
# plt.scatter(root, J(root), color='r', zorder=5, label=f"Root at w = {root:.4f}")
# plt.xlabel('w')
# plt.ylabel('J(w)')
# plt.title('Objective Function and Gradient Descent')
# plt.legend()
# plt.grid(True)
# plt.show()
'''
SGD
'''
# import numpy as np
# import matplotlib.pyplot as plt

# # True function g(w) = w^3 - 5
# def g(w):
#     return w**3 - 5

# # Noisy observation of g(w), including Gaussian noise (eta ~ N(0,1))
# def noisy_g(w, noise_std=1.0):
#     noise = np.random.normal(0, noise_std)  # Gaussian noise with zero mean and std=1
#     return g(w) + noise

# # Robbins-Monro algorithm to estimate the root of g(w) = 0
# def robbins_monro(learning_rate_func, initial_guess, max_iter=100, noise_std=1.0, max_w=10.0):
#     w = initial_guess
#     estimates = [w]
    
#     for k in range(1, max_iter + 1):
#         # Noisy observation of the function at current estimate w_k
#         noisy_value = noisy_g(w, noise_std)
        
#         # Update the estimate using the Robbins-Monro formula
#         a_k = learning_rate_func(k)  # Learning rate at iteration k
#         w = w - a_k * noisy_value
        
#         # Apply a maximum value cap to prevent overflow
#         w = np.clip(w, -max_w, max_w)
        
#         # Store the estimate
#         estimates.append(w)
    
#     return np.array(estimates)

# # Define the learning rate function (step size a_k = 1 / k)
# def learning_rate(k):
#     return 1 / k

# # Run the Robbins-Monro algorithm
# initial_guess = 1.0  # Starting with a better initial guess
# max_iter = 100  # Maximum number of iterations
# estimates = robbins_monro(learning_rate, initial_guess, max_iter)

# # Plotting the results
# plt.figure(figsize=(8, 6))
# plt.plot(estimates, label="Estimate of w", color='b')
# plt.axhline(np.cbrt(5), color='r', linestyle='--', label=f"True root (w = {np.cbrt(5):.2f})")
# plt.title("Robbins-Monro Algorithm for Root Finding of $w^3 - 5 = 0$")
# plt.xlabel("Iteration (k)")
# plt.ylabel("Estimated value of w")
# plt.legend()
# plt.grid(True)
# plt.show()



'''

'''
import numpy as np
import matplotlib.pyplot as plt

# Define the function g(w) = w - E[X]
def g(w, E_X):
    return w - E_X

# Define noisy observation of g(w), where eta = E[X] - x
def noisy_observation(w, x, E_X):
    eta = E_X - x  # noise term
    return g(w, E_X) + eta  # noisy version of g(w)

# Simulate the process
np.random.seed(42)

# True expected value of X (E[X])
E_X = 5

# Generate a noisy sample from X (e.g., x ~ N(E[X], 1))
x = np.random.normal(E_X, 1)

# Guess w (we are trying to find the root, where g(w) = 0)
w = 6

# Compute g(w) and noisy observation
g_w = g(w, E_X)
noisy_g_w = noisy_observation(w, x, E_X)

# Print results
print(f"True expected value of X: {E_X}")
print(f"Sample x: {x}")
print(f"g(w) = {g_w}")
print(f"Noisy observation of g(w): {noisy_g_w}")

# Plot to visualize the difference
w_vals = np.linspace(0, 10, 100)
g_vals = g(w_vals, E_X)

plt.plot(w_vals, g_vals, label='g(w) = w - E[X]')
plt.axhline(0, color='black', linewidth=1)
plt.scatter(w, noisy_g_w, color='red', label=f"Noisy g(w) at w={w}")
plt.xlabel('w')
plt.ylabel('g(w)')
plt.title('Root-finding for g(w) = 0 with Noisy Observations')
plt.legend()
plt.grid(True)
plt.show()
