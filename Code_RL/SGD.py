
import numpy as np
import matplotlib.pyplot as plt
'''
RM

# True expected value of X (E[X])
E_X = 5
alpha_k = 0.1
w_k = 0
iterations = 100
w_history = [w_k]
# Iterative update
for k in range(iterations):
    x_k = np.random.normal(loc=E_X, scale=1)  # Sample x_k from N(E[X], 1)
    w_k = w_k - alpha_k * (w_k - x_k)  # Update rule
    w_history.append(w_k)
# Print final result and plot history
print(f"Final value of w_k after {iterations} iterations: {w_k}")

# Optional: Plotting
import matplotlib.pyplot as plt

plt.plot(w_history, label="w_k")
plt.xlabel("Iteration")
plt.ylabel("w_k")
plt.title("Convergence of w_k")
plt.legend()
plt.show()
'''

'''
sample method to compute J value

E_X = 5  
std_X = 1  
num_samples = 10000  
def J(w, samples):
    return np.mean((w - samples)**2)

def grad_J(w, samples):
    return np.mean(2 * (w - samples)) 
# Calculate Optimal value by sampling
X_samples = np.random.normal(loc=E_X, scale=std_X, size=num_samples) # sample
w_values = np.linspace(0, 10, 100)  # sample
J_values = [J(w, X_samples) for w in w_values]
optimal_w = w_values[np.argmin(J_values)]
print(f"Optimal w: {optimal_w:.4f}")
import matplotlib.pyplot as plt
plt.plot(w_values, J_values, label="J(w)")
plt.axvline(optimal_w, color='r', linestyle='--', label=f"Optimal w = {optimal_w:.4f}")
plt.xlabel("w")
plt.ylabel("J(w)")
plt.title("Minimizing J(w)")
plt.legend()
plt.show()
'''

'''
gradient descent algorithm
using all samples to update the variable update

E_X = 5  
std_X = 1  
num_samples = 10000  
alpha_k = 0.1  # Learning rate
num_iterations = 100  # Number of SGD iterations


def J(w, samples):
    return np.mean((w - samples)**2)

def grad_J(w, samples):
    return np.mean(2 * (w - samples)) 

X_samples = np.random.normal(loc = E_X, scale=std_X, size=num_samples)

w_k = 0
w_history = [w_k]

for k in range(num_iterations):
    # x_k = np.random.choice(X_samples)
    gradient = grad_J(w_k, X_samples)
    w_k = w_k - alpha_k * gradient
    w_history.append(w_k)
# Print final result
print(f"Final value of w_k after {num_iterations} iterations: {w_k:.4f}")

# Plot J(w) and its gradient descent path
w_values = np.linspace(0, 10, 100)  # Range of w values for plotting J(w)
J_values = [J(w, X_samples) for w in w_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_values, J_values, label="J(w)", color='blue')
plt.scatter(w_history, [J(w, X_samples) for w in w_history], color='red', label="SGD Path", zorder=5)
plt.xlabel("w")
plt.ylabel("J(w)")
plt.title("Stochastic Gradient Descent for Minimizing J(w)")
plt.legend()
plt.grid(True)
plt.show()
'''


'''
stochastic gradient descent algorithm
update using variable with just one sample 

E_X = 5  
std_X = 1  
num_samples = 10000  
alpha_k = 0.1  # Learning rate
num_iterations = 100  # Number of SGD iterations


def J(w, samples):
    return np.mean((w - samples)**2)

def grad_J(w, samples):
    return np.mean(2 * (w - samples)) 

X_samples = np.random.normal(loc = E_X, scale=std_X, size=num_samples)

w_k = 0
w_history = [w_k]

for k in range(num_iterations):
    x_k = np.random.choice(X_samples)
    gradient = grad_J(w_k, x_k)
    w_k = w_k - alpha_k * gradient
    w_history.append(w_k)
# Print final result
print(f"Final value of w_k after {num_iterations} iterations: {w_k:.4f}")

# Plot J(w) and its gradient descent path
w_values = np.linspace(0, 10, 100)  # Range of w values for plotting J(w)
J_values = [J(w, X_samples) for w in w_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_values, J_values, label="J(w)", color='blue')
plt.scatter(w_history, [J(w, X_samples) for w in w_history], color='red', label="SGD Path", zorder=5)
plt.xlabel("w")
plt.ylabel("J(w)")
plt.title("Stochastic Gradient Descent for Minimizing J(w)")
plt.legend()
plt.grid(True)
plt.show()
'''

'''
Batch gradient descent (BGD)
Stochastic Gradient Descent (SGD)
mini-batch gradient descent (MBGD)
'''
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)  # Features
y = 4 + 3 * X + np.random.randn(n_samples, 1)  # Target with noise

# Add bias term to X (for intercept calculation)
X_b = np.c_[np.ones((n_samples, 1)), X]  # Add a column of ones for the bias term

# Parameters
alpha = 0.01  # Learning rate
n_iterations = 1000
m = 20  # Mini-batch size for MBGD

# Batch Gradient Descent
def batch_gradient_descent(X, y, alpha, n_iterations):
    w = np.random.randn(X.shape[1], 1)  # Random initialization (2x1 for bias + slope)
    for _ in range(n_iterations):
        gradients = -2 / len(X) * X.T.dot(y - X.dot(w))
        w -= alpha * gradients
    return w

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, alpha, n_iterations):
    w = np.random.randn(X.shape[1], 1)  # Random initialization
    for _ in range(n_iterations):
        for i in range(len(X)):
            random_index = np.random.randint(len(X))
            xi = X[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradients = -2 * xi.T.dot(yi - xi.dot(w))
            w -= alpha * gradients
    return w

# Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, y, alpha, n_iterations, m):
    w = np.random.randn(X.shape[1], 1)  # Random initialization
    for _ in range(n_iterations):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, len(X), m):
            X_mini_batch = X_shuffled[i:i + m]
            y_mini_batch = y_shuffled[i:i + m]
            gradients = -2 / m * X_mini_batch.T.dot(y_mini_batch - X_mini_batch.dot(w))
            w -= alpha * gradients
    return w

# Run the algorithms
w_batch = batch_gradient_descent(X_b, y, alpha, n_iterations)
w_sgd = stochastic_gradient_descent(X_b, y, alpha, n_iterations)
w_mbgd = mini_batch_gradient_descent(X_b, y, alpha, n_iterations, m)

print(f"Batch Gradient Descent Weights: {w_batch.flatten()}")
print(f"Stochastic Gradient Descent Weights: {w_sgd.flatten()}")
print(f"Mini-Batch Gradient Descent Weights: {w_mbgd.flatten()}")

# Visualize results
plt.scatter(X, y, label="Data", alpha=0.5)
x_line = np.linspace(0, 2, 100).reshape(-1, 1)
x_line_b = np.c_[np.ones((100, 1)), x_line]

# Plot predictions
plt.plot(x_line, x_line_b.dot(w_batch), "r-", label="Batch GD")
plt.plot(x_line, x_line_b.dot(w_sgd), "g--", label="SGD")
plt.plot(x_line, x_line_b.dot(w_mbgd), "b:", label="Mini-Batch GD")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Gradient Descent Methods")
plt.legend()
plt.show()
