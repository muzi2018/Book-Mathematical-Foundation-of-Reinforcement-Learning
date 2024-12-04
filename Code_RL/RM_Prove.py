import numpy as np
import matplotlib.pyplot as plt

# Number of time steps and sample paths
num_steps = 10
num_paths = 10
t = np.linspace(0, 1, num_steps)

# Generate a stochastic process X_k(t) = W(t) / sqrt(k), where W(t) is Brownian motion
np.random.seed(42)
X_k = [np.cumsum(np.random.normal(0, 1/np.sqrt(num_steps), num_steps)) / np.sqrt(k) for k in range(1, num_paths + 1)]

# Plot the processes for illustration
plt.figure(figsize=(10, 6))
for k, process in enumerate(X_k, start=1):
    plt.plot(t, process, label=f"X_{k}(t), k={k}")

plt.axhline(0, color="black", linestyle="--", label="Limit")
plt.title("Uniformly Almost Sure Convergence Example")
plt.xlabel("t")
plt.ylabel("X_k(t)")
plt.legend()
plt.grid(True)
plt.show()
