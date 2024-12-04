import numpy as np
from itertools import permutations

# Function to calculate the total distance of a route
def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

# Non-Incremental: Calculate TSP solution from scratch
def non_incremental_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    all_routes = permutations(range(num_cities))  # Generate all possible routes
    min_distance = float('inf')
    best_route = None

    for route in all_routes:
        dist = total_distance(route, distance_matrix)
        if dist < min_distance:
            min_distance = dist
            best_route = route
    return min_distance, best_route
# Incremental: Build the TSP solution step by step
def incremental_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    route = [0]  # Start with the first city
    visited = set(route)

    while len(route) < num_cities:
        last_city = route[-1]
        next_city = min(
            (i for i in range(num_cities) if i not in visited),
            key=lambda i: distance_matrix[last_city, i]
        )
        route.append(next_city)
        visited.add(next_city)

    # Compute the total distance of the incremental route
    return total_distance(route, distance_matrix), route
# Generate a random distance matrix for 50 cities
np.random.seed(42)
num_cities = 8
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)  # Distance from a city to itself is 0

# Time the non-incremental and incremental methods
import time

# Non-Incremental Method
start_time = time.time()
non_inc_distance, non_inc_route = non_incremental_tsp(distance_matrix)
non_inc_time = time.time() - start_time

# Incremental Method
start_time = time.time()
inc_distance, inc_route = incremental_tsp(distance_matrix)
inc_time = time.time() - start_time

print(f"Non-Incremental Time: {non_inc_time:.6f} seconds")
print(f"Incremental Time: {inc_time:.6f} seconds")
