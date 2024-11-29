﻿from scipy.optimize import linprog
pi_a1s = 0.2
pi_a2s = 0.3
pi_a3s = 0.5
# pi doesn't influent the q value
q_sa1 = 1
q_sa2 = 2
q_sa3 = 3
vs = pi_a1s * q_sa1 + pi_a2s * q_sa2 + pi_a3s * q_sa3
print("######")
print("The policy value for state s: ", vs)

exit()

# Given q values (replace these with your actual values)
q1, q2, q3 = 5, 8, 10  # Example values, where q3 >= q1, q2

# Coefficients for the objective function (negative because linprog minimizes by default)
c = [-q1, -q2, -q3]

# Coefficients for the equality constraint (c1 + c2 + c3 = 1)
A_eq = [[1, 1, 1]]
b_eq = [1]

# Bounds for each variable (c1, c2, c3 >= 0)
x_bounds = [(0, None), (0, None), (0, None)]

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

# Display the results
if result.success:
    print("Optimal values for c1, c2, c3:")
    print(f"c1* = {result.x[0]}")
    print(f"c2* = {result.x[1]}")
    print(f"c3* = {result.x[2]}")
    print(f"Maximum value of the objective function: {-result.fun}")
else:
    print("No solution found.")
