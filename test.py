from DP.row_optimizer import row_wise_optimizer
import numpy as np

# Define p_theta(x) and p_theta_dot(x)
def binomial_p_theta(x, n, theta):
    from scipy.special import comb
    return comb(n, x) * (theta ** x) * ((1 - theta) ** (n - x))

def binomial_p_theta_dot(x, n, theta):
    from scipy.special import comb
    # Derivative with respect to theta
    return comb(n, x) * (x / theta - (n - x) / (1 - theta)) * (theta ** x) * ((1 - theta) ** (n - x))

# Parameters
n_trials = 3
theta = 0.5
epsilon = 1.0

# Compute p_theta and p_theta_dot
p_theta = np.array([binomial_p_theta(x, n_trials, theta) for x in range(n_trials + 1)])
p_theta_dot = np.array([binomial_p_theta_dot(x, n_trials, theta) for x in range(n_trials + 1)])

# Run the optimizer
result = row_wise_optimizer(p_theta, p_theta_dot, theta, epsilon, n_trials)

# Output results
P_opt = result['P_matrix']
status = result['status']
history = result['history']

print(status)
print("Optimized Privacy Matrix P:")
print(P_opt)
