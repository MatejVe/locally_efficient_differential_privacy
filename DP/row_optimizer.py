import numpy as np
from scipy.optimize import minimize

from DP.utils import print_matrix


def row_optimization(p_theta, p_theta_dot, theta, epsilon, P, x, tol=1e-6):
    """
    Optimize row x of the privacy matrix P.

    Parameters:
    - p_theta: Original data distribution p_theta(x).
    - p_theta_dot: Derivative of p_theta with respect to theta.
    - theta: Parameter theta.
    - epsilon: Privacy parameter.
    - P: Current privacy matrix of shape (n_x, n_y).
    - x: Index of the row to optimize.
    - tol: Tolerance for the optimization solver.

    Returns:
    - P_new: Updated privacy matrix after optimizing row x.
    """

    n_x, n_y = P.shape

    # Fixed rows
    P_fixed = np.delete(P, x, axis=0)

    # Variables: P_x = P[x, :]
    P_x_initial = P[x, :].copy()

    # Objective function: Negative Fisher information (since we minimize)
    def objective(P_x):
        # Reconstruct the full P matrix
        P_full = P.copy()
        P_full[x, :] = P_x

        # Compute P(Y = y; theta)
        P_Y_theta = np.dot(P_full.T, p_theta)

        # Compute derivative of P(Y = y; theta) with respect to theta
        dP_Y_theta = np.dot(P_full.T, p_theta_dot)

        # Compute Fisher information
        fisher_info = np.sum((dP_Y_theta**2) / P_Y_theta)

        return -fisher_info  # Negative for minimization

    # Constraints
    constraints = []

    # Privacy constraints for row x
    for y in range(n_y):
        for y_prime in range(n_y):
            if y != y_prime:
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda P_x, y=y, y_prime=y_prime: P_x[y]
                        - np.exp(epsilon) * P_x[y_prime],
                    }
                )

    # Non-negativity
    bounds = [(0, None) for _ in range(n_y)]

    # Column sum adjustments
    # After updating P_x, we need to adjust P_fixed to maintain column sums
    # We can include equality constraints to ensure column sums remain 1

    # Optimize
    result = minimize(
        objective,
        P_x_initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "disp": False},
    )

    if not result.success:
        print(f"Optimization failed for row {x}: {result.message}")
        return P  # Return original P if optimization fails

    # Update P
    P_new = P.copy()
    P_new[x, :] = result.x

    return P_new


def row_wise_optimizer(
    p_theta, p_theta_dot, theta, epsilon, n_trials, tol=1e-6, max_iter=100
):
    """
    Optimize the privacy matrix P in a row-by-row manner.

    Parameters:
    - p_theta: Original data distribution p_theta(x).
    - p_theta_dot: Derivative of p_theta with respect to theta.
    - theta: Parameter theta.
    - epsilon: Privacy parameter.
    - n_trials: Number of trials (determines the size of P).
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - P_opt: Optimized privacy matrix.
    - status: Convergence status.
    - history: List of privacy matrices over iterations.
    """

    n_x = n_trials + 1  # Assuming counts from 0 to n_trials
    n_y = n_x  # Assuming output space is the same as input space

    # Initialize P with uniform probabilities satisfying column sums
    P = np.ones((n_x, n_y)) / n_x

    # Ensure P satisfies the privacy constraints
    # (Additional steps may be required)

    history = [P.copy()]

    for iteration in range(max_iter):
        P_old = P.copy()

        for x in range(n_x - 1):
            P = row_optimization(p_theta, p_theta_dot, theta, epsilon, P, x, tol)
            print("==================")
            print_matrix(P)
            print("==================")

        P = P / np.sum(P, axis=0)

        # Convergence check
        if np.linalg.norm(P - P_old, ord="fro") < tol:
            status = f"Converged after {iteration+1} iterations."
            break

        history.append(P.copy())
    else:
        status = "Max iterations reached without convergence"

    return {"P_matrix": P, "status": status, "history": history}
