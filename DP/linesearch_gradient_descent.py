import numpy as np
from utils import fisher_gradient, fisher_information_privatized, is_epsilon_private
import cvxpy as cp


def project_onto_feasible_set(Q, epsilon):
    n_plus_1 = Q.shape[0]
    Q_var = cp.Variable((n_plus_1, n_plus_1))
    Q_param = Q

    # Objective: minimize ||Q_var - Q_param||_F^2
    objective = cp.Minimize(cp.sum_squares(Q_var - Q_param))

    # Constraints
    constraints = []

    # Non-negativity
    constraints += [Q_var >= 0]

    # Column sums
    for j in range(n_plus_1):
        constraints += [cp.sum(Q_var[:, j]) == 1]

    # ε-Differential Privacy constraints
    exp_eps = np.exp(epsilon)
    for i in range(n_plus_1):
        for j in range(n_plus_1):
            if i < n_plus_1 - 1:
                constraints += [Q_var[i, j] - exp_eps * Q_var[i + 1, j] <= 0]
            if i > 0:
                constraints += [Q_var[i, j] - exp_eps * Q_var[i - 1, j] <= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return Q_var.value


def linesearch(p_theta, p_theta_dot, epsilon, q, n, theta):
    max_iters = 100
    alpha_init = 0.1
    beta = 0.8
    c = 1e-4
    history = [q]

    for t in range(max_iters):
        I_current = fisher_information_privatized(q, n, theta)
        grad_I = fisher_gradient(p_theta, p_theta_dot, q)

        alpha = alpha_init
        while True:
            q_new = q + alpha * grad_I
            q_new = project_onto_feasible_set(q_new, epsilon)

            I_new = fisher_information_privatized(q_new, n, theta)

            if I_new >= I_current + c * alpha * np.sum(grad_I + (q_new - q)):
                break
            else:
                alpha *= beta

            if alpha < 1e-8:
                print("Line search failed to find a suitable step size.")

        if np.allclose(q, q_new):
            status = f"Converged after {t+1} iterations."
            break

        if abs(I_current - I_new) < 1e-8:
            status = f"Converged after {t+1} iteratons."
            break

        q = q_new
        history.append(q)
        status = "Max iterations reached without convergence"

    return {"Q_matrix": q, "status": status, "history": history}


if __name__ == "__main__":
    from scipy.stats import binom
    from utils import binom_derivative

    n = 3
    theta = 0.5
    epsilon = 1.0

    p_theta = binom.pmf([0, 1, 2, 3], 3, theta)
    p_theta_dot = [binom_derivative(i, n, theta) for i in range(4)]

    Q_init = np.ones((n + 1, n + 1)) / (n + 1) + np.random.normal(size=(n + 1, n + 1))

    Q_init = project_onto_feasible_set(Q_init, epsilon)

    result = linesearch(p_theta, p_theta_dot, epsilon, Q_init, n, theta)

    print(result["Q_matrix"])
    print(result["status"])
    print(result["history"])
