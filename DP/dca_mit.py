import numpy as np
from scipy.optimize import minimize
from DP.utils import fisher_gradient


def indicator_subgradient(Y, epsilon):
    nrows, ncols = Y.shape

    def objective(Q):
        Q = Q.reshape((nrows, ncols))
        return -np.sum(Q * Y)

    constraints = []
    # Privacy constraints
    for i in range(nrows):
        for j in range(ncols):
            for j_prime in range(ncols):
                if j != j_prime:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda Q, i=i, j=j, j_prime=j_prime: Q.reshape(
                                (nrows, ncols)
                            )[i, j]
                            - np.exp(-epsilon) * Q.reshape((nrows, ncols))[i, j_prime],
                        }
                    )
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda Q, i=i, j=j, j_prime=j_prime: np.exp(epsilon)
                            * Q.reshape((nrows, ncols))[i, j_prime]
                            - Q.reshape((nrows, ncols))[i, j],
                        }
                    )
    # Sum-to-1 column-wise constraints
    for j in range(ncols):
        constraints.append(
            {
                "type": "eq",
                "fun": lambda Q, j=j: np.sum(Q.reshape((nrows, ncols))[:, j]) - 1,
            }
        )

    # Bounds for each entry in Q
    bounds = [(0, 1) for _ in range(nrows * ncols)]

    # Initial guess for Q as a flattened array
    Q_initial = np.ones((nrows, ncols)) / ncols + np.random.uniform(
        -0.01, 0.01, (nrows, ncols)
    )
    Q_initial = np.clip(Q_initial, 0, 1)
    Q_initial /= np.sum(Q_initial, axis=0)  # Normalize to sum to 1 column-wise
    Q_initial = Q_initial.flatten()  # Flatten for the optimizer

    result = minimize(objective, Q_initial, constraints=constraints, bounds=bounds)

    return result.x.reshape((nrows, ncols))


class dca_mit:
    name = "DCA MIT solver"

    def __call__(self, p_theta, p_theta_dot, theta, epsilon, n_trials, tol=1e-6, max_iter=100):
        q0 = np.random.uniform(size=n_trials + 1)
        q0 = np.vstack([q0] * (n_trials + 1))
        for i in range(n_trials + 1):
            q0[i] = q0[i] / np.sum(q0[i])

        q = q0
        history = [q]

        for i in range(max_iter):
            y = fisher_gradient(p_theta, p_theta_dot, q)
            q_next = indicator_subgradient(y, epsilon)

            if np.allclose(q, q_next, rtol=tol, atol=tol):
                status = f"Converged after {i+1} iterations."
                break

            q = q_next
            history.append(q)
        else:
            status = "Max iterations reached without convergence"

        return {"Q_matrix": q, "status": status, "history": history}
