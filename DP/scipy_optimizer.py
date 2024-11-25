from scipy.optimize import minimize
from DP.utils import fisher_information_privatized
import numpy as np


class scipy_optimizer:
    name = "SCIPY OPTIMIZER"

    def __call__(self, p_theta, p_theta_dot, theta, epsilon, n_trials):
        nrows = n_trials + 1
        ncols = n_trials + 1

        def objective(Q):
            Q = Q.reshape((nrows, ncols))
            fish = fisher_information_privatized(Q, n_trials, theta)
            return -fish

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
                                )[i, j_prime]
                                - np.exp(-epsilon) * Q.reshape((nrows, ncols))[i, j],
                            }
                        )
                        constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda Q, i=i, j=j, j_prime=j_prime: np.exp(
                                    epsilon
                                )
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

        return {
            "Q_matrix": result.x.reshape((nrows, ncols)),
            "status": result.message,
            "history": None,
        }
