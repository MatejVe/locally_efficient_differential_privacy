from scipy.optimize import minimize
import numpy as np

from DP.utils import (
    fisher_information_privatized,
    print_matrix,
    reduce_optimal_matrix,
)


class ScipySolver:
    name = "scipy_solver"

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, p_theta, p_theta_dot, epsilon, k):
        Q_init = np.ones((k, k)) / (k) + np.random.normal(size=(k, k), scale=0.1)

        def objective(Q):
            return -fisher_information_privatized(Q.reshape(k, k), p_theta, p_theta_dot)

        constraints = []
        # Privacy constraints
        for i in range(k):
            for j in range(k):
                for j_prime in range(k):
                    if j != j_prime:
                        constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda Q, i=i, j=j, j_prime=j_prime: Q.reshape(
                                    (k, k)
                                )[i, j_prime]
                                - np.exp(-epsilon) * Q.reshape((k, k))[i, j],
                            }
                        )
                        constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda Q, i=i, j=j, j_prime=j_prime: np.exp(
                                    epsilon
                                )
                                * Q.reshape((k, k))[i, j_prime]
                                - Q.reshape((k, k))[i, j],
                            }
                        )
        # Sum-to-1 column-wise constraints
        for j in range(k):
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda Q, j=j: np.sum(Q.reshape((k, k))[:, j]) - 1,
                }
            )
        # Bounds for each entry in Q
        bounds = [(0, 1) for _ in range(k * k)]

        def callback(Q):
            print_matrix(Q.reshape(k, k))

        if self.verbose:
            result = minimize(
                objective,
                Q_init.flatten(),
                constraints=constraints,
                bounds=bounds,
                callback=callback,
                method="SLSQP",
            )
        else:
            result = minimize(
                objective,
                Q_init.flatten(),
                constraints=constraints,
                bounds=bounds,
                method="SLSQP",
            )

        Q_optimal = result.x.reshape(k, k)
        Q_optimal = reduce_optimal_matrix(Q_optimal)
        status = result.success
        if status:
            status = "Converged"
        else:
            status = "Did not converge"

        return {"Q_matrix": Q_optimal, "status": status, "history": None}


class ScipySolverRestarts:
    name = "scipy_solver_restarts"

    def __init__(self, n_restarts: int = 10, verbose=False):
        self.n_restarts = n_restarts
        self.verbose = verbose

    def __call__(self, p_theta, p_theta_dot, epsilon, k):
        best_fish = -np.inf
        best_q = None
        stat = None
        history = None

        for _ in range(self.n_restarts):
            solver = ScipySolver(verbose=self.verbose)
            results = solver(p_theta, p_theta_dot, epsilon, k)
            q = results["Q_matrix"]
            status = results["status"]
            hist = results["history"]
            fish_value = fisher_information_privatized(q, p_theta, p_theta_dot)

            if fish_value > best_fish:
                best_fish = fish_value
                best_q = q
                stat = status
                history = hist

        return {"Q_matrix": best_q, "status": stat, "history": history}
