import numpy as np
import cvxpy as cp
from DP.utils import fisher_gradient, fisher_information_privatized, binom_optimal_privacy
from typing import Tuple


def initialize_projection_solver(n_trials: int, epsilon: float) -> Tuple[cp.Problem, cp.Variable, cp.Parameter]:
    n_plus_1 = n_trials + 1
    Q_var = cp.Variable((n_plus_1, n_plus_1))
    Q_param = cp.Parameter((n_plus_1, n_plus_1))
    
    objective = cp.Minimize(cp.sum_squares(Q_var - Q_param))

    constraints = []
    constraints += [Q_var >= 0]

    for j in range(n_plus_1):
        constraints += [cp.sum(Q_var[:, j]) == 1]

    exp_eps = np.exp(epsilon)
    exp_neg_eps = np.exp(-epsilon)
    for i in range(n_plus_1):
        Q_i = Q_var[i, :]
        constraints += [Q_i[j_prime] - exp_neg_eps * Q_i[j] >= 0
                        for j in range(n_plus_1)
                        for j_prime in range(n_plus_1) if j != j_prime]
        constraints += [exp_eps * Q_i[j_prime] - Q_i[j] >= 0
                        for j in range(n_plus_1)
                        for j_prime in range(n_plus_1) if j != j_prime]
        
    prob = cp.Problem(objective, constraints)

    return prob, Q_var, Q_param


class projected_gradient_ascent:
    name = "PGA"

    def __call__(self, p_theta, p_theta_dot, theta, epsilon, n_trials, tol=1e-6, max_iter=100):
        Q_init = np.ones((n_trials + 1, n_trials + 1)) / (n_trials + 1) + np.random.normal(size=(n_trials+1, n_trials+1), scale=0.01)
        #Q_init = project_onto_feasible_set(Q_init, epsilon)

        projection_problem, Q_var, Q_param = initialize_projection_solver(n_trials, epsilon)

        q = Q_init
        fish = fisher_information_privatized(q, n_trials, theta)
        history = [Q_init]

        for i in range(max_iter):
            grad_I = fisher_gradient(p_theta, p_theta_dot, q)
            #grad_I[-1, :] = np.zeros_like(grad_I[-1, :])

            # We need to clip the gradient since for very small rows
            # that tend to 0 we will get bonkers gradients
            # On the other hand, at extreme values of theta
            # we need these bonkers gradients
            # This is why we cannot simply bound the gradients
            # to say [-1, 1]
            grad_I[(grad_I > 10000) | (grad_I < -10000)] = grad_I[(grad_I > 10000) | (grad_I < -10000)] / 2
            grad_I[(grad_I > 1e7) | (grad_I < -1e7)] = 0

            q_next = q + grad_I
            
            Q_param.value = q_next
            Q_var.value = q_next
            projection_problem.solve(solver=cp.SCS)
            q_next = Q_var.value

            fish_next = fisher_information_privatized(q_next, n_trials, theta)

            if np.allclose(q, q_next, rtol=tol, atol=tol):
                status = f"Converged after {i+1} iterations."
                break

            if abs(fish - fish_next) < 1e-8:
                status = f"Converged after {i+1} iteratons."
                break

            q = q_next
            fish = fish_next
            history.append(q)
            status = "Max iterations reached without convergence"
        
        return {"Q_matrix": q, "status": status, "history": history}
    

if __name__ == "__main__":
    solver = projected_gradient_ascent()

    q, status, history = binom_optimal_privacy(solver, 1, 1.0, 0.5)

    print(q)
    print(status)
    print(history)
        
