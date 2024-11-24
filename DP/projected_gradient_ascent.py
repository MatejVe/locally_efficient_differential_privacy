import numpy as np
import cvxpy as cp
from DP.utils import fisher_gradient, fisher_information_privatized, binom_optimal_privacy

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
    exp_neg_eps = np.exp(-epsilon)
    for i in range(n_plus_1):
        Q_i = Q_var[i, :]
        constraints += [Q_i[j_prime] - exp_neg_eps * Q_i[j] >= 0
                        for j in range(n_plus_1)
                        for j_prime in range(n_plus_1) if j != j_prime]
        constraints += [exp_eps * Q_i[j_prime] - Q_i[j] >= 0
                        for j in range(n_plus_1)
                        for j_prime in range(n_plus_1) if j != j_prime]

    # Solve the problem
    Q_var.value = Q_param
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    if prob.status in ["infeasible", "unbounded"]:
        print(prob.status)
        print("Q matrix that is to be projected")
        print(Q_param)
        raise Exception("Problem with the solver")

    return Q_var.value

class projected_gradient_ascent:
    name = "PGA"

    def __call__(self, p_theta, p_theta_dot, theta, epsilon, n_trials, tol=1e-6, max_iter=100):
        Q_init = np.ones((n_trials + 1, n_trials + 1)) / (n_trials + 1) + np.random.normal(size=(n_trials+1, n_trials+1), scale=0.01)
        Q_init = project_onto_feasible_set(Q_init, epsilon)

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
            try:
                q_next = project_onto_feasible_set(q_next, epsilon)
            except Exception as e:
                print("Current Q matrix")
                print(q)
                print("Gradient")
                print(grad_I)
                raise e
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
        
