import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use("ggplot")
from time import time

import numpy as np
from tqdm import tqdm

from DP.utils import (binom_optimal_privacy, fisher_information_binom,
                      fisher_information_privatized)


class DP_tester:
    @staticmethod
    def plot_fisher_infos(solver, ns: list, epsilon: float):
        ncols = 2
        nrows = len(ns) // 2

        thetas = np.linspace(1e-1, 1 - 1e-1, 100)

        fig, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(8, 3 * nrows), sharey=True, sharex=True
        )
        axes = axes.flatten()

        for i, n in tqdm(enumerate(ns)):
            orig_fisher_infs = fisher_information_binom(n, thetas)
            privatized_fisher_infs = list()
            for theta in thetas:
                q, status, history = binom_optimal_privacy(solver, n, epsilon, theta)
                finfo = fisher_information_privatized(q, n, theta)
                privatized_fisher_infs.append(finfo)
            axes[i].plot(thetas, orig_fisher_infs, label="Original model")
            axes[i].plot(thetas, privatized_fisher_infs, label="Optimal Private Q")
            axes[i].set_xlabel(r"$\theta$")
            axes[i].set_ylabel(r"$I(\theta, Q)$")
            axes[i].set_title(f"$n={n}$")
        axes[0].legend()
        plt.suptitle(
            f"Binomial model Fisher information, solver {solver.name}, epsilon {epsilon}"
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def fisher_inf_vs_epsilon(solver, n, theta, epsilon_min=1e-2, epsilon_max=10):
        orig_fisher_information = fisher_information_binom(n, theta)
        epsilons = np.linspace(epsilon_min, epsilon_max, 100)

        fishers_private = list()
        for eps in tqdm(epsilons):
            q_matrix, _, _ = binom_optimal_privacy(solver, n, eps, theta)
            finfo = fisher_information_privatized(q_matrix, n, theta)
            fishers_private.append(finfo)

        fig, ax = plt.subplots()

        ax.plot(
            epsilons,
            fishers_private,
            label="Private data fisher information",
            linestyle="--",
        )
        ax.hlines(
            [orig_fisher_information],
            xmin=0,
            xmax=epsilon_max,
            label="Original model fisher information",
            color="green",
        )
        plt.legend()
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$I(\theta)$")
        plt.title(
            "Fisher information as a function of epsilon \n"
            + f"$n={n}$, $\theta={theta}$, solver is {solver.name}"
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_fisher_two_solvers(solver1, solver2, n, epsilon):
        thetas = np.linspace(1e-1, 1 - 1e-1, 100)

        orig_fisher_infs = fisher_information_binom(n, thetas)

        solver1_fisher_infs = list()
        solver2_fisher_infs = list()
        converged_solver1 = list()
        converged_solver2 = list()
        for theta in tqdm(thetas):
            q1, status, _ = binom_optimal_privacy(solver1, n, epsilon, theta)
            if "Converged" in status:
                converged_solver1.append(True)
            else:
                converged_solver1.append(False)
            finfo1 = fisher_information_privatized(q1, n, theta)
            solver1_fisher_infs.append(finfo1)

            q2, status, _ = binom_optimal_privacy(solver2, n, epsilon, theta)
            if "Converged" in status:
                converged_solver2.append(True)
            else:
                converged_solver2.append(False)
            finfo2 = fisher_information_privatized(q2, n, theta)
            solver2_fisher_infs.append(finfo2)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(thetas, orig_fisher_infs, label="Original model")
        ax.plot(thetas, solver1_fisher_infs, label=f"Optimal Q {solver1.name}")
        ax.plot(thetas, solver2_fisher_infs, label=f"Optimal Q {solver2.name}")
        converged_solver2 = np.array(converged_solver2)
        non_converged_indices_2 = ~converged_solver2
        solver2_fisher_infs = np.array(solver2_fisher_infs)
        print(thetas[non_converged_indices_2])
        ax.scatter(
            thetas[non_converged_indices_2],
            solver2_fisher_infs[non_converged_indices_2],
            marker="x",
            color="red",
            s=50,
            label=f"{solver2.name} not converged",
        )
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$I(\theta, Q)$")
        ax.set_title(
            rf"$n={n}, \epsilon={epsilon}$, solver 1 {solver1.name}, solver 2 {solver2.name}"
        )
        ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_runtimes(solvers, ns, theta, epsilon, log=False):

        times = list()

        for solver in solvers:
            print(f"Calculating for {solver.name}")
            solver_times = list()
            for n in tqdm(ns):
                t_start = time()
                q, status, _ = binom_optimal_privacy(solver, n, epsilon, theta)
                t_end = time()

                solver_times.append(t_end - t_start)
            times.append(solver_times)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(len(solvers)):
            ax.plot(ns, times[i], label=solvers[i].name)
        ax.set_xlabel("$n$")
        ax.set_ylabel("Time (s)")
        ax.set_title(rf"Runtime comparisons, $\theta={theta}, \epsilon={epsilon}$")
        if log:
            ax.set_yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.show()
