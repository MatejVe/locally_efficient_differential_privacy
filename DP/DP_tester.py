import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use("ggplot")
from time import time

import numpy as np
from tqdm import tqdm

from DP.utils import (
    binom_optimal_privacy,
    fisher_information_binom,
    fisher_information_privatized,
)
from DP.linear_solver import LinearSolver


class DP_tester:
    @staticmethod
    def plot_fisher_infos(solver, ns: list, epsilon: float, n_thetas: int = 50, include_original=True):
        ncols = 2
        nrows = len(ns) // 2

        thetas = np.linspace(1e-1, 1 - 1e-1, n_thetas)

        fig, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(8, 3 * nrows), sharey=True, sharex=True
        )
        axes = axes.flatten()

        for i, n in tqdm(enumerate(ns)):
            orig_fisher_infs = fisher_information_binom(n, thetas)
            privatized_fisher_infs = list()
            for theta in thetas:
                q, _, _, best_fisher = binom_optimal_privacy(solver, n, epsilon, theta)
                privatized_fisher_infs.append(best_fisher)
            if include_original:
                axes[i].plot(thetas, orig_fisher_infs, label="Unsanitized data")
            axes[i].plot(thetas, privatized_fisher_infs, label="Optimal Private Q")
            axes[i].set_xlabel(r"$\theta$ (success probability parameter)")
            axes[i].set_ylabel(r"$I(\theta, Q)$")
            axes[i].set_title(f"$n={n}$")
        axes[0].legend()
        plt.suptitle(
            fr"Binomial model Fisher information, solver {solver.name}, $\epsilon = {epsilon}$"
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def fisher_inf_vs_epsilon(
        solver, n, theta, epsilon_min=1e-2, epsilon_max=10, n_epsilons=100
    ):
        orig_fisher_information = fisher_information_binom(n, theta)
        epsilons = np.linspace(epsilon_min, epsilon_max, n_epsilons)

        fishers_private = list()
        for eps in tqdm(epsilons):
            q_matrix, _, _, best_fisher = binom_optimal_privacy(solver, n, eps, theta)
            fishers_private.append(best_fisher)

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
        plt.xlabel(r"$\epsilon$ (privacy parameter)")
        plt.ylabel(r"$I(\theta)$")
        plt.title(
            "Fisher information as a function of epsilon \n"
            + f"$n={n}$, $\theta={theta}$, solver is {solver.name}"
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_fisher_two_solvers(solver1, solver2, n, epsilon, n_thetas=50):
        thetas = np.linspace(1e-1, 1 - 1e-1, n_thetas)

        orig_fisher_infs = fisher_information_binom(n, thetas)

        solver1_fisher_infs = list()
        solver2_fisher_infs = list()
        converged_solver1 = list()
        converged_solver2 = list()
        for theta in tqdm(thetas):
            q1, status, _, best_fisher = binom_optimal_privacy(solver1, n, epsilon, theta)
            if "Converged" in status:
                converged_solver1.append(True)
            else:
                converged_solver1.append(False)
            solver1_fisher_infs.append(best_fisher)

            q2, status, _, best_fisher = binom_optimal_privacy(solver2, n, epsilon, theta)
            if "Converged" in status:
                converged_solver2.append(True)
            else:
                converged_solver2.append(False)
            solver2_fisher_infs.append(best_fisher)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(thetas, orig_fisher_infs, label="Unsanitized data")
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
        ax.set_xlabel(r"$\theta$ (success probability parameter)")
        ax.set_ylabel(r"$I(\theta, Q)$")
        ax.set_title(
            rf"$n={n}, \epsilon={epsilon}$, solver 1 {solver1.name}, solver 2 {solver2.name}"
        )
        ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_fisher_multiple_solvers(solvers, n, epsilon, n_thetas=50):
        thetas = np.linspace(1e-1, 1 - 1e-1, n_thetas)

        orig_fisher_infs = fisher_information_binom(n, thetas)
        linear_fishes = list()
        for theta in thetas:
            q, _, _, best_fisher = binom_optimal_privacy(LinearSolver(), n, epsilon, theta)
            linear_fishes.append(best_fisher)
        linear_fishes = np.array(linear_fishes)

        fig, ax = plt.subplots(figsize=(6, 8), nrows=2, sharex=True)

        ax[0].plot(thetas, orig_fisher_infs, label="Unsanitized data")
        ax[0].plot(thetas, linear_fishes, label=f"Optimal Q Linear Solver")

        # empty plot to use up the blue color that the linear solver uses
        ax[1].plot([0.1, 0.9], [0, 0])
        ax[1].plot([0.1, 0.9], [0, 0])

        for solver in solvers:
            print(f"Calculating for {solver.name}")
            fisher_storage = list()
            for theta in tqdm(thetas):
                q, _, _, best_fisher = binom_optimal_privacy(solver, n, epsilon, theta)
                fisher_storage.append(best_fisher)
            fisher_storage = np.array(fisher_storage)

            ax[0].plot(thetas, fisher_storage, label=f"Optimal Q {solver.name}")
            ax[1].plot(
                thetas,
                linear_fishes - fisher_storage,
                label=f"Difference from optimal {solver.name}",
            )

        ax[1].set_xlabel(r"$\theta$ (success probability parameter)")
        ax[0].set_ylabel(r"$I(\theta, Q)$")
        ax[1].set_ylabel(r"$I(\theta, Q_{opt}) - I(\theta, Q)$")
        fig.suptitle(rf"$n={n}, \epsilon={epsilon}$")
        ax[0].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_runtimes(solvers, ns, theta, epsilon, log=False, n_restarts: int = 10):
        """ns is a list of max values of n to test for"""

        avg_times = list()
        stds = list()

        for i in range(len(solvers)):
            solver = solvers[i]
            print(f"Calculating for {solver.name}")
            solver_times = list()
            solver_stds = list()
            n_max = ns[i]
            for n in range(1, n_max + 1):
                print(f"Calculating for n={n}")
                times_for_n = list()
                for _ in range(n_restarts):
                    t_start = time()
                    _, _, _, _ = binom_optimal_privacy(solver, n, epsilon, theta)
                    t_end = time()

                    time_taken = t_end - t_start

                    times_for_n.append(time_taken)
                avg_time = np.mean(times_for_n)
                std_time = np.std(times_for_n)
                solver_times.append(avg_time)
                solver_stds.append(std_time)

            avg_times.append(solver_times)
            stds.append(solver_stds)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(len(solvers)):
            ns_to_plot = np.arange(1, ns[i] + 1)
            ax.plot(ns_to_plot, avg_times[i], label=solvers[i].name)
            ax.fill_between(ns_to_plot, np.array(avg_times[i]) - np.array(stds[i]), np.array(avg_times[i]) + np.array(stds[i]), alpha=0.3)
        ax.set_xlabel("n (input alphabet size)")
        ax.set_ylabel("Time (s)")
        ax.set_title(rf"Runtime comparisons, $\theta={theta}, \epsilon={epsilon}$")
        if log:
            ax.set_yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def max_discrepancy_between_two_solvers(
        solver1, solver2, ns, epsilons, sampled_thetas=100
    ):
        thetas = np.linspace(1e-2, 1 - 1e-2, sampled_thetas)

        discrepancies = np.zeros(shape=(len(ns), len(epsilons)))

        for i, n in tqdm(enumerate(ns)):
            for j, eps in enumerate(epsilons):
                abs_discrepancies = list()
                for t in thetas:
                    q1, _, _, best_fish1 = binom_optimal_privacy(solver1, n, eps, t)

                    q2, _, _, best_fish2 = binom_optimal_privacy(solver2, n, eps, t)

                    abs_discrepancies.append(abs(best_fish1 - best_fish2))
                discrepancies[i, j] = np.max(abs_discrepancies)

        # Plotting the discrepancies as a heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(
            discrepancies.T,
            aspect="auto",
            origin="lower",
            extent=[ns[0], ns[-1], epsilons[0], epsilons[-1]],
            cmap="coolwarm",
        )
        plt.colorbar(label="Max Error")
        plt.xlabel("n (input alphabet size)")
        plt.ylabel("ε (epsilon values)")
        plt.title("Maximum Error Between Two Solvers for Sampled Thetas")
        plt.show()

        return discrepancies
