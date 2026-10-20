#!/usr/bin/env python3
#######################################################
#                                                     #
#           Surrogate Based Optimization              #
#     framework for multistage compressor design      #
#                                                     #
#######################################################
# Authors:                                            #
#    Murillo S. S. Pereira Neto,                      #
#    @ University of São Paulo, Brazil                #
#    Bruno José Almeida Nagy,                         #
#    @ University of São Paulo, Brazil                #
#                                                     #
#                                                     #
# Description:                                        #
#######################################################

#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.logger import *

class Optimization():

    scipy_gradient_free_unconstrained = [
        "Nelder-Mead",
        "Powell",
    ]

    scipy_gradient_free_constrained = [
        "COBYLA",
        "COBYQA",
    ]

    scipy_gradient_based_unconstrained = [
        "CG",
        "BFGS",
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]

    scipy_gradient_based_constrained = [
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "trust-constr",
    ]

    scipy_algorithms = (scipy_gradient_free_unconstrained + scipy_gradient_free_constrained
                        + scipy_gradient_based_unconstrained + scipy_gradient_based_constrained)

    def get_optimization_algorithm(self, algorithm='nsga2'):

        self.algorithm = algorithm

        if self.algorithm == 'nsga2':
            return self.optimize_nsga2

        elif self.algorithm in self.scipy_algorithms:
            return self.optimize_scipy

        else:
            raise Exception(f"Algorithm {self.algorithm} not available for optimization")

    # -----------------------------------------------------#
    # NSGA-II
    # -----------------------------------------------------#

    def optimize_nsga2(self, func, X_min, X_max, n_obj=1, n_constr=0, IN={}):

        class ProblemWrapper(ElementwiseProblem):
            def __init__(self, X_min, X_max, n_obj, n_constr):
                n_var = len(X_min)
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=n_constr,
                    xl=X_min,
                    xu=X_max,
                )
                log_print(f"Number of DVs: {n_var}\nNumber of constraints: {n_constr}")

            def _evaluate(self, x, out, *args, **params):
                F, G = func(np.array(x))
                out["F"] = F
                out["G"] = G

        # Defining problem
        problem = ProblemWrapper(X_min, X_max, n_obj, n_constr)

        class PrintBestCallback(Callback):
            def __init__(self):
                super().__init__()

            def notify(self, algorithm):
                F = algorithm.pop.get("F")  # objective values
                X = algorithm.pop.get("X")  # decision variable vectors
                G = algorithm.pop.get("G")  # constraint violations

                i_best = F[:, 0].argmin()
                best_F = float(F[i_best, 0])
                best_X = X[i_best]

                # Worst constraint violation in current population
                if G is not None and G.size > 0:
                    worst_g = float(np.amax(G))
                else:
                    worst_g = 0.0
                g_status = "" if worst_g > 0 else " (ok)"

                EPS = 1e-4
                active_constraints = []
                for i, x in enumerate(best_X):
                    active_constraints.append(x - X_min[i] < EPS
                                              or X_max[i] - x < EPS)

                log_print(
                    f"Gen {algorithm.n_gen}: "
                    f"f_best = {best_F:.4e}, "
                    f"X_best = [" + ", ".join(f"{x:.4f}{' (x)' if active_constraints[i] else ''}" for i, x in enumerate(best_X)) + "], "
                    f"worst_g = {worst_g:.2e}{g_status}"
                )

        # Population size
        pop_size = IN['nsga2']["opt_nsga2_pop_size"][0]
        # Number of generations
        n_gen = IN['nsga2']["opt_nsga2_n_gen"][0]
        stop_criteria = ('n_gen', n_gen)
        # LatinHypercubeSampling:       pymoo\operators\sampling\lhs.py
        sampling = LHS()
        # SimulatedBinaryCrossover:     pymoo\operators\crossover\sbx.py
        crossover = SBX(prob=IN['nsga2']["opt_nsga2_prob_cross"][0])
        # PolynomialMutation:           pymoo\operators\mutation\pm.py
        mutation = PolynomialMutation(prob=IN['nsga2']["opt_nsga2_prob_mut"][0])
        # Genetic algorithm
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation)
        # Solve optimization/ minimization problem
        results = minimize(problem=problem,
                           algorithm=algorithm,
                           termination=stop_criteria,
                           callback=PrintBestCallback(),
                           use_threads=False)

        # Pareto front
        X_Pareto = results.X
        Obj_Pareto = results.F
        Constraints_Pareto = results.G
        return X_Pareto, Obj_Pareto[0], Constraints_Pareto

    # -----------------------------------------------------#
    # NSGA-II (vectorized: func evaluates the whole population)
    # -----------------------------------------------------#

    def optimize_nsga2_batch(self, func_batch, X_min, X_max, n_obj=1, n_constr=0, IN={}):
        """ func_batch(X) -> (F, G), X: (n_pop, n_var), F: (n_pop, n_obj), G: (n_pop, n_constr)

        Returns the best feasible solution (first Pareto point for n_obj > 1). If no feasible
        solution is found, the least-violating individual of the final population is returned.
        Returns X_opt, F_opt, G_opt, feasible
        """

        class BatchProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(X_min), n_obj=n_obj, n_ieq_constr=n_constr, xl=X_min, xu=X_max)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = func_batch(X)
                out["F"] = F
                if n_constr > 0:
                    out["G"] = G

        class PrintBestCallback(Callback):
            def __init__(self, n_print):
                super().__init__()
                self.n_print = n_print

            def notify(self, algorithm):
                if algorithm.n_gen % self.n_print != 0:
                    return
                F = algorithm.pop.get("F")
                CV = algorithm.pop.get("CV")
                feasible = (CV[:, 0] <= 0) if CV is not None else np.ones(len(F), dtype=bool)
                if np.any(feasible):
                    best_F = float(np.amin(F[feasible, 0]))
                    log_print(f"Gen {algorithm.n_gen}: f_best = {best_F:.6e}, feasible = {int(np.sum(feasible))}/{len(F)}")
                else:
                    log_print(f"Gen {algorithm.n_gen}: no feasible individual (min CV = {float(np.amin(CV)):.2e})")

        nsga2_IN = IN['nsga2']
        pop_size = nsga2_IN["opt_nsga2_pop_size"][0]
        n_gen = nsga2_IN["opt_nsga2_n_gen"][0]
        seed = nsga2_IN["opt_nsga2_seed"][0] if "opt_nsga2_seed" in nsga2_IN else None
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=LHS(),
                          crossover=SBX(prob=nsga2_IN["opt_nsga2_prob_cross"][0]),
                          mutation=PolynomialMutation(prob=nsga2_IN["opt_nsga2_prob_mut"][0]))
        log_print(f"NSGA-II: pop_size = {pop_size}, n_gen = {n_gen}, n_var = {len(X_min)}, "
                  f"n_obj = {n_obj}, n_constr = {n_constr}")
        results = minimize(problem=BatchProblem(),
                           algorithm=algorithm,
                           termination=('n_gen', n_gen),
                           callback=PrintBestCallback(max(1, n_gen // 10)),
                           seed=seed)

        if results.X is not None:
            X = np.atleast_2d(results.X)
            F = np.atleast_2d(results.F)
            i_best = int(np.argmin(F[:, 0]))
            X_opt, F_opt = X[i_best], F[i_best]
            feasible = True
        else:
            # No feasible solution: least constraint violation, then best objective
            pop = results.pop
            CV = pop.get("CV")[:, 0]
            F = pop.get("F")
            candidates = np.where(CV == np.amin(CV))[0]
            i_best = candidates[np.argmin(F[candidates, 0])]
            X_opt, F_opt = pop.get("X")[i_best], F[i_best]
            feasible = False
        F_opt, G_opt = func_batch(np.atleast_2d(X_opt))
        return np.array(X_opt), F_opt[0], (G_opt[0] if n_constr > 0 else np.array([])), feasible

    def optimize_scipy(self, func, X_min, X_max, n_obj=1, n_constr=0, IN={}):
        pass