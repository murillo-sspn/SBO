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
import time
from abc import ABC, abstractmethod

# Metamodeling libraries
#   sklearn: https://scikit-learn.org/stable/index.html
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
#   smt: https://doi.org/10.1016/j.advengsoft.2019.03.005
from smt.surrogate_models import KRG, RBF
#   gstools: https://doi.org/10.5194/gmd-15-3161-2022
#   pykrige: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html

# Optimization
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback

from common.logger import log_print


#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#

# -------------------------------
# Common functions
# -------------------------------

class Scaling():

    def _init_scaler_Xy(self):
        if self.scale_Xy:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()

    def _scale_Xy(self, X, y):
        if self.scale_Xy:
            X_ = self.x_scaler.fit_transform(X)
            y_ = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            X_ = X
            y_ = y.ravel()
        return X_, y_

    def _scale_X(self, X):
        if self.scale_Xy:
            X_ = self.x_scaler.transform(X)
        else:
            X_ = X
        return X_

    def _scale_y_inverse(self, y):
        if self.scale_Xy:
            y_ = self.y_scaler.inverse_transform(y.reshape(-1,1)).transpose()[0]
        else:
            y_ = y
        return y_


# -------------------------------
# Abstract Base Surrogate Model
# -------------------------------
class SurrogateModel(ABC, Scaling):
    def __init__(self, cfg=None, verbose=False):
        self.cfg = cfg
        self.verbose = verbose

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def validate(self, y, y_pred):
        r2 = r2_score(y, y_pred)
        mae = max_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred)
        meae = mean_absolute_error(y, y_pred)


        return {
            "R2":r2,
            "MAE": mae,
            "RMSE": rmse,
            "MEAE": meae,
        }

# -------------------------------
# Models
# -------------------------------

class KrigingModel(SurrogateModel):
    name = 'kriging'
    def __init__(self, cfg=None, index_var_dep=0):
        super().__init__(cfg)
        self.model = KRG(print_global=False)

    def train(self, X, y):
        self.model.set_training_values(X, y)
        self.grid_search(X, y)
        self.model.train()

    def predict(self, X):
        return self.model.predict_values(X).ravel()

    def grid_search(self, X, y):
        pass

    def optimize_hyperparameters(self, X_train, y_train, X_validation, y_validation,parameters={}):
        pass

class RBFModel(SurrogateModel):
    name = 'rbf'
    def __init__(self, cfg=None, index_var_dep=0):
        super().__init__(cfg)
        self.model = RBF(print_global=False)

    def train(self, X, y):
        self.model.set_training_values(X, y)
        self.grid_search(X, y)
        self.model.train()

    def predict(self, X):
        return self.model.predict_values(X).ravel()

    def grid_search(self, X, y):
        pass

    def optimize_hyperparameters(self, X_train, y_train, X_validation, y_validation,parameters={}):
        pass

class NNModel(SurrogateModel):
    name = 'nn'
    def __init__(self, cfg=None, index_var_dep=0):
        super().__init__(cfg)
        self.nn_opt_hyperparameters     = (cfg.get("nn_opt_hyperparameters")[index_var_dep] == "True")
        self.hidden_layers              = cfg.get("nn_hidden_layers")[index_var_dep]
        self.n_hidden_layers            = len(self.hidden_layers)
        self.activation                 = cfg.get("nn_activation")[index_var_dep]
        self.solver                     = cfg.get("nn_solver")[index_var_dep]
        self.max_iter                   = cfg.get("nn_max_iter")[index_var_dep]
        self.tol                        = cfg.get("nn_tol")[index_var_dep]
        self.random_state               = int(cfg.get("nn_random_seed")[index_var_dep])
        self.learning_rate_init         = cfg.get("nn_learning_rate_init")[index_var_dep]
        self.scale_Xy                   = cfg.get("nn_scale_Xy", True)
        self._init_scaler_Xy()
        self._update_model()

        # NN optimization
        self.nn_opt_n_neurons_min   = cfg.get("nn_opt_n_neurons_min")[0]
        self.nn_opt_n_neurons_max   = cfg.get("nn_opt_n_neurons_max")[0]
        self.nn_opt_pop_size        = cfg.get("nn_opt_pop_size")[0]
        self.nn_opt_n_gen           = cfg.get("nn_opt_n_gen")[0]
        self.nn_opt_prob_cross      = cfg.get("nn_opt_prob_cross")[0]
        self.nn_opt_prob_mut        = cfg.get("nn_opt_prob_mut")[0]

    def _update_model(self):

        # Multi-layer Perceptron regressor
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layers,
                                  activation=self.activation,
                                  solver=self.solver,
                                  max_iter=self.max_iter,
                                  tol=self.tol,
                                  random_state=self.random_state,
                                  learning_rate_init=self.learning_rate_init,
                                  verbose=self.verbose)

    # -------------------------------
    # Scaling
    # -------------------------------



    def train(self, X, y, verbose=False):

        if verbose: log_print(f"Training neural network with provided data")
        # Train on scaled model
        X_train, y_train = self._scale_Xy(X, y)
        if self.cfg.get("nn_param_grid_flag", False): self.grid_search(X_train, y_train)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        # Scale
        X_predict = self._scale_X(X)
        # Predict
        y_predict = self.model.predict(X_predict)
        # Inverse scaling
        y = self._scale_y_inverse(y_predict)
        return y

    def grid_search(self, X, y):


        param_grid = {
            'hidden_layer_sizes':   self.cfg.get("nn_param_grid_hidden_layers"),
            'activation':           self.cfg.get("nn_param_grid_activation"),
            'alpha':                self.cfg.get("nn_param_grid_alpha"),
            'learning_rate_init':   self.cfg.get("nn_param_grid_learning_rate_init"),
            'tol':                  self.cfg.get("nn_param_grid_tol"),
            'max_iter':             self.cfg.get("nn_param_grid_max_iter")
        }

        grid = GridSearchCV(
            MLPRegressor(solver='adam', random_state=42),
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid.fit(X, y)

        log_print("Best parameters:", grid.best_params_)
        log_print("Best score:", -grid.best_score_)

    def optimize_hyperparameters(self, X_train, y_train, X_validation, y_validation,
                                 parameters={}):

        if self.nn_opt_hyperparameters:

            log_print(f"Optimizing hyperparameters of neural network")

            if len(parameters.keys()) == 0:
                parameters = {
                    "n_neurons_min"  : self.cfg.get("nn_opt_n_neurons_min", 5)[0],
                    "n_neurons_max"  : self.cfg.get("nn_opt_n_neurons_max", 5)[0],
                    "pop_size"       : self.cfg.get("nn_opt_pop_size", 20)[0],
                    "n_gen"          : self.cfg.get("nn_opt_n_gen", 10)[0],
                    "prob_cross"     : self.cfg.get("nn_opt_prob_cross", 0.8)[0],
                    "prob_mut"       : self.cfg.get("nn_opt_prob_mut", 0.8)[0],
                }

            # Get parameters
            n_neurons_max   = parameters["n_neurons_max"]
            n_neurons_min   = parameters["n_neurons_min"]
            pop_size        = parameters["pop_size"]
            n_gen           = parameters["n_gen"]
            prob_cross      = parameters["prob_cross"]
            prob_mut        = parameters["prob_mut"]

            # Fixed number of layers
            n_neurons_baseline = np.array([layer for layer in self.hidden_layers])
            try:
                n_neurons_min = n_neurons_min * np.ones_like(n_neurons_baseline)
            except:
                breakpoint()
            n_neurons_max = n_neurons_max * np.ones_like(n_neurons_baseline)

            # History
            history = []
            self.func_eval = 0

            def func(n_neurons):
                # Set model
                model = MLPRegressor(hidden_layer_sizes=n_neurons,
                                      activation=self.activation,
                                      solver=self.solver,
                                      max_iter=self.max_iter,
                                      tol=self.tol,
                                      random_state=self.random_state,
                                      learning_rate_init=self.learning_rate_init,
                                      verbose=self.verbose)
                # Train
                _X_train, _y_train = self._scale_Xy(X_train, y_train)
                model.fit(_X_train, _y_train)
                # Predict
                _X_validation = self._scale_X(X_validation)
                _y_predicted_validation = np.array([model.predict(_X_validation)]).transpose()
                y_predicted_validation = self._scale_y_inverse(_y_predicted_validation)
                # Validate
                validation = self.validate(y_validation, y_predicted_validation)
                # Get R2
                R2 = validation["R2"]
                # Get R2 error
                error = np.abs(R2 - 1)
                # Update history dict
                _dict = {"eval":self.func_eval, "R2":R2, "error":error}
                history.append(_dict)
                #print(f"eval = {self.func_eval}\tR2 = {'%0.2f'%R2}")
                # Update eval
                self.func_eval += 1
                return error

            func(n_neurons_baseline)

            class IntegerOptimizationProblem(ElementwiseProblem):
                def __init__(self, X_min, X_max):
                    super().__init__(
                        n_var=len(X_min),
                        n_obj=1,
                        n_constr=0,
                        xl=X_min,
                        xu=X_max,
                        vtype=int  # IMPORTANT: makes variables integer!
                    )

                def _evaluate(self, x, out, *args, **kwargs):
                    try:
                        out["F"] = func(np.array(x, dtype=int))
                    except:
                        breakpoint()

            # Defining problem
            problem = IntegerOptimizationProblem(n_neurons_min, n_neurons_max)

            class PrintBestCallback(Callback):
                def __init__(self):
                    super().__init__()

                def notify(self, algorithm):
                    F = algorithm.pop.get("F")  # objective values
                    X = algorithm.pop.get("X")  # decision variable vectors

                    i_best = F.argmin()  # best individual index
                    best_F = float(F[i_best])  # scalar value
                    best_X = X[i_best]  # vector

                    # Convert to integers
                    best_X_int = [int(x) for x in best_X]

                    print(
                        f"Gen {algorithm.n_gen}: f_best = {best_F:.6f}, "
                        f"X_best = {best_X_int}"
                    )

            # Stopping criteria
            stop_criteria = ('n_gen', n_gen)
            # LatinHypercubeSampling
            sampling = LHS()
            # SimulatedBinaryCrossover
            crossover = SBX(prob=prob_cross)
            # PolynomialMutation
            mutation = PolynomialMutation(prob=prob_mut)
            # Genetic algorithm
            algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
                              crossover=crossover, mutation=mutation,
                              eliminate_duplicates=True)

            # Optimize
            result = minimize(problem, algorithm,
                              termination=stop_criteria, callback=PrintBestCallback(), use_threads=False)
            # Get results
            dimX = len(np.shape(result.X))
            if dimX == 1:   n_neurons_opt = (result.X).astype(int).tolist()
            elif dimX == 2: n_neurons_opt = (result.X)[0].astype(int).tolist()
            error_opt = func(n_neurons_opt)

            log_print("===== OPTIMIZATION RESULT =====")
            log_print(f"Best X: {n_neurons_opt}")
            log_print(f"Best objective: {error_opt}")

            # Update model
            self.hidden_layers = n_neurons_opt
            self._update_model()



# -------------------------------
# Factory for Surrogate Models
# -------------------------------
class SurrogateFactory:
    @staticmethod
    def create(model_type, cfg, index_var_dep):
        models = [KrigingModel, # Kriging model
                  RBFModel,     # Radial Basis Function model
                  NNModel]      # Neural Network model
        for model in models:
            if model_type == model.name:
                return model(cfg, index_var_dep)
        else:
            raise ValueError(f"Unknown surrogate model type: {model_type}")
