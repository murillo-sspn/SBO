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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
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
        self.continuous = True

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
# Models (continuous variables)
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
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def init_model(self, index_var_dep=0):
        self.nn_opt_hyperparameters     = (self.cfg['neural_networks']["nn_opt_hyperparameters"][index_var_dep] == "True")
        self.hidden_layers              = self.cfg['neural_networks']["nn_hidden_layers"][index_var_dep]
        self.n_hidden_layers            = len(self.hidden_layers)
        self.activation                 = self.cfg['neural_networks']["nn_activation"][index_var_dep]
        self.solver                     = self.cfg['neural_networks']["nn_solver"][index_var_dep]
        self.max_iter                   = self.cfg['neural_networks']["nn_max_iter"][index_var_dep]
        self.tol                        = self.cfg['neural_networks']["nn_tol"][index_var_dep]
        self.random_state               = int(self.cfg['neural_networks']["nn_random_seed"][index_var_dep])
        self.learning_rate_init         = self.cfg['neural_networks']["nn_learning_rate_init"][index_var_dep]
        self.scale_Xy                   = self.cfg['neural_networks']["scale_Xy"]
        self._init_scaler_Xy()
        self._update_model()

        # NN optimization
        self.nn_opt_n_neurons_min   = self.cfg['neural_networks']["nn_opt_n_neurons_min"][0]
        self.nn_opt_n_neurons_max   = self.cfg['neural_networks']["nn_opt_n_neurons_max"][0]
        self.nn_opt_pop_size        = self.cfg['neural_networks']["nn_opt_pop_size"][0]
        self.nn_opt_n_gen           = self.cfg['neural_networks']["nn_opt_n_gen"][0]
        self.nn_opt_prob_cross      = self.cfg['neural_networks']["nn_opt_prob_cross"][0]
        self.nn_opt_prob_mut        = self.cfg['neural_networks']["nn_opt_prob_mut"][0]

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
        self.model.fit(X_train, y_train)

    def predict(self, X):
        # Scale
        X_predict = self._scale_X(X)
        # Predict
        y_predict = self.model.predict(X_predict)
        # Inverse scaling
        y = self._scale_y_inverse(y_predict)
        return y

    def optimize_hyperparameters(self, X_train, y_train, X_validation, y_validation,
                                 parameters={}):

        if self.nn_opt_hyperparameters:

            log_print(f"Optimizing hyperparameters of neural network")

            if len(parameters.keys()) == 0:
                parameters = {
                    "n_layers_min"   : self.cfg['neural_networks']["nn_opt_layers_min"][0],
                    "n_layers_max"   : self.cfg['neural_networks']["nn_opt_layers_max"][0],
                    "n_neurons_min"  : self.cfg['neural_networks']["nn_opt_n_neurons_min"][0],
                    "n_neurons_max"  : self.cfg['neural_networks']["nn_opt_n_neurons_max"][0],
                    "activation"     : self.cfg['neural_networks']["nn_opt_activation"],
                    "pop_size"       : self.cfg['neural_networks']["nn_opt_pop_size"][0],
                    "n_gen"          : self.cfg['neural_networks']["nn_opt_n_gen"][0],
                    "prob_cross"     : self.cfg['neural_networks']["nn_opt_prob_cross"][0],
                    "prob_mut"       : self.cfg['neural_networks']["nn_opt_prob_mut"][0],
                }

            # Get parameters
            _n_layers_max    = parameters["n_layers_max"]
            _n_layers_min    = parameters["n_layers_min"]
            _n_neurons_max   = parameters["n_neurons_max"]
            _n_neurons_min   = parameters["n_neurons_min"]
            _activation      = parameters["activation"]
            pop_size        = parameters["pop_size"]
            n_gen           = parameters["n_gen"]
            prob_cross      = parameters["prob_cross"]
            prob_mut        = parameters["prob_mut"]

            errors = []
            n_neurons_layers = []
            activation_func = []

            def _optimize_hyperparameters(X_train, y_train, X_validation, y_validation,
                                          n_layers, activation):

                # Fixed number of layers
                n_neurons_min = (_n_neurons_min * np.ones((1, n_layers)))[0]
                n_neurons_max = (_n_neurons_max * np.ones((1, n_layers)))[0]

                # History
                history = []
                self.func_eval = 0

                def func(n_neurons):
                    # Set model
                    model = MLPRegressor(hidden_layer_sizes=n_neurons,
                                          activation=activation,
                                          solver=self.solver,
                                          max_iter=self.max_iter,
                                          tol=self.tol,
                                          random_state=self.random_state,
                                          learning_rate_init=self.learning_rate_init,
                                          verbose=self.verbose)
                    # Train
                    _X_train, _y_train = self._scale_Xy(X_train, y_train)
                    model.fit(_X_train, _y_train)
                    _y_predicted_train = np.array([model.predict(_X_train)]).transpose()
                    y_predicted_train = self._scale_y_inverse(_y_predicted_train)
                    # Predict
                    _X_validation = self._scale_X(X_validation)
                    _y_predicted_validation = np.array([model.predict(_X_validation)]).transpose()
                    y_predicted_validation = self._scale_y_inverse(_y_predicted_validation)
                    # Validate
                    training = self.validate(y_train, y_predicted_train)
                    validation = self.validate(y_validation, y_predicted_validation)
                    # Get R2
                    #R2 = float(np.minimum(training["R2"],validation["R2"]))
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
                return n_neurons_opt, error_opt
                #self.hidden_layers = n_neurons_opt
                #self._update_model()

            for n_layers in range(_n_layers_min, _n_layers_max):
                for activation in _activation:
                    log_print(f"n_layers = {n_layers},\tactivation = {activation}")
                    n_neurons_opt, error_opt = _optimize_hyperparameters(X_train, y_train, X_validation, y_validation,
                                                              n_layers, activation)
                    n_neurons_layers.append(n_neurons_opt)
                    errors.append(error_opt)
                    activation_func.append(activation)
                    log_print("")
            _argmin = np.argmin(np.array(errors))
            self.hidden_layers = n_neurons_layers[_argmin]
            self.activation = activation_func[_argmin]
            self._update_model()



# -------------------------------
# Models (binary variables)
# -------------------------------

class RFModel(SurrogateModel):
    # Random Forest
    name = 'rf'
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.continuous = False

    def init_model(self, index_var_dep=0):
        self.model = RandomForestClassifier()

    def train(self, X, y, verbose=False):
        if verbose: log_print(f"Training Random Forest with provided data")
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X)

class SVMModel(SurrogateModel):
    # Random Forest
    name = 'svm'
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.continuous = False

    def init_model(self, index_surrogate_model_binary=0):
        self.svm_model = self.cfg['support_vector_machine']['svm_model'][index_surrogate_model_binary]
        if self.svm_model == 'SVC':         self.model = SVC()
        elif self.svm_model == 'NuSVC':     self.model = NuSVC()
        elif self.svm_model == 'LinearSVC': self.model = LinearSVC()

    def train(self, X, y, verbose=False):
        if verbose: log_print(f"Training Random Forest with provided data")
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X)

# -------------------------------
# Factory for Surrogate Models
# -------------------------------
class SurrogateFactory:
    @staticmethod
    def create(model_type, cfg):
        models = [KrigingModel, # Kriging model
                  RBFModel,     # Radial Basis Function model
                  NNModel,      # Neural Network model
                  RFModel,      # Random Forest (binary)
                  SVMModel]     # Support Vector Machine (binary)
        for model in models:
            if model_type == model.name:
                return model(cfg)
        else:
            raise ValueError(f"Unknown surrogate model type: {model_type}")
