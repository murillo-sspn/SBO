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
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from smt.surrogate_models import KRG, RBF

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
            X_ = self.x_scaler.fit_transform(X)
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
        rmse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        }

# -------------------------------
# Models
# -------------------------------

class KrigingModel(SurrogateModel):
    def __init__(self, cfg=None):
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

class RBFModel(SurrogateModel):
    def __init__(self, cfg=None):
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

class NNModel(SurrogateModel):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.hidden_layers = cfg.get("nn_hidden_layers", (10, 10))
        self.activation = cfg.get("nn_activation", "relu")
        self.solver = cfg.get("nn_solver", "adam")
        self.max_iter = (cfg.get("nn_max_iter", [500]))[0]
        self.tol = cfg.get("nn_tol", [1e-5])[0]
        self.random_state = int(cfg.get("nn_random_seed", [42])[0])
        self.learning_rate_init = (cfg.get("nn_learning_rate_init", [1e-3]))[0]
        self.scale_Xy = cfg.get("nn_scale_Xy", True)
        self._init_scaler_Xy()

        # Multi-layer Perceptron regressor
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layers,
                                  activation=self.activation,
                                  solver=self.solver,
                                  max_iter=self.max_iter,
                                  tol=self.tol,
                                  random_state=self.random_state,
                                  learning_rate_init = self.learning_rate_init,
                                  verbose=self.verbose)

    # -------------------------------
    # Scaling
    # -------------------------------



    def train(self, X, y):
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
            'hidden_layer_sizes': self.cfg.get("nn_param_grid_hidden_layers"),
            'activation': self.cfg.get("nn_param_grid_activation"),
            'alpha': self.cfg.get("nn_param_grid_alpha"),
            'learning_rate_init': self.cfg.get("nn_param_grid_learning_rate_init"),
            'tol': self.cfg.get("nn_param_grid_tol"),
            'max_iter': self.cfg.get("nn_param_grid_max_iter")
        }

        grid = GridSearchCV(
            MLPRegressor(solver='adam', random_state=42),
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid.fit(X, y)

        print("Best parameters:", grid.best_params_)
        print("Best score:", -grid.best_score_)

# -------------------------------
# Factory for Surrogate Models
# -------------------------------
class SurrogateFactory:
    @staticmethod
    def create(model_type, cfg):
        if model_type == "kriging":
            return KrigingModel(cfg)
        elif model_type == "rbf":
            return RBFModel(cfg)
        elif model_type == "nn":
            return NNModel(cfg)
        else:
            raise ValueError(f"Unknown surrogate model type: {model_type}")
