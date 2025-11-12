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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from smt.surrogate_models import KRG, RBF

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#

# -------------------------------
# Abstract Base Surrogate Model
# -------------------------------
class SurrogateModel(ABC):
    def __init__(self, cfg=None):
        self.cfg = cfg

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
# Kriging Model
# -------------------------------
class KrigingModel(SurrogateModel):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.model = KRG(print_global=False)

    def train(self, X, y):
        self.model.set_training_values(X, y)
        self.model.train()

    def predict(self, X):
        return self.model.predict_values(X).ravel()


# -------------------------------
# Radial Basis Function Model
# -------------------------------
class RBFModel(SurrogateModel):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.model = RBF(print_global=False)

    def train(self, X, y):
        self.model.set_training_values(X, y)
        self.model.train()

    def predict(self, X):
        return self.model.predict_values(X).ravel()


# -------------------------------
# Neural Network Model
# -------------------------------
class NNModel(SurrogateModel):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        hidden_layers = cfg.get("hidden_layers", (10, 10))
        activation = cfg.get("activation", "relu")
        solver = cfg.get("solver", "adam")
        max_iter = int(cfg.get("max_iter", 500))
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                                  activation=activation,
                                  solver=solver,
                                  max_iter=max_iter,
                                  random_state=int(cfg.get("random_seed", 42)))

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


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
