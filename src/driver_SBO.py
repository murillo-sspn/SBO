#!/usr/bin/env python3
#######################################################
#                                                     #
#           Surrogate Based Optimization              #
#        Framework for the Parametric Design          #
#              of Compressor Stages                   #
#                                                     #
#######################################################
# Authors:                                            #
#    Murillo S. S. Pereira Neto,                      #
#    @ University of São Paulo, Brazil                #
#                                                     #
#                                                     #
# Description:                                        #
#######################################################

#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.utils import *
from common.plotter import *
from src.models import *
from src.thermo import Thermo
from src.opt import Optimization

def print_banner():
    log_print("#######################################################")
    log_print("#                                                     #")
    log_print("#           Surrogate Based Optimization              #")
    log_print("#        Framework for the Parametric Design          #")
    log_print("#              of Compressor Stages                   #")
    log_print("#                                                     #")
    log_print("#######################################################")
    log_print("")

class Driver(Thermo, Optimization):

    def __init__(self, IN, directories):
        print_banner()
        super().__init__()
        # Arguments
        self.IN = IN
        self.directories = directories
        # Reading datasets
        self.keys_var_dependent     = self.IN['var_dependent']
        self.keys_var_independent   = self.IN['var_independent']
        self.dataset_training       = csv_to_dict(self.IN['dataset_training'][0])
        self.dataset_validation     = csv_to_dict(self.IN['dataset_validation'][0])
        self.datasets               = [self.dataset_training, self.dataset_validation]

        def get_dataset_size(keys_var, dataset):
            intersection = list(set(dataset).intersection(dataset.keys()))
            return np.shape(dataset[intersection[0]])[1]

        self.N_training     = get_dataset_size(self.keys_var_independent, self.dataset_training)
        self.N_validation   = get_dataset_size(self.keys_var_independent, self.dataset_validation)
        # Internal datasets
        self.dataset = {}

        # Pre process
        self.pre_process()

        # Train models
        self.train_models()

        # Set optimization problem
        self.set_optimization()

        # If flag is active...
        if self.IN.get("opt_flag"):
            # Optimize multistage compressor
            self.optimize()

        # Post process solution
        self.post_process()



    # -----------------------------------------------------#
    # Pre processing
    # -----------------------------------------------------#

    def pre_process(self):
        pass

    # -----------------------------------------------------#
    # Training models
    # -----------------------------------------------------#

    def train_models(self):

        # Input variables
        X_training      = self._get_X_from_dataset(self.dataset_training)
        X_validation    = self._get_X_from_dataset(self.dataset_validation)
        self.surrogate_models        = []
        self.best_surrogate_models   = []
        # Looping output variables
        for index_var_dep, key_var_dependent in enumerate(self.keys_var_dependent):
            y_training      = self.dataset_training[key_var_dependent].transpose()
            y_validation    = self.dataset_validation[key_var_dependent].transpose()
            y_surrogate_models = []
            # Looping metamodels
            for model_type in self.IN['surrogate_models']:
                # Create and train metamodel
                model = SurrogateFactory.create(model_type, self.IN, index_var_dep)
                # Optimize hyperparameters to minimize R² w.r.t. validation
                model.optimize_hyperparameters(X_training, y_training, X_validation, y_validation)
                # Train model
                model.train(X_training, y_training)
                # Get model results
                y_predicted_training = np.array([model.predict(X_training)]).transpose()
                y_predicted_validation = np.array([model.predict(X_validation)]).transpose()
                summary = dict({'model': model,
                                'training': model.validate(y_training, y_predicted_training),
                                'validation': model.validate(y_validation, y_predicted_validation)})
                y_surrogate_models.append(summary)

                # Plot validation results
                self._plot_model(y_validation, y_predicted_validation,
                                 model, key_var_dependent, summary['validation'], 'validation')
                self._plot_model(y_training, y_predicted_training,
                                 model, key_var_dependent, summary['training'], 'training')

                # Prints to screen
                res_txt = ''
                for key, val in summary['validation'].items(): res_txt += f'\t{key}={"%0.2e"%val}'
                log_print(f"{key_var_dependent}\t{model_type}{res_txt}")
            # Append to surrogate_models
            self.surrogate_models.append(y_surrogate_models)
            # Get best model for this variable
            y_surrogate_models_r2 = []
            for i, model_type in enumerate(self.IN['surrogate_models']):
                y_surrogate_models_r2.append(y_surrogate_models[i]["validation"]["R2"])
            best_model = y_surrogate_models[int(np.argmax(y_surrogate_models_r2))]['model']
            self.best_surrogate_models.append(best_model)
            log_print(f"    Selecting {best_model.name}")

    def _get_X_from_dataset(self, dataset):
        return np.array([dataset[key_var_independent][0] for key_var_independent in
                               self.keys_var_independent]).transpose()

    def evaluate_best_model_on_dataset(self, dataset):

        # Looping output variables
        for i, key_var_dependent in enumerate(self.keys_var_dependent):
            best_model_i = self.best_surrogate_models[i]
            X_grid = np.array([dataset[key_var_independent]
                               for key_var_independent in self.keys_var_independent]).transpose()[0]
            y_grid_i = np.array([best_model_i.predict(X_grid)])
            dataset[key_var_dependent] = y_grid_i

    # -----------------------------------------------------#
    # Parametric optimization
    # -----------------------------------------------------#

    def set_optimization(self):

        # Set bounds
        X_min, X_max = [], []
        for key_var in self.keys_var_independent:
            X_min.append(np.amin(self.dataset_training[key_var]))
            X_max.append(np.amax(self.dataset_training[key_var]))
        self.X_min = np.array(X_min)
        self.X_max = np.array(X_max)

        # Number of objective functions
        opt_obj = self.IN.get('opt_obj')
        self.n_obj = len(opt_obj)
        self.obj_surrogates = {}
        for i in range(self.n_obj):
            obj_key = opt_obj[i]
            self.obj_surrogates[obj_key] = self.best_surrogate_models[self.keys_var_dependent.index(obj_key)]

        # Number of constraints
        opt_constraints = self.IN.get('opt_constraints')
        self.n_constraints = len(opt_constraints)
        self.constraints_surrogates = {}
        for i in range(self.n_constraints):
            constraints_key = opt_constraints[i]
            self.constraints_surrogates[constraints_key] = self.best_surrogate_models[self.keys_var_dependent.index(constraints_key)]

    def _get_performance(self, DV):


        if len(np.shape(DV)) == 1:    DV = np.array([DV])

        obj_val = {}
        opt_obj = self.IN.get('opt_obj')
        for i in range(self.n_obj):
            obj_key = opt_obj[i]
            obj_val[obj_key]=(float(self.obj_surrogates[obj_key].predict(DV)[0]))

        constraints_val = {}
        opt_constraints = self.IN.get('opt_constraints')
        for i in range(self.n_constraints):
            constraints_key = opt_constraints[i]
            constraints_val[constraints_key]=(float(self.constraints_surrogates[constraints_key].predict(DV)[0]))

        return obj_val, constraints_val

    def _get_objective_and_constraints(self, DV):

        obj_val, constraints_val = self._get_performance(DV)

        obj = []
        for i, obj_option in enumerate(self.IN.get('opt_obj')):
            min_max = self.IN.get('opt_min_max')[i]
            _obj = obj_val[obj_option]
            if 'abs' in min_max:
                _obj = np.abs(_obj)
            if 'max' in min_max:
                _obj = -_obj
            obj.append(float(_obj))

        constraints = []
        opt_constraints = self.IN.get('opt_constraints')
        opt_constraints_baseline = self.IN.get('opt_constraints_baseline')
        opt_constraints_above_below_baseline = self.IN.get('opt_constraints_above_below_baseline')
        for i in range(self.n_constraints):
            constraints_key = opt_constraints[i]
            _baseline = opt_constraints_baseline[i]
            _above_below = opt_constraints_above_below_baseline[i]
            if _above_below == "above":
                _constraint = _baseline - constraints_val[constraints_key] # < 0
            elif _above_below == "below":
                _constraint = constraints_val[constraints_key] - _baseline # < 0
            else:
                raise Exception(f"Incorrect option: _above_below = {_above_below}")
            constraints.append(_constraint)

        return obj, constraints

    def optimize(self):

        # Get optimization algorithm
        for i, opt_method in enumerate(self.IN.get('opt_method')):

            self.optimization_algorithm = self.get_optimization_algorithm(algorithm=opt_method)

            out = self.optimization_algorithm(func = self._get_objective_and_constraints,
                                              X_min = self.X_min,
                                              X_max = self.X_max,
                                              n_obj = self.n_obj,
                                              n_constr = self.n_constraints,
                                              IN = self.IN)
            X_opt, objective_opt, constraints_opt = out
            objective_opt = objective_opt.tolist()
            constraints_opt = constraints_opt[0].tolist()

    def post_process(self):
        pass

    # -----------------------------------------------------#
    # Plotting
    # -----------------------------------------------------#

    def _plot_model(self, y_model, y_predicted, model, key_var_dependent, statistics, dataset_name=""):

        key_var_independent = ""
        for key_var in self.keys_var_independent: key_var_independent += f"{key_var}, "
        key_var_independent = f"({key_var_independent[:-2]})"

        x_label = 'Model data'
        y_label = 'Predicted data'

        if dataset_name == "":  _dataset_name = ""
        else:                   _dataset_name = f" ({dataset_name})"
        title = f"Model and predicted data{_dataset_name}\n{key_var_dependent.replace('_', ' ')}\nModel: {model.name} (R²={'%0.2f'%statistics['R2']})"
        if dataset_name == "":  _dataset_name = ""
        else:                   _dataset_name = f"-{dataset_name}"
        filename = os.path.join("Model",f"{model.name}-predicted-{key_var_dependent}{_dataset_name}")
        y_45 = np.array([np.linspace(min(np.amin(y_model),np.amin(y_predicted)),
                                     max(np.amax(y_model), np.amax(y_predicted)),
                                     2)]).transpose()
        x_lim = [np.amin(y_model),np.amax(y_model)]
        dx, eps = x_lim[1] - x_lim[0], 0.02
        x_lim = [x_lim[0]-dx*eps, x_lim[1]+dx*eps]

        plot_SBO(x_label, y_label, title, x_lim=x_lim,
                 X0=y_model, Y0=y_predicted, x0y0_kind="points", labels_list_x0=["data"],
                 X1=y_45, Y1=y_45, x1y1_kind="dotted", labels_list_x1=["y=x"],
                 flag_aspect_ratio=False,
                 output_directory=self.directories.outputs_directory, loc='upper left',
                 filename=filename)

class Solution:

    def __init__(self, X, objective, constraints, objective_reduction):

        self.X = X
        self.objective = objective
        self.constraints = constraints
        self.objective_reduction = objective_reduction
