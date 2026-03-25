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
import csv
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
    log_print("#            Surrogate Based Optimization             #")
    log_print("#        Framework for the Parametric Design          #")
    log_print("#    with Convergence/Divergence Classification       #")
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
        # Reading variables and datasets
        self.keys_var_independent           = self.IN['general_info']['var_independent']
        self.keys_var_dependent             = self.IN['general_info']['var_dependent']
        self.keys_var_dependent_binary      = self.IN['general_info']['var_dependent_binary']
        self.keys_surrogate_models          = self.IN['general_info']['surrogate_models']
        self.keys_surrogate_models_binary   = self.IN['general_info']['surrogate_models_binary']
        self.dataset_training               = csv_to_dict(self.IN['general_info']['dataset_training'][0])
        self.dataset_validation             = csv_to_dict(self.IN['general_info']['dataset_validation'][0])
        self.datasets                       = [self.dataset_training, self.dataset_validation]
        # Get parametrization
        self.key_parametrization = self.IN['general_info']['parametrization'][0]
        if self.key_parametrization == "rotor_and_diffuser_blades":
            from src.parametrization import rotor_and_diffuser_blades
            self.parametrization = rotor_and_diffuser_blades(self.IN)
        else:
            breakpoint()


        # Converged solutions
        if 'Convergence' in self.dataset_training.keys():
            self.dataset_training_converged     = (self.dataset_training['Convergence'] == 1)[0]
            self.dataset_validation_converged   = (self.dataset_validation['Convergence'] == 1)[0]

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
        if self.IN['optimization']["opt_flag"]:
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

        self.surrogate_models, self.best_surrogate_models = self._train_models(
            self.keys_var_dependent,self.keys_surrogate_models)
        self.surrogate_models_binary, self.best_surrogate_models_binary = self._train_models(
            self.keys_var_dependent_binary,self.keys_surrogate_models_binary)
    
    def _train_models(self, keys_var_dependent, keys_surrogate_models):

        # Input variables
        X_training      = self._get_X_from_dataset(self.dataset_training)
        X_validation    = self._get_X_from_dataset(self.dataset_validation)
        surrogate_models        = []
        best_surrogate_models   = []
        # Looping output variables
        for index_var_dep, key_var_dependent in enumerate(keys_var_dependent):
            y_training      = self.dataset_training[key_var_dependent].transpose()
            y_validation    = self.dataset_validation[key_var_dependent].transpose()
            y_surrogate_models = []
            # Looping metamodels
            for index_model_type, model_type in enumerate(keys_surrogate_models):
                # Create and train metamodel
                model = SurrogateFactory.create(model_type, self.IN)
                # Continuous and binary variables:
                if model.continuous and 'Convergence' in self.keys_var_dependent_binary:
                    model.init_model(index_var_dep)
                    _y_training = y_training[self.dataset_training_converged]
                    _X_training = X_training[self.dataset_training_converged]
                    _y_validation = y_validation[self.dataset_validation_converged]
                    _X_validation = X_validation[self.dataset_validation_converged]
                else:
                    model.init_model(index_model_type)
                    _y_training = y_training + 0.0
                    _X_training = X_training + 0.0
                    _y_validation = y_validation + 0.0
                    _X_validation = X_validation + 0.0
                # Optimize hyperparameters to minimize R² w.r.t. validation
                model.optimize_hyperparameters(_X_training, _y_training, _X_validation, _y_validation)
                # Train model
                model.train(_X_training, _y_training)
                # Get model results
                y_predicted_training = np.array([model.predict(_X_training)]).transpose()
                y_predicted_validation = np.array([model.predict(_X_validation)]).transpose()
                summary = dict({'model': model,
                                'training': model.validate(_y_training, y_predicted_training),
                                'validation': model.validate(_y_validation, y_predicted_validation)})
                y_surrogate_models.append(summary)

                # Plot validation results
                self._plot_model(_y_validation, y_predicted_validation,
                                 model, key_var_dependent, summary['validation'], 'validation')
                self._plot_model(_y_training, y_predicted_training,
                                 model, key_var_dependent, summary['training'], 'training')

                # Prints to screen
                res_txt = '\nTraining'
                for key, val in summary['training'].items(): res_txt += f'\t{key}={"%0.2e" % val}'
                res_txt += '\nValidation'
                for key, val in summary['validation'].items(): res_txt += f'\t{key}={"%0.2e"%val}'

                log_print(f"{key_var_dependent}\t{model_type}{res_txt}")
            # Append to surrogate_models
            surrogate_models.append(y_surrogate_models)
            # Get best model for this variable
            y_surrogate_models_r2 = []
            for i, model_type in enumerate(keys_surrogate_models):
                y_surrogate_models_r2.append(y_surrogate_models[i]["validation"]["R2"])
            best_model = y_surrogate_models[int(np.argmax(y_surrogate_models_r2))]['model']
            best_surrogate_models.append(best_model)
            log_print(f"    Selecting {best_model.name}\n\n")
        return surrogate_models, best_surrogate_models

    def _get_X_from_dataset(self, dataset):
        return np.array([dataset[key_var_independent][0] for key_var_independent in
                               self.keys_var_independent]).transpose()

    # -----------------------------------------------------#
    # Parametric optimization
    # -----------------------------------------------------#

    def set_optimization(self):

        # Set bounds
        X_min, X_max = [], []
        for key_var in self.keys_var_independent:
            #X_min.append(np.amin(self.dataset_training[key_var]))
            #X_max.append(np.amax(self.dataset_training[key_var]))
            X_min.append(self.parametrization.dvs_doe_lower_bound[key_var])
            X_max.append(self.parametrization.dvs_doe_upper_bound[key_var])
        self.X_min = np.array(X_min)
        self.X_max = np.array(X_max)

        # Number of objective functions
        opt_obj = self.IN['optimization']['opt_obj']
        self.n_obj = len(opt_obj)
        self.obj_surrogates = {}
        for i in range(self.n_obj):
            obj_key = opt_obj[i]
            self.obj_surrogates[obj_key] = self.best_surrogate_models[self.keys_var_dependent.index(obj_key)]

        # Number of constraints
        opt_constraints = self.IN['optimization']['opt_constraints']
        self.n_constraints = len(opt_constraints)
        self.constraints_surrogates = {}
        for i in range(self.n_constraints):
            constraints_key = opt_constraints[i]
            if constraints_key == "Convergence":
                self.constraints_surrogates[constraints_key] = self.best_surrogate_models_binary[0]
            else:
                self.constraints_surrogates[constraints_key] = self.best_surrogate_models[self.keys_var_dependent.index(constraints_key)]


    def _get_performance(self, DV):

        if len(np.shape(DV)) == 1:    DV = np.array([DV])

        obj_val = {}
        opt_obj = self.IN['optimization']['opt_obj']
        for i in range(self.n_obj):
            obj_key = opt_obj[i]
            obj_val[obj_key]=(float(self.obj_surrogates[obj_key].predict(DV)[0]))

        constraints_val = {}
        opt_constraints = self.IN['optimization']['opt_constraints']
        for i in range(self.n_constraints):
            constraints_key = opt_constraints[i]
            constraints_val[constraints_key]=(float(self.constraints_surrogates[constraints_key].predict(DV)[0]))

        return obj_val, constraints_val

    def _get_objective_and_constraints(self, DV):

        obj_val, constraints_val = self._get_performance(DV)

        obj = []
        for i, obj_option in enumerate(self.IN['optimization']['opt_obj']):
            min_max = self.IN['optimization']['opt_min_max'][i]
            _obj = obj_val[obj_option]
            if 'abs' in min_max:
                _obj = np.abs(_obj)
            if 'max' in min_max:
                _obj = -_obj
            obj.append(float(_obj))

        constraints = []
        opt_constraints = self.IN['optimization']['opt_constraints']
        opt_constraints_baseline = self.IN['optimization']['opt_constraints_baseline']
        opt_constraints_above_below_baseline = self.IN['optimization']['opt_constraints_above_below_baseline']
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
        for i, opt_method in enumerate(self.IN['optimization']['opt_method']):

            self.optimization_algorithm = self.get_optimization_algorithm(algorithm=opt_method)

            out = self.optimization_algorithm(func = self._get_objective_and_constraints,
                                              X_min = self.X_min,
                                              X_max = self.X_max,
                                              n_obj = self.n_obj,
                                              n_constr = self.n_constraints,
                                              IN = self.IN)
            self.X_opt, objective_opt, constraints_opt = out
            self.objective_opt = objective_opt.tolist()
            self.constraints_opt = constraints_opt[0].tolist()

    def post_process(self):

        # Convert optimal vector as a cfd output
        dvs_doe_i_design = {}
        for i, key_var in enumerate(self.keys_var_independent):
            dvs_doe_i_design[key_var] = float(self.X_opt[i])
        self.parametrization.add_dvs_doe_to_cfd(dvs_doe_i_design)

        self.write_csv_file(self.parametrization.dvs_doe, 'output_dvs_doe.csv')
        self.write_csv_file(self.parametrization.dvs_cfd, 'output_dvs_cfd.csv')

    def write_csv_file(self, dvs, filename):

        keys = list(dvs.keys())
        values = list(dvs.values())

        with open(os.path.join(self.directories.outputs_directory, filename), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")  # ← IMPORTANT: use ;
            writer.writerow(keys)

            for row in zip(*values):
                formatted_row = [
                    str(v).replace(".", ",") if isinstance(v, (int, float)) else v
                    for v in row
                ]
                writer.writerow(formatted_row)


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
        if model.continuous:
            y_45 = np.array([np.linspace(min(np.amin(y_model),np.amin(y_predicted)),
                                         max(np.amax(y_model), np.amax(y_predicted)),
                                         2)]).transpose()
            x_lim = [np.amin(y_model), np.amax(y_model)]
            dx, eps = x_lim[1] - x_lim[0], 0.02
            x_lim = [x_lim[0] - dx * eps, x_lim[1] + dx * eps]
        else:
            y_45 = np.array([np.linspace(0,1,2)]).transpose()
            x_lim = np.array([np.linspace(0,1,2)]).transpose()


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
