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
import CoolProp.CoolProp as CP
import numpy as np
from joblib.memory import extract_first_line

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.utils import *
from common.plotter import *
from src.models import *
from src.thermo import Thermo

class Driver(Thermo):

    def __init__(self, IN, directories):
        super().__init__()
        # Arguments
        self.IN = IN
        self.directories = directories
        # Reading datasets
        self.keys_var_dependent     = self.IN['var_dependent']
        self.keys_var_independent   = self.IN['var_independent']
        self.dataset_training       = csv_to_dict(self.IN['dataset_training'])
        self.dataset_validation     = csv_to_dict(self.IN['dataset_validation'])
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

        # Evaluate on uniform grid
        self.evaluate_best_model_for_PRs()
        self.plot_surrogate_model_contours()

        # Plot contours of trained models



    # -----------------------------------------------------#
    # Pre processing
    # -----------------------------------------------------#

    def pre_process(self):

        self._compute_variables()
        self._get_minimum_and_maximum_values()
        self._set_dataset_grid()
        self._set_datasets_grid_for_PRs()
        self._set_saturation_curve()
        self._set_isobars_inlet()
        self.plot_training_and_validation_contours()

    def _compute_variables(self):

        # Thermo
        for i, dataset in enumerate(self.datasets):
            self.compute_thermo_variables(dataset)

    def _get_minimum_and_maximum_values(self, N_P01=5, N_T01=6):

        # Mimimum and maximum ranges of datasets
        self.P01_min, self.P01_max = self._get_min_max('P01')
        self.T01_min, self.T01_max = self._get_min_max('T01')
        self.PR_rotor_min, self.PR_rotor_max = self._get_min_max('PR_rotor')
        self.PR_stage_min, self.PR_stage_max = self._get_min_max('PR_stage')
        if 'PR_rotor' in self.keys_var_independent:
            self.PR_key = 'PR_rotor'
        elif 'PR_stage' in self.keys_var_independent:
            self.PR_key = 'PR_stage'
        self.PR_min, self.PR_max = self._get_min_max(self.PR_key)

    def _get_min_max(self, var_label):
        x_training = self.dataset_training[var_label]
        x_validation = self.dataset_validation[var_label]
        x_all = np.concatenate((x_training, x_validation),axis=1)
        x_min, x_max = np.amin(x_all), np.amax(x_all)
        return x_min, x_max

    def _set_dataset_grid(self, N_P01=5, N_T01=6):


        # Set mesh grids
        def get_mesh_grid(u_min, u_max, Nu, v_min, v_max, Nv):
            u = np.linspace(u_min, u_max, Nu)
            v = np.linspace(v_min, v_max, Nv)
            u, v = np.meshgrid(u, v, indexing='ij')
            u, v = u.flatten(), v.flatten()
            return np.array([u]), np.array([v])

        P01_grid, T01_grid = get_mesh_grid(self.P01_min, self.P01_max, N_P01,
                                           self.T01_min, self.T01_max, N_T01)
        self.dataset_grid = {'P01':P01_grid, 'T01':T01_grid}
        self.compute_thermo_variables(self.dataset_grid)

    def _set_datasets_grid_for_PRs(self, N_PR=3):

        self.PRs = np.linspace(self.PR_min, self.PR_max, N_PR)
        self.dataset_grid_PRs = []
        for PR in self.PRs:
            self.dataset_grid_PR = {key:val for key,val in self.dataset_grid.items()}
            self.dataset_grid_PR[self.PR_key] = PR*np.ones_like(self.dataset_grid_PR['T01'])
            self.dataset_grid_PRs.append(self.dataset_grid_PR)


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

    def evaluate_best_model_for_PRs(self):

        for i, PR in enumerate(self.PRs):
            dataset_grid_PR = self.dataset_grid_PRs[i]
            X = self._get_X_from_dataset(dataset_grid_PR)
            for j, key_var_dependent in enumerate(self.keys_var_dependent):
                best_model = self.best_surrogate_models[j]
                #log_print(f"Evaluating {best_model} for {key_var_dependent}")
                y_predicted = np.array([best_model.predict(X)])
                #log_print(y_predicted)
                dataset_grid_PR[key_var_dependent] = y_predicted
                #log_print("")

    def evaluate_best_model_on_dataset(self, dataset):

        # Looping output variables
        for i, key_var_dependent in enumerate(self.keys_var_dependent):
            best_model_i = self.best_surrogate_models[i]
            X_grid = np.array([dataset[key_var_independent]
                               for key_var_independent in self.keys_var_independent]).transpose()[0]
            y_grid_i = np.array([best_model_i.predict(X_grid)])
            dataset[key_var_dependent] = y_grid_i

    def _plot_model(self, y_model, y_predicted, model, key_var_dependent, statistics, dataset_name=""):

        key_var_independent = ""
        for key_var in self.keys_var_independent: key_var_independent += f"{key_var}, "
        key_var_independent = f"({key_var_independent[:-2]})"

        x_label = 'Model data'
        y_label = 'Predicted data'

        if dataset_name == "":  _dataset_name = ""
        else:                   _dataset_name = f" ({dataset_name})"
        title = f"Model and predicted data{_dataset_name}\n{key_var_independent} - {key_var_dependent}\nModel: {model.name} (R²={'%0.2f'%statistics['R2']})"
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


    # -----------------------------------------------------#
    # Plotting
    # -----------------------------------------------------#

    def plot_training_and_validation_contours(self):

        # Plotting datasets

        self._plot_training_and_validation_contours('P01', 'T01',
                                                    'P_sat','T_sat',
                                                    'P_isobars','T_isobars')

        self._plot_training_and_validation_contours('s01', 'T01',
                                                    's_sat','T_sat',
                                                    's_isobars','T_isobars')

    def _plot_training_and_validation_contours(self, x_label, y_label, x_label_sat, y_label_sat, x_label_isobars, y_label_isobars):

        # Plotting data
        x_training = self.dataset_training[x_label].transpose()
        y_training = self.dataset_training[y_label].transpose()
        x_validation = self.dataset_validation[x_label].transpose()
        y_validation = self.dataset_validation[y_label].transpose()
        x_sat = self.dataset[x_label_sat].transpose()
        y_sat = self.dataset[y_label_sat].transpose()
        x_isobars = self.dataset[x_label_isobars].transpose()
        y_isobars = self.dataset[y_label_isobars].transpose()
        eps_isobar_L = 0.50
        eps_isobar_U = 0.20

        # X limits
        x_min = min(np.amin(x_training), np.amin(x_validation), np.amin(x_sat))
        x_max = max(np.amax(x_training), np.amax(x_validation), np.amax(x_sat))
        dx = x_max - x_min
        eps = 0.05
        x_lim = [x_min - dx * eps, x_max + dx * eps]
        # Y limits
        y_min = min(np.amin(y_training), np.amin(y_validation), np.amin(y_sat))
        y_max = max(np.amax(y_training), np.amax(y_validation), np.amax(y_sat))
        dy = y_max - y_min
        eps = 0.05
        y_lim = [y_min - dy * eps, y_max + dy * eps]

        for iz, z_label in enumerate(self.keys_var_dependent):
            z_training = self.dataset_training[z_label].transpose()
            z_validation = self.dataset_validation[z_label].transpose()
            _title = f"{x_label} {y_label} {z_label}"
            title = f"Dataset Contour {_title}"
            filename = os.path.join("Dataset Contour", _title.replace(" ", "-"))
            plot_SBO(x_label, y_label, title, x_lim=x_lim, y_lim=y_lim,
                     z_label=z_label, X_contour=x_training[:,0], Y_contour=y_training[:,0], Z_contour=z_training[:,0],
                     X0=x_training, Y0=y_training, x0y0_kind="training", labels_list_x0=["training"],
                     X1=x_validation, Y1=y_validation, x1y1_kind="validation", labels_list_x1=["validation"],
                     X2=x_sat, Y2=y_sat, x2y2_kind="xy", labels_list_x2=["saturation"],
                     X3=x_isobars, Y3=y_isobars, x3y3_kind="dotted", labels_list_x3=["p=cte"]+(self.n_isobars-1)*[""],
                     X4=[x_isobars[int(eps_isobar_L*self.n_entropy),0], x_isobars[int(eps_isobar_U*self.n_entropy),-1]],
                     Y4=[y_isobars[int(eps_isobar_L*self.n_entropy),0], y_isobars[int(eps_isobar_U*self.n_entropy),-1]],
                     x4y4_kind="annotate",
                     labels_list_x4=[f"{'%0.1f' % (self.dataset['P_isobars'][0,0]/10**6)} MPa",
                                     f"{'%0.1f' % (self.dataset['P_isobars'][-1,-1]/10**6)} MPa"],
                     ha_4=['center','right'],
                     va_4=['center','bottom'],
                     output_directory=self.directories.outputs_directory, loc='upper left',
                     filename=filename)

    def plot_surrogate_model_contours(self):

        for i, PR in enumerate(self.PRs):
            dataset_grid_PR = self.dataset_grid_PRs[i]

            # Plotting metamodels
            self._plot_surrogate_model_contours(dataset_grid_PR, 'P01', 'T01',
                                                        'P_sat', 'T_sat',
                                                        'P_isobars', 'T_isobars')

            self._plot_surrogate_model_contours(dataset_grid_PR, 's01', 'T01',
                                                        's_sat', 'T_sat',
                                                        's_isobars', 'T_isobars')

    def _plot_surrogate_model_contours(self, dataset, x_label, y_label, x_label_sat, y_label_sat, x_label_isobars, y_label_isobars):

        # Plotting data
        x_grid = dataset[x_label].transpose()
        y_grid = dataset[y_label].transpose()
        x_sat = self.dataset[x_label_sat].transpose()
        y_sat = self.dataset[y_label_sat].transpose()
        x_isobars = self.dataset[x_label_isobars].transpose()
        y_isobars = self.dataset[y_label_isobars].transpose()
        eps_isobar_L = 0.50
        eps_isobar_U = 0.20

        # X limits
        x_min = min(np.amin(x_grid), np.amin(x_sat))
        x_max = max(np.amax(x_grid), np.amax(x_sat))
        dx = x_max - x_min
        eps = 0.05
        x_lim = [x_min - dx * eps, x_max + dx * eps]
        # Y limits
        y_min = min(np.amin(y_grid), np.amin(y_sat))
        y_max = max(np.amax(y_grid), np.amax(y_sat))
        dy = y_max - y_min
        eps = 0.05
        y_lim = [y_min - dy * eps, y_max + dy * eps]

        # Pressure Ratio
        PR_val = f"{'%0.2f'%float(dataset[self.PR_key][0,0])}"

        for iz, z_label in enumerate(self.keys_var_dependent):
            best_model_name = self.best_surrogate_models[iz].name
            z_grid = dataset[z_label].transpose()
            _title = f"{x_label} {y_label} {z_label}\n{self.PR_key}={PR_val}\nSurrogate {best_model_name}"
            title = f"Contour {_title}"
            filename = os.path.join("Contour",_title.replace(" ","-").replace("\n","-"))

            plot_SBO(x_label, y_label, title, x_lim=x_lim, y_lim=y_lim,
                     z_label=z_label, X_contour=x_grid[:,0], Y_contour=y_grid[:,0], Z_contour=z_grid[:,0],
                     X2=x_sat, Y2=y_sat, x2y2_kind="xy", labels_list_x2=["saturation"],
                     X3=x_isobars, Y3=y_isobars, x3y3_kind="dotted", labels_list_x3=["p=cte"]+(self.n_isobars-1)*[""],
                     X4=[x_isobars[int(eps_isobar_L*self.n_entropy),0], x_isobars[int(eps_isobar_U*self.n_entropy),-1]],
                     Y4=[y_isobars[int(eps_isobar_L*self.n_entropy),0], y_isobars[int(eps_isobar_U*self.n_entropy),-1]],
                     x4y4_kind="annotate",
                     labels_list_x4=[f"{'%0.1f' % (self.dataset['P_isobars'][0,0]/10**6)} MPa",
                                     f"{'%0.1f' % (self.dataset['P_isobars'][-1,-1]/10**6)} MPa"],
                     ha_4=['center','right'],
                     va_4=['center','bottom'],
                     output_directory=self.directories.outputs_directory, loc='upper left',
                     filename=filename)
