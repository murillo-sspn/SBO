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
        self.N_training             = np.amax(np.shape(self.dataset_training[self.keys_var_independent[0]]))
        self.dataset_validation     = csv_to_dict(self.IN['dataset_validation'])
        self.N_validation           = np.amax(np.shape(self.dataset_validation[self.keys_var_independent[0]]))
        # Internal datasets
        self.dataset = {}

        # Pre process
        self.pre_process()

        # Train models
        self.train_models()



    # -----------------------------------------------------#
    # Pre processing
    # -----------------------------------------------------#

    def pre_process(self):

        self._compute_variables()
        self._set_saturation_curve()
        self._set_isobars_inlet()
        self.plot_training_and_validation_contours()

    def _compute_variables(self):

        datasets = [self.dataset_training, self.dataset_validation]
        N_points = [self.N_training, self.N_validation]
        for i, dataset in enumerate(datasets): self._compute_thermo_variables(dataset, N_points[i])

    # -----------------------------------------------------#
    # Training models
    # -----------------------------------------------------#

    def train_models(self):

        # Input variables
        X_training      = np.array([self.dataset_training[key_var_independent][0] for key_var_independent in self.keys_var_independent]).transpose()
        X_validation    = np.array([self.dataset_validation[key_var_independent][0] for key_var_independent in self.keys_var_independent]).transpose()
        surrogate_models = []
        # Looping output variables
        for key_var_dependent in self.keys_var_dependent:
            y_training      = self.dataset_training[key_var_dependent].transpose()
            y_validation    = self.dataset_validation[key_var_dependent].transpose()
            y_surrogate_models = []
            # Looping metamodels
            for model_type in self.IN['surrogate_models']:
                # Create and train metamodel
                model = SurrogateFactory.create(model_type, self.IN)
                model.train(X_training, y_training)

                # Get validation results
                y_predicted = np.array([model.predict(X_validation)]).transpose()
                summary = dict({'model': model,
                                'validation': model.validate(y_validation, y_predicted)})
                y_surrogate_models.append(summary)

                # Plot validation results
                self._plot_model_validation(y_validation, y_predicted, model_type, key_var_dependent)

                # Prints to screen
                res_txt = ''
                for key, val in summary['validation'].items(): res_txt += f'\t{key}={"%0.2e"%val}'
                log_print(f"{key_var_dependent}\t{model_type}{res_txt}")

            surrogate_models.append(y_surrogate_models)

    def _plot_model_validation(self, y_validation, y_predicted, model, key_var_dependent):

        x_label = 'Validation data'
        y_label = 'Predicted data'
        title = f"Validation and predicted data:\n{key_var_dependent} ({model})"
        filename = f"Validation-{key_var_dependent}-{model}"
        y_45 = np.array([np.linspace(min(np.amin(y_validation),np.amin(y_predicted)),
                                     max(np.amax(y_validation), np.amax(y_predicted)),
                                     2)]).transpose()

        plot_SBO(x_label, y_label, title,
                 X0=y_validation, Y0=y_predicted, x0y0_kind="points", labels_list_x0=["data"],
                 X1=y_45, Y1=y_45, x1y1_kind="dotted", labels_list_x1=["y=x"],
                 flag_aspect_ratio=True,
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

        for iz, z_label in enumerate(self.IN['var_dependent']):
            z_training = self.dataset_training[z_label].transpose()
            z_validation = self.dataset_validation[z_label].transpose()
            title = f"Contour {x_label} {y_label} {z_label}"
            filename = title.replace(" ","-")
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


