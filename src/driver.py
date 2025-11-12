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

class Driver():
    def __init__(self, IN, directories):
        self.IN = IN
        self.directories = directories
        # Reading datasets
        self.dataset_training = csv_to_dict(self.IN['dataset_training'])
        self.N_training = np.amax(np.shape(self.dataset_training[self.IN['var_independent'][0]]))
        self.dataset_validation = csv_to_dict(self.IN['dataset_validation'])
        self.N_validation = np.amax(np.shape(self.dataset_validation[self.IN['var_independent'][0]]))
        # Internal datasets
        self.dataset = {}

        # Inlet state
        self.get_inlet_state()
        # Set saturation curve
        self.set_saturation_curve()
        # Set isobars
        self.set_isobars_inlet()

        # Plot
        self.plot_training_and_validation_contours()

        breakpoint()

    # -----------------------------------------------------#
    # Pre processing
    # -----------------------------------------------------#

    def get_inlet_state(self):

        self.dataset_training['s01'] = np.array([[CP.PropsSI('S',
                                                             'T', self.dataset_training['T01'][0, i],
                                                             'P', self.dataset_training['P01'][0, i],
                                                             self.IN['fluid']) for i in range(self.N_training)]])

        self.dataset_validation['s01'] = np.array([[CP.PropsSI('S',
                                                               'T', self.dataset_validation['T01'][0, i],
                                                               'P', self.dataset_validation['P01'][0, i],
                                                               self.IN['fluid']) for i in range(self.N_validation)]])

    def set_saturation_curve(self):
        # Getting saturation curves
        T_crit = CP.PropsSI('Tcrit', self.IN['fluid'])

        # Temperature range (T > 0.9 * T_crit)
        T_min = 0.9 * T_crit
        T_vals = np.linspace(T_min, T_crit, 200)[::-1]

        # Compute saturated liquid (Q=0) and vapor (Q=1) entropies
        s_liq = np.array([CP.PropsSI('S', 'T', T, 'Q', 0, self.IN['fluid']) for T in T_vals])
        s_vap = np.array([CP.PropsSI('S', 'T', T, 'Q', 1, self.IN['fluid']) for T in T_vals])

        # Pressure
        P_liq = np.array([CP.PropsSI('P', 'T', T, 'Q', 0, self.IN['fluid']) for T in T_vals])
        P_vap = np.array([CP.PropsSI('P', 'T', T, 'Q', 0, self.IN['fluid']) for T in T_vals])

        # Saturation
        s_sat = np.array([np.concatenate((s_liq[::-1], s_vap))])
        T_sat = np.array([np.concatenate((T_vals[::-1], T_vals))])
        P_sat = np.array([np.concatenate((P_liq[::-1], P_vap))])

        self.dataset['s_sat'] = s_sat
        self.dataset['T_sat'] = T_sat
        self.dataset['P_sat'] = P_sat

    def set_isobars_inlet(self):

        self.n_isobars = 5
        self.n_entropy = 200
        P_linspace = np.linspace(np.amin(self.dataset_training['P01']),
                                 np.amax(self.dataset_training['P01']), self.n_isobars)
        s_linspace = np.linspace(min(np.amin(self.dataset['s_sat']),np.amin(self.dataset_training['s01'])),
                                 max(np.amax(self.dataset['s_sat']),np.amax(self.dataset_training['s01'])), self.n_entropy)

        P_array, T_array, s_array = [], [], []
        for i in range(self.n_isobars):
            P_array.append(P_linspace[i] * np.ones_like(s_linspace))
            s_array.append(s_linspace)
            T_array.append([CP.PropsSI('T', 'P', P_linspace[i], 'SMASS', s, self.IN['fluid']) for s in s_linspace])

        self.dataset['s_isobars'] = np.array(s_array)
        self.dataset['T_isobars'] = np.array(T_array)
        self.dataset['P_isobars'] = np.array(P_array)

    # -----------------------------------------------------#
    # Plotting
    # -----------------------------------------------------#

    def plot_training_and_validation_contours(self):

        # Plotting datasets
        self._plot_training_and_validation_contours('P01', 'T01','P_sat','T_sat','P_isobars','T_isobars')
        self._plot_training_and_validation_contours('s01', 'T01','s_sat','T_sat','s_isobars','T_isobars')

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
            title = f"Contour_{x_label}_{y_label}_{z_label}"
            filename = title
            plot_SBO(x_label, y_label, title, x_lim=x_lim, y_lim=y_lim,
                     z_label=z_label, X_contour=x_training[:,0], Y_contour=y_training[:,0], Z_contour=z_training[:,0],
                     X0=x_training, Y0=y_training, x0y0_kind="training", labels_list_x0=["training"],
                     X1=x_validation, Y1=y_validation, x1y1_kind="validation", labels_list_x1=["validation"],
                     X2=x_sat, Y2=y_sat, x2y2_kind="xy", labels_list_x2=["saturation"],
                     X3=x_isobars, Y3=y_isobars, x3y3_kind="dotted", labels_list_x3=["p=cte"]+(self.n_isobars-1)*[""],
                     output_directory=self.directories.outputs_directory, loc='upper left',
                     filename=filename)


