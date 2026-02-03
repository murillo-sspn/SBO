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
import CoolProp.CoolProp as CP

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.logger import *

class Thermo():

    def _set_saturation_curve(self):
        # Getting saturation curves
        T_crit = CP.PropsSI('Tcrit', self.IN['fluid'][0])

        # Temperature range (T > 0.9 * T_crit)
        T_min = self.IN.get("T_plot_min", [0.85 * T_crit])[0]
        T_vals = np.linspace(T_min, T_crit, 200)[::-1]

        # Compute saturated liquid (Q=0) and vapor (Q=1) entropies
        s_liq = np.array([self.get_propssi('S', 'T', T, 'Q', 0) for T in T_vals])
        s_vap = np.array([self.get_propssi('S', 'T', T, 'Q', 1) for T in T_vals])

        # Pressure
        P_liq = np.array([self.get_propssi('P', 'T', T, 'Q', 0) for T in T_vals])
        P_vap = np.array([self.get_propssi('P', 'T', T, 'Q', 1) for T in T_vals])

        # Saturation
        s_sat = np.array([np.concatenate((s_liq[::-1], s_vap))])
        T_sat = np.array([np.concatenate((T_vals[::-1], T_vals))])
        P_sat = np.array([np.concatenate((P_liq[::-1], P_vap))])

        self.dataset['s_sat'] = s_sat
        self.dataset['T_sat'] = T_sat
        self.dataset['P_sat'] = P_sat

    def _set_isobars_inlet(self):

        self.n_isobars_inlet = 5
        P_linspace = np.linspace(np.amin(self.dataset_training['P01']),
                                 np.amax(self.dataset_training['P01']), self.n_isobars_inlet)

        self.n_entropy = 300
        s_min = min(np.amin(self.dataset['s_sat']),np.amin(self.dataset_training['s01']))
        s_max = max(np.amax(self.dataset['s_sat']),np.amax(self.dataset_training['s01']))
        ds = s_max - s_min
        eps = 0.20
        s_lim = [s_min - ds * eps, s_max + ds * eps]
        self.s_linspace = np.linspace(s_lim[0], s_lim[1], self.n_entropy)

        P_array, T_array, s_array = self._get_isobars(self.s_linspace, P_linspace)

        self.dataset['s_isobars_inlet'] = np.array(s_array)
        self.dataset['T_isobars_inlet'] = np.array(T_array)
        self.dataset['P_isobars_inlet'] = np.array(P_array)

    def _set_isobars_solution(self, solution):

        P_linspace = np.array(solution.P0_io)

        P_array, T_array, s_array = self._get_isobars(self.s_linspace, P_linspace)

        solution.s_isobars = np.array(s_array)
        solution.T_isobars = np.array(T_array)
        solution.P_isobars = np.array(P_array)
        solution.n_isobars = len(P_linspace)

    def _get_isobars(self, s_isobars, P_isobars):
        P_array, T_array, s_array = [], [], []
        for i in range(len(P_isobars)):
            P_array.append(P_isobars[i] * np.ones_like(s_isobars))
            s_array.append(s_isobars)
            T_list = []
            for s in s_isobars:
                try:
                    _T = self.get_propssi('T', 'P', P_isobars[i], 'SMASS', s)
                except:
                    _T = 0
                T_list.append(_T)
            T_array.append(T_list)
        return P_array, T_array, s_array

    def compute_thermo_variables(self, dataset):

        key_0 = [key for key in dataset.keys()][0]
        N_points = np.shape(dataset[key_0])[1]

        # Entropy at inlet
        if 'P01' in dataset.keys() and 'T01' in dataset.keys():
            dataset['s01'] = np.array([[self.get_propssi('S',
                                                   'T', dataset['T01'][0, i],
                                                   'P', dataset['P01'][0, i]) for i in range(N_points)]])

        # Pressure at inlet
        elif 's01' in dataset.keys() and 'T01' in dataset.keys():
            dataset['P01'] = np.array([[self.get_propssi('S',
                                                   'T', dataset['T01'][0, i],
                                                   'S', dataset['s01'][0, i]) for i in range(N_points)]])

        # Total pressure ratio
        if 'PR_rotor' in dataset.keys():
            dataset['P02'] = dataset['P01'] * dataset['PR_rotor']
            dataset['P04'] = dataset['P01'] * dataset['PR_stage']

        # Properties @ rotor LE
        positions = [1,2,3,4]
        props_dict = {'h0': 'H',
                      's0': 'S',
                      'a0': 'SPEED_OF_SOUND',
                      'Z0': 'Z'}

        for position in positions:
            dataset_keys = [f"{key}{position}" for key in props_dict.keys()]
            propssi_out = [val for val in props_dict.values()]
            self._set_prop_T0_P0(dataset_keys, propssi_out, f"T0{position}", f"P0{position}", dataset, N_points)

        # Total enthalpy
        if 'h01' in dataset.keys() and 'h02' in dataset.keys():
            dataset['delta_h0'] = dataset['h02'] - dataset['h01']

    def _set_prop_T0_P0(self, dataset_keys, propssi_out, propssi_T0, propssi_P0, dataset, N_points, verbose=False):


        for i, key in enumerate(dataset_keys):
            if not(key in dataset.keys()):
                try:
                    dataset[key] = np.array([[self.get_propssi(propssi_out[i],
                                           'T', dataset[propssi_T0][0, i],
                                           'P', dataset[propssi_P0][0, i]) for i in range(N_points)]])
                except:
                    if verbose: log_print(f"    Could not get property {propssi_out[i]} for {propssi_T0} and {propssi_P0}")

    def get_propssi(self, prop_out, prop_in_1, prop_in_1_val, prop_in_2, prop_in_2_val):
        return CP.PropsSI(prop_out, prop_in_1, prop_in_1_val, prop_in_2, prop_in_2_val, self.IN['fluid'][0])