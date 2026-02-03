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
import copy

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.utils import *
from common.plotter import *
from src.models import *
from src.thermo import Thermo
from src.opt import Optimization

class Driver(Thermo, Optimization):

    def __init__(self, IN, directories):
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

        # Evaluate on uniform grid
        self.evaluate_best_model_for_PRs()
        self.plot_surrogate_model_contours()

        # Set optimization problem
        self.set_optimization()

        # If flag is active...
        if self.IN.get("opt_flag"):
            # Optimize multistage compressor
            self.optimize()

        # Post process solution
        self.post_process()

        # Plot solution
        self.plot_solutions()



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
        self.PR_stage_min_opt = self.IN.get('PR_min_opt', [self.PR_stage_min])[0]
        self.PR_stage_max_opt = self.IN.get('PR_max_opt', [self.PR_stage_max])[0]
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

    # -----------------------------------------------------#
    # Multistage optimization
    # -----------------------------------------------------#

    def set_optimization(self):

        # Initialize optimization parameters
        self._set_optimization_parameters()

        # Get bounds
        self._set_optimization_bounds()

        # Solutions
        self.solutions = {}

        # Get objective function for baseline
        self.objective_baseline, self.constraints_baseline = self._get_objective_and_constraints(self.DV_baseline)
        self.n_constraints = len(self.constraints_baseline)
        self.solutions['baseline'] = Solution(self.DV_baseline, self.objective_baseline, self.constraints_baseline, 0)

    def _set_optimization_parameters(self):
        # Number of stages
        self.n_stages           = int(self.IN.get('n_stages')[0])

        # Inlet state of first stage (Pa and K)
        self.P0_in              = self.IN.get('P0_in')[0]
        self.T0_in              = self.IN.get('T0_in')[0]
        self.h0_in              = self.get_propssi('H', 'P', self.P0_in, 'T', self.T0_in)
        self.s0_in              = self.get_propssi('S', 'P', self.P0_in, 'T', self.T0_in)
        # Inlet temperature at remaining stages (K)
        self.T0_in_remaining    = self.IN.get('T0_in_remaining')[0]
        # Outlet state of the last intercooler (Pa and K)
        self.T0_out             = self.IN.get('T0_out')[0]
        self.P0_out             = self.IN.get('P0_out')[0]
        self.h0_out             = self.get_propssi('H', 'P', self.P0_out, 'T', self.T0_out)
        self.s0_out             = self.get_propssi('S', 'P', self.P0_out, 'T', self.T0_out)
        # Total pressure loss in each intercooler (Pa)
        self.P0_loss_HX         = self.IN.get('P0_loss_HX')[0]

        # PR of each stage if all PR are equal and there are no losses
        self.PR_equal           = (self.P0_out/self.P0_in)**(1/self.n_stages)

        # Inlet pressure of each stage (constant PR)
        self.P0_in_stages = [(self.P0_in*self.PR_equal**i-self.P0_loss_HX*i) for i in range(self.n_stages)]

        # Inlet temperature of each stage
        self.T0_in_stages = [self.T0_in]+[self.T0_in_remaining]*(self.n_stages-1)

        # Design vector
        self.DV_baseline    = self.P0_in_stages[1:] + self.T0_in_stages[1:]
        self.n_DV           = len(self.DV_baseline)
        self.n_DV_over_2    = int(self.n_DV/2)

        # Index of stage isentropic efficiency
        self.eff_surrogate = self.best_surrogate_models[self.keys_var_dependent.index('Efficiency_t_t_stage')]

        def set_surrogate_inputs_selection_strategy():

            if 'P01' in self.keys_var_independent:
                self.get_surrogate_inputs = _get_surrogate_inputs_pressure_temperature_pressureratio
            elif 's01' in self.keys_var_independent:
                self.get_surrogate_inputs = _get_surrogate_inputs_entropy_temperature_pressureratio

        def _get_surrogate_inputs_pressure_temperature_pressureratio(P0, s0, T0, PR):
            return P0, T0, PR

        def _get_surrogate_inputs_entropy_temperature_pressureratio(P0, s0, T0, PR):
            return s0, T0, PR

        set_surrogate_inputs_selection_strategy()

        # Number of objective functions
        self.n_obj = len(self.IN.get('opt_obj'))

    def _set_optimization_bounds(self):

        X_min = []
        X_max = []

        # Pressure - avoiding overlap
        # First to penultimate stages
        for i in range(1, self.n_stages - 1):
            X_min.append((self.P0_in_stages[i - 1] + self.P0_in_stages[i]) / 2)
            X_max.append((self.P0_in_stages[i + 1] + self.P0_in_stages[i]) / 2)
        # Last stage
        X_min.append((self.P0_in_stages[-2] + self.P0_in_stages[-1]) / 2)
        X_max.append((self.P0_out + self.P0_in_stages[-1]) / 2)

        # Temperature - All stages
        for i in range(1, self.n_stages):
            X_min.append(self.IN.get('temp_min')[0])
            X_max.append(self.IN.get('temp_max')[0])

        self.X_min = X_min
        self.X_max = X_max
    
    def _get_performance(self, DV_array, solution=None):
        DV = DV_array if isinstance(DV_array, list) else DV_array.tolist()
        obj = self if solution is None else solution
        # Inlet of stages
        obj.P0_in_stages = [self.P0_in] + DV[:self.n_DV_over_2]
        obj.T0_in_stages = [self.T0_in] + DV[self.n_DV_over_2:]
        obj.h0_in_stages, obj.s0_in_stages = [], []
        for i in range(self.n_stages):
            obj.h0_in_stages.append(self.get_propssi('H', 'P', obj.P0_in_stages[i], 'T', obj.T0_in_stages[i]))
            obj.s0_in_stages.append(self.get_propssi('S', 'P', obj.P0_in_stages[i], 'T', obj.T0_in_stages[i]))

        # Outlet of stages (cnosidering losses)
        obj.P0_out_stages = (np.array(obj.P0_in_stages[1:] + [self.P0_out]) + self.P0_loss_HX).tolist()

        # Pressure ratio of stages
        obj.PR_stages = [obj.P0_out_stages[i] / obj.P0_in_stages[i] for i in range(self.n_stages)]

        # Pressure ratio constraints
        obj.constraints = []
        for i in range(self.n_stages):
            # Input parameters
            PR = obj.PR_stages[i]
            obj.constraints.append(float(self.PR_stage_min_opt - PR))
            obj.constraints.append(float(PR - self.PR_stage_max_opt))

        # Isentropic head
        obj.h0_out_is_stages, obj.dh0_is_stages = [], []
        for i in range(self.n_stages):
            obj.h0_out_is_stages.append(self.get_propssi('H', 'P', obj.P0_out_stages[i], 'S', obj.s0_in_stages[i]))
            obj.dh0_is_stages.append(obj.h0_out_is_stages[i] - obj.h0_in_stages[i])

        # Isentropic efficiency (using metamodel)
        obj.eff_s_stages = []
        obj.penalties = []
        for i in range(self.n_stages):
            # Input parameters
            P0, s0, T0, PR = obj.P0_in_stages[i], obj.s0_in_stages[i], obj.T0_in_stages[i], obj.PR_stages[i]
            if PR >= self.PR_stage_min and PR <= self.PR_stage_max:
                # Penalty factor (=1: no penalty)
                penalty = 1
                # Input vector
                X = np.array([self.get_surrogate_inputs(P0, s0, T0, PR)])
                # Predicting efficiency
                eff_s_stage = (float(self.eff_surrogate.predict(X)[0]))
            else:
                # Penalty factor
                PR_stage_bound = self.PR_stage_max if (PR > self.PR_stage_max) else self.PR_stage_min
                # Error between pressure ratio and training/validation bounds
                PR_error = np.abs(PR - PR_stage_bound)
                # Penalty factor (>1: with penalty)
                penalty = np.exp(PR_error)
                # Input vector
                X = np.array([self.get_surrogate_inputs(P0, s0, T0, PR_stage_bound)])
                # Predicting efficiency at training/validation bound and applying penalty
                eff_s_stage = (float(self.eff_surrogate.predict(X)[0]))
            obj.eff_s_stages.append(eff_s_stage)
            obj.penalties.append(penalty)
        obj.penalty_ave = np.mean(np.array(obj.penalties))

        # Non-isentropic head
        obj.h0_out_stages, obj.dh0_stages, obj.s0_out_stages, obj.ds0_stages, obj.T0_out_stages = [], [], [], [], []
        for i in range(self.n_stages):
            obj.dh0_stages.append(obj.dh0_is_stages[i] / obj.eff_s_stages[i])
            obj.h0_out_stages.append(obj.h0_in_stages[i] + obj.dh0_stages[i])
            obj.s0_out_stages.append(self.get_propssi('S', 'P', obj.P0_out_stages[i], 'H', obj.h0_out_stages[i]))
            obj.T0_out_stages.append(self.get_propssi('T', 'P', obj.P0_out_stages[i], 'H', obj.h0_out_stages[i]))
            obj.ds0_stages.append(obj.s0_out_stages[i] - obj.s0_in_stages[i])
        obj.dh0_multistage = float(np.sum(obj.dh0_stages))

        # Heat transfer in intercoolers
        obj.q_ics = []
        for i in range(self.n_stages):
            if i < self.n_stages - 1:
                obj.q_ics.append(obj.h0_in_stages[i + 1] - obj.h0_out_stages[i])
            elif i == self.n_stages - 1:
                obj.q_ics.append(self.h0_out - obj.h0_out_stages[i])
        obj.q_multistage = float(np.sum(obj.q_ics))

        # Return performance
        obj.performance = {
            'power':            obj.dh0_multistage,
            'heat_transfer':    obj.q_multistage,
            'self.penalty_ave': obj.penalty_ave,
            'constraints':      obj.constraints,
        }

        return obj.performance

    def _get_objective_and_constraints(self, DV, solution=None):

        performance = self._get_performance(DV, solution)
        obj = []
        for i, obj_option in enumerate(self.IN.get('opt_obj')):
            min_max = self.IN.get('opt_min_max')[i]
            _obj = performance[obj_option]
            if 'abs' in min_max:
                _obj = np.abs(_obj)
            if 'max' in min_max:
                _obj = -_obj
            obj.append(float(_obj))
        return obj, performance['constraints']

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
            X_opt = X_opt[0].tolist()
            objective_opt = objective_opt.tolist()
            constraints_opt = constraints_opt[0].tolist()

            # Get objective function reduction
            objective_reduction = []
            for j in range(self.n_obj):
                objective_diff = self.objective_baseline[j]-objective_opt[j]
                if self.objective_baseline[j] != 0:
                    _objective_reduction = objective_diff/self.objective_baseline[j]
                else:
                    _objective_reduction = objective_diff
                objective_reduction.append(_objective_reduction)
                log_print(f"Reduction in objective ({j+1} of {self.n_obj}): {'%0.3f'%(_objective_reduction*100)}%")
            self.solutions[opt_method] = Solution(X_opt, objective_opt, constraints_opt, objective_reduction)

    def post_process(self):
        
        for sol_name, solution in self.solutions.items():

            # Updating variables of self. object for current solution
            solution.objective, solution.constraints = self._get_objective_and_constraints(solution.X, solution=solution)

            solution.T0_io, solution.P0_io = [], []
            solution.h0_io, solution.s0_io = [], []
            for i_stage in range(self.n_stages):
                # Temperature
                solution.T0_io.append(solution.T0_in_stages[i_stage])
                solution.T0_io.append(solution.T0_out_stages[i_stage])
                # Pressure
                solution.P0_io.append(solution.P0_in_stages[i_stage])
                solution.P0_io.append(solution.P0_out_stages[i_stage])
                # Enthalpy
                solution.h0_io.append(solution.h0_in_stages[i_stage])
                solution.h0_io.append(solution.h0_out_stages[i_stage])
                # Entropy
                solution.s0_io.append(solution.s0_in_stages[i_stage])
                solution.s0_io.append(solution.s0_out_stages[i_stage])
            solution.T0_io.append(self.T0_out)
            solution.P0_io.append(self.P0_out)
            solution.h0_io.append(self.h0_out)
            solution.s0_io.append(self.s0_out)

            # Isobars
            self._set_isobars_solution(solution)

            # Compressor and intercooler processes
            def _get_c_hx(_array_io):
                n_io = len(_array_io)
                array_io = np.array([_array_io]).transpose()
                x_c, x_hx = [], []
                for k in range(n_io):

                    if k % 2 == 0:
                        # Compressor inlet
                        _x_c = [array_io[k, 0]]
                        try:
                            # Heat exchanger outlet
                            _x_hx.append(array_io[k, 0])
                            x_hx.append(_x_hx)
                        except:
                            pass
                    else:
                        # Heat exchanger inlet
                        _x_hx = [array_io[k, 0]]
                        try:
                            # Compressor outlet
                            _x_c.append(array_io[k, 0])
                            x_c.append(_x_c)
                        except:
                            pass
                return np.array(x_c).transpose(), np.array(x_hx).transpose()
            # I/O
            solution.T0_c_io, solution.T0_hx_io = _get_c_hx(solution.T0_io)
            solution.P0_c_io, solution.P0_hx_io = _get_c_hx(solution.P0_io)
            solution.h0_c_io, solution.h0_hx_io = _get_c_hx(solution.h0_io)
            solution.s0_c_io, solution.s0_hx_io = _get_c_hx(solution.s0_io)

            # Continuous process (for intercoolers)
            self.n_pts_continuous = 50
            def _get_continuous_process(array_io):
                array_continuous = []
                for i_stage in range(self.n_stages):
                    process = np.linspace(array_io[0,i_stage], array_io[1,i_stage], self.n_pts_continuous)
                    array_continuous.append(process)
                array_continuous = np.array(array_continuous).transpose()
                return array_continuous

            solution.T0_hx = _get_continuous_process(solution.T0_hx_io)
            solution.P0_hx = _get_continuous_process(solution.P0_hx_io)
            solution.h0_hx = np.zeros_like(solution.T0_hx)
            solution.s0_hx = np.zeros_like(solution.T0_hx)

            for i_stage in range(self.n_stages):
                for i in range(self.n_pts_continuous):
                    solution.h0_hx[i, i_stage] = self.get_propssi('H',
                                                                  'T', solution.T0_hx[i, i_stage],
                                                                  'P', solution.P0_hx[i, i_stage])
                    solution.s0_hx[i, i_stage] = self.get_propssi('S',
                                                                  'T', solution.T0_hx[i, i_stage],
                                                                  'P', solution.P0_hx[i, i_stage])

            # Continuous process (for compressors)
            solution.T0_c  = copy.copy(solution.T0_c_io)
            solution.P0_c  = copy.copy(solution.P0_c_io)
            solution.h0_c  = copy.copy(solution.h0_c_io)
            solution.s0_c  = copy.copy(solution.s0_c_io)


    # -----------------------------------------------------#
    # Plotting
    # -----------------------------------------------------#

    def plot_training_and_validation_contours(self):

        # Plotting datasets

        self._plot_training_and_validation_contours('P01', 'T01',
                                                    'P_sat','T_sat',
                                                    'P_isobars_inlet','T_isobars_inlet')

        self._plot_training_and_validation_contours('s01', 'T01',
                                                    's_sat','T_sat',
                                                    's_isobars_inlet','T_isobars_inlet')

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
        y_isobar_L = self.IN.get("T_plot_isobars_annotate", [300])[0]
        id_L = np.argmin(np.abs(y_isobars[:,0]- y_isobar_L))
        y_isobar_U = self.IN.get("T_plot_isobars_annotate", [300])[0]
        id_U = np.argmin(np.abs(y_isobars[:, -1] - y_isobar_U))

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
                     X3=x_isobars, Y3=y_isobars, x3y3_kind="dotted", labels_list_x3=["p=cte"] + (self.n_isobars_inlet - 1) * [""],
                     X4=[x_isobars[id_L, 0], x_isobars[id_U, -1]],
                     Y4=[y_isobars[id_L, 0], y_isobars[id_U, -1]],
                     x4y4_kind="annotate",
                     labels_list_x4=[f"{'%0.1f' % (self.dataset['P_isobars_inlet'][0,0]/10**6)} MPa",
                                     f"{'%0.1f' % (self.dataset['P_isobars_inlet'][-1,-1]/10**6)} MPa"],
                     ha_4=['right', 'center'],
                     va_4=['center', 'center'],
                     output_directory=self.directories.outputs_directory, loc='upper left',
                     filename=filename)

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

    def plot_surrogate_model_contours(self):

        for i, PR in enumerate(self.PRs):
            dataset_grid_PR = self.dataset_grid_PRs[i]

            # Plotting metamodels
            self._plot_surrogate_model_contours(
                dataset_grid_PR, 'P01', 'T01',
                'P_sat', 'T_sat',
                'P_isobars_inlet', 'T_isobars_inlet',
                y_lim=[self.IN.get('T_plot_min')[0], self.IN.get('T_plot_max')[0]]
            )

            self._plot_surrogate_model_contours(
                dataset_grid_PR, 's01', 'T01',
                's_sat', 'T_sat',
                's_isobars_inlet', 'T_isobars_inlet',
                y_lim=[self.IN.get('T_plot_min')[0], self.IN.get('T_plot_max')[0]]
            )

    def _plot_surrogate_model_contours(self, dataset, x_label, y_label,
                                       x_label_sat, y_label_sat,
                                       x_label_isobars, y_label_isobars,
                                       x_lim=None, y_lim=None):

        # Plotting data
        x_grid = dataset[x_label].transpose()
        y_grid = dataset[y_label].transpose()
        x_sat = self.dataset[x_label_sat].transpose()
        y_sat = self.dataset[y_label_sat].transpose()
        x_isobars = self.dataset[x_label_isobars].transpose()
        y_isobars = self.dataset[y_label_isobars].transpose()
        y_isobar_L = self.IN.get("T_plot_isobars_inlet_annotate", [300])[0]
        id_L = np.argmin(np.abs(y_isobars[:, 0] - y_isobar_L))
        y_isobar_U = self.IN.get("T_plot_isobars_inlet_annotate", [300])[0]
        id_U = np.argmin(np.abs(y_isobars[:, -1] - y_isobar_U))

        # X limits
        if not(isinstance(x_lim, list)):
            x_min = min(np.amin(x_grid), np.amin(x_sat))
            x_max = max(np.amax(x_grid), np.amax(x_sat))
            dx = x_max - x_min
            eps = 0
            x_lim = [x_min - dx * eps, x_max + dx * eps]
        # Y limits
        if not(isinstance(y_lim, list)):
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
                     X3=x_isobars, Y3=y_isobars, x3y3_kind="dotted", labels_list_x3=["p=cte"] + (self.n_isobars_inlet - 1) * [""],
                     X4=[x_isobars[id_L, 0], x_isobars[id_U, -1]],
                     Y4=[y_isobars[id_L, 0], y_isobars[id_U, -1]],
                     x4y4_kind="annotate",
                     labels_list_x4=[f"{'%0.1f' % (self.dataset['P_isobars_inlet'][0,0]/10**6)} MPa",
                                     f"{'%0.1f' % (self.dataset['P_isobars_inlet'][-1,-1]/10**6)} MPa"],
                     ha_4=['right', 'center'],
                     va_4=['center', 'center'],
                     output_directory=self.directories.outputs_directory, loc='upper left',
                     filename=filename)

    def plot_solutions(self):

        # -----------------------------------------------------#
        # Separate
        # -----------------------------------------------------#
        for sol_name, solution in self.solutions.items():
            # Get inlet and outlet states

            self._plot_solution(sol_name, 's01', 'T01',
                                solution.s0_io, solution.T0_io,
                                solution.s0_c, solution.T0_c, solution.s0_hx, solution.T0_hx,
                                's_sat', 'T_sat',
                                solution.s_isobars, solution.T_isobars, solution.P_isobars, solution.n_isobars,
                                y_lim=[self.IN.get('T_plot_min')[0], self.IN.get('T_plot_max')[0]])

        # -----------------------------------------------------#
        # Baseline and optimized
        # -----------------------------------------------------#

        # -----------------------------------------------------#
        # Baseline and all optimized
        # -----------------------------------------------------#
        pass

    def _plot_solution(self, sol_name, x_label, y_label,
                       x_io, y_io, x_c, y_c, x_hx, y_hx,
                       x_label_sat, y_label_sat,
                       x_isobars, y_isobars, p_isobars, n_isobars,
                       x_lim=None, y_lim=None):



        # In/out states
        n_io = len(x_io)
        _x_io = np.array([x_io]).transpose()
        _y_io = np.array([y_io]).transpose()

        # Saturation curves
        x_sat = self.dataset[x_label_sat].transpose()
        y_sat = self.dataset[y_label_sat].transpose()

        # Isobars
        x_isobars = x_isobars.transpose()
        y_isobars = y_isobars.transpose()
        x_isobars_annotate, y_isobars_annotate, id_isobars_annotate, label_isobars_annotate = [], [], [], []
        n_isobars_annotate = int(np.ceil(n_isobars/2))
        for i_isobar in range(n_isobars_annotate):
            y_annotate = self.IN.get("T_plot_isobars_solution_annotate", [375])[0] - (-1)**i_isobar*10
            y_isobars_annotate.append(y_annotate)
            index = np.argmin(np.abs(y_isobars[:, 2*i_isobar] - y_annotate))
            id_isobars_annotate.append(index)
            x_annotate = x_isobars[index, 2*i_isobar]
            x_isobars_annotate.append(x_annotate)
            label_isobars_annotate.append(f"{'%0.1f' % (p_isobars[2*i_isobar, 0] / 10 ** 6)} MPa")

        # X limits
        if not(isinstance(x_lim, list)):
            x_min = min(np.amin(x_io), np.amin(x_sat))
            x_max = max(np.amax(x_io), np.amax(x_sat))
            dx = x_max - x_min
            eps = 0.10
            x_lim = [x_min - dx * eps, x_max + dx * eps]
        # Y limits
        if not(isinstance(y_lim, list)):
            y_min = min(np.amin(y_io), np.amin(y_sat))
            y_max = max(np.amax(y_io), np.amax(y_sat))
            dy = y_max - y_min
            eps = 0.05
            y_lim = [y_min - dy * eps, y_max + dy * eps]

        _title = f"{x_label} {y_label}\nSolution {sol_name}"
        title = f"Multistage {_title}"
        filename = os.path.join("Solution", _title.replace(" ", "-").replace("\n", "-"))
        plot_SBO(x_label, y_label, title, x_lim=x_lim, y_lim=y_lim,
                 X3=x_c,        Y3=y_c,         x3y3_kind="arrow",      labels_list_x3=["compression"] + (self.n_stages - 1) * [""], colors_3=['red']*self.n_stages,
                 X4=x_hx,       Y4=y_hx,        x4y4_kind="arrow",      labels_list_x4=["intercooling"] + (self.n_stages - 1) * [""], colors_4=['blue']*self.n_stages,
                 X5=_x_io,      Y5=_y_io,       x5y5_kind="training",   labels_list_x5=["states"],
                 X6=np.array([_x_io[0], _x_io[-1]]),
                 Y6=np.array([_y_io[0], _y_io[-1]]),
                 x6y6_kind="validation",        labels_list_x6=["i/o"],
                 X0=x_sat,      Y0=y_sat,       x0y0_kind="xy",         labels_list_x0=["saturation"],
                 X1=x_isobars,  Y1=y_isobars,   x1y1_kind="dotted",     labels_list_x1=["p=cte"] + (n_isobars - 1) * [""],
                 X2=x_isobars_annotate,
                 Y2=y_isobars_annotate,
                 x2y2_kind="annotate",
                 labels_list_x2=label_isobars_annotate,
                 ha_2=['center']*n_isobars_annotate,
                 va_2=['center']*n_isobars_annotate,
                 output_directory=self.directories.outputs_directory, loc='upper left',
                 filename=filename)

class Solution:

    def __init__(self, X, objective, constraints, objective_reduction):

        self.X = X
        self.objective = objective
        self.constraints = constraints
        self.objective_reduction = objective_reduction
