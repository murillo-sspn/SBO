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
#   Online SBO: the metamodels are updated after each #
#   CFD run. Each generation OPT/GEN_XXXX runs:       #
#     pre_process   copy datasets and ranges          #
#     train_models  select/train models (models.cfg)  #
#     optimize      optimum of the models, new ranges #
#     run_cfd       simulate the optimum with CFX     #
#     post_process  add CFD result to training_new    #
#######################################################

#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#
import csv
import json
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, balanced_accuracy_score, confusion_matrix
#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.utils import *
from common.config import read_user_input
from common.plotter import *
from src.models import *
from src.opt import Optimization
from src.parametrization import rotor_and_diffuser_blades
from src.cfx_runner import WorkbenchSettings, CFXRunner

def print_banner():
    log_print("#######################################################")
    log_print("#                                                     #")
    log_print("#            Surrogate Based Optimization             #")
    log_print("#                  Online SBO (CFX)                   #")
    log_print("#                                                     #")
    log_print("#######################################################")
    log_print("")

def print_header(title):
    log_print("")
    log_print("")
    log_print("=======================================================")
    log_print(f"   {title}")
    log_print("=======================================================")

def print_section(title):
    log_print("")
    log_print(f"--- {title} " + "-" * max(3, 50 - len(title)))
    log_print("")

class Driver(Optimization):

    def __init__(self, IN, directories):
        print_banner()
        super().__init__()
        # Arguments
        self.IN = IN
        self.directories = directories
        self.opt_directory = directories.opt_directory
        # Reading variables and datasets
        self.keys_var_independent           = self.IN['general_info']['var_independent']
        self.keys_var_dependent             = self.IN['general_info']['var_dependent']
        self.keys_var_dependent_binary      = self.IN['general_info']['var_dependent_binary']
        self.keys_surrogate_models          = self.IN['general_info']['surrogate_models']
        self.keys_surrogate_models_binary   = self.IN['general_info']['surrogate_models_binary']
        self.dataset_training_file          = os.path.join(directories.case_directory, self.IN['general_info']['dataset_training'][0])
        self.dataset_validation_file        = os.path.join(directories.case_directory, self.IN['general_info']['dataset_validation'][0])
        # Validation of the metamodels
        self.validation             = self.IN['general_info']['validation'][0]
        self.validation_binary      = self.IN['general_info'].get('validation_binary', [self.validation])[0]
        self.cv_folds               = self.IN['general_info'].get('cv_folds', [5])[0]  # only for cross_validation
        self.r2_min                 = self.IN['general_info']['r2_min'][0]
        self.max_retune_attempts    = self.IN['general_info']['max_retune_attempts'][0]
        # Online settings
        self.max_generations        = self.IN['online']['max_generations'][0]
        self.selection_every        = self.IN['online']['selection_every'][0]
        self.bound_tolerance        = self.IN['online']['bound_tolerance'][0]
        self.n_local                = self.IN['online'].get('n_local', [15])[0]
        self.infill                 = self.IN['online']['infill'][0]
        self.range_update           = self.IN['online']['range_update'][0]
        for validation in [self.validation, self.validation_binary]:
            if validation not in ["holdout", "cross_validation"]:
                raise Exception(f"Incorrect option: validation = {validation}")
        # Get parametrization
        self.key_parametrization = self.IN['general_info']['parametrization'][0]
        if self.key_parametrization == "rotor_and_diffuser_blades":
            self.parametrization = rotor_and_diffuser_blades(self.IN)
        else:
            raise Exception(f"Parametrization not available: {self.key_parametrization}")
        self.read_range_expansion()
        # CFD
        self.wb_settings = WorkbenchSettings(directories.cfg_filepath, self.IN)
        self.cfx_runner = CFXRunner(self.wb_settings, shared_project_dir=os.path.join(self.opt_directory, "Workbench"))
        if self.wb_settings.run_workbench:
            self.wb_settings.validate(list(self.parametrization.dvs_cfd.keys()), self.keys_var_dependent)
        # Copy of the config file used at the start
        if not os.path.isfile(os.path.join(self.opt_directory, "config_used.cfg")):
            shutil.copy(directories.cfg_filepath, os.path.join(self.opt_directory, "config_used.cfg"))

        # Run generations
        self.run()

    # -----------------------------------------------------#
    # Generations
    # -----------------------------------------------------#

    def run(self):

        gen = self.get_last_generation()
        while gen < self.max_generations:
            if os.path.isfile(os.path.join(self.opt_directory, "STOP")):
                print_header("STOP file found in OPT: stopping (delete it to continue)")
                return
            if not self.run_generation(gen):
                return
            gen += 1
        print_header(f"Maximum number of generations reached ({self.max_generations})")

    def run_generation(self, gen):

        # Returns False if the loop must stop
        self.gen = gen
        self.gen_dir = os.path.join(self.opt_directory, f"GEN_{gen:04d}")
        os.makedirs(self.gen_dir, exist_ok=True)
        status = self.read_status()
        print_header(f"GEN_{gen:04d}")
        if len(status) > 0:
            log_print(f"Resuming. Completed stages: {', '.join(status)}")

        if "pre_process" not in status:
            self.pre_process()
            self.set_status("pre_process")
        self.dataset_training   = csv_to_dict(self._file("training.csv"))
        self.dataset_validation = csv_to_dict(self._file("validation.csv"))

        if "train_models" not in status:
            self.train_models()
            self.set_status("train_models")

        if "optimize" not in status:
            if "train_models" in status:
                self.load_models()
            self.set_optimization()
            self.optimize()
            self.set_status("optimize")

        if "cfd_done" not in status:
            if not self.run_cfd(resume=("cfd_submitted" in status)):
                return False
            self.set_status("cfd_done")

        if "post_process" not in status:
            self.post_process()
            self.set_status("post_process")
        return True

    def get_last_generation(self):

        gen = 0
        while os.path.isdir(os.path.join(self.opt_directory, f"GEN_{gen + 1:04d}")):
            gen += 1
        self.gen_dir = os.path.join(self.opt_directory, f"GEN_{gen:04d}")
        if "post_process" in self.read_status():
            gen += 1
        return gen

    def read_status(self):

        filename = os.path.join(self.gen_dir, "status.json")
        if not os.path.isfile(filename):
            return []
        with open(filename, "r") as f:
            return json.load(f)["completed"]

    def set_status(self, stage):

        status = self.read_status()
        status.append(stage)
        with open(os.path.join(self.gen_dir, "status.json"), "w") as f:
            json.dump({"completed": status}, f, indent=2)

    def _file(self, filename, gen=None):

        if gen is None:
            return os.path.join(self.gen_dir, filename)
        return os.path.join(self.opt_directory, f"GEN_{gen:04d}", filename)

    # -----------------------------------------------------#
    # Pre processing
    # -----------------------------------------------------#

    def pre_process(self):

        if self.gen == 0:
            shutil.copy(self.dataset_training_file, self._file("training.csv"))
            shutil.copy(self.dataset_validation_file, self._file("validation.csv"))
            X_min, X_max = [], []
            for key_var in self.keys_var_independent:
                X_min.append(self.parametrization.dvs_doe_lower_bound[key_var])
                X_max.append(self.parametrization.dvs_doe_upper_bound[key_var])
            self.write_ranges("ranges.cfg", X_min, X_max)
        else:
            shutil.copy(self._file("training_new.csv", self.gen - 1), self._file("training.csv"))
            shutil.copy(self._file("validation_new.csv", self.gen - 1), self._file("validation.csv"))
            shutil.copy(self._file("ranges_new.cfg", self.gen - 1), self._file("ranges.cfg"))

    # -----------------------------------------------------#
    # Training models
    # -----------------------------------------------------#

    def train_models(self):

        # Select models (every selection_every generations) or use models.cfg of the previous generation
        previous_models_cfg = self._file("models.cfg", self.gen - 1)
        flag_selection = (self.gen % self.selection_every == 0) or not os.path.isfile(previous_models_cfg)
        if flag_selection:
            print_section(f"Metamodels: selection")
        else:
            print_section(f"Metamodels: from GEN_{self.gen - 1:04d}/models.cfg")
            shutil.copy(previous_models_cfg, self._file("models.cfg"))
            self.models_cfg = self.read_models_cfg()
        log_print(f"  Validation: {self.validation} (continuous), {self.validation_binary} (binary)")
        log_print("")

        self.models = {}
        self.models_info = {}
        self._train_models(self.keys_var_dependent, self.keys_surrogate_models, False, flag_selection)
        self._train_models(self.keys_var_dependent_binary, self.keys_surrogate_models_binary, True, flag_selection)
        self.write_models_cfg()
        for key_var_dependent in self.keys_var_dependent_binary:
            self.write_classifier_report(key_var_dependent)

        # Plot results
        for key_var_dependent in self.keys_var_dependent:
            for dataset_name, dataset in [("training", self.dataset_training), ("validation", self.dataset_validation)]:
                X, y = self._get_Xy(dataset, key_var_dependent, False)
                if len(y) > 1:
                    model = self.models[key_var_dependent]
                    y_predicted = np.array([model.predict(X)]).transpose()
                    self._plot_model(y, y_predicted, model, key_var_dependent, r2_score(y, y_predicted), dataset_name)

    def _train_models(self, keys_var_dependent, keys_surrogate_models, binary, flag_selection):

        score_name = "balanced accuracy" if binary else "R2"
        for index_var_dep, key_var_dependent in enumerate(keys_var_dependent):
            log_print(f"  {key_var_dependent}")
            if flag_selection:
                info = self._select_model(key_var_dependent, index_var_dep, keys_surrogate_models, binary)
            else:
                info = self.models_cfg[key_var_dependent]
                info["score"], confusion = self._get_score(info["model"], info["index"], info["hyperparameters"],
                                                           key_var_dependent, binary)
                info["source"] = "previous_generation"
                info["details"] = confusion
                self._print_score(info["model"], info["hyperparameters"], score_name, info["score"], confusion)

            # Retune hyperparameters while R2 < r2_min
            attempt = 0
            while not binary and info["score"] < self.r2_min and attempt < self.max_retune_attempts:
                attempt += 1
                log_print(f"    {score_name} = {info['score']:.4f} < r2_min = {self.r2_min}: retuning hyperparameters "
                          f"({attempt}/{self.max_retune_attempts})")
                new_info = self._select_model(key_var_dependent, index_var_dep, keys_surrogate_models, binary,
                                              flag_tuning=True)
                if new_info["score"] > info["score"]:
                    info = new_info
                    info["source"] = f"retuned_{attempt}"
            if not binary and info["score"] < self.r2_min:
                log_print(f"    WARNING: {score_name} still below r2_min = {self.r2_min}, using best model found")

            # Train model with training data
            self.models[key_var_dependent] = self._get_trained_model(info, key_var_dependent, binary)
            self.models_info[key_var_dependent] = info
            log_print(f"    -> Selecting {info['model']} ({info['source']}), {score_name} = {info['score']:.4f}")
            log_print("")

    def _select_model(self, key_var_dependent, index_var_dep, keys_surrogate_models, binary, flag_tuning=False):

        score_name = "balanced accuracy" if binary else "R2"
        X_training, y_training = self._get_Xy(self.dataset_training, key_var_dependent, binary)
        X_validation, y_validation = self._get_Xy(self.dataset_validation, key_var_dependent, binary)
        best_info = None
        for index_model_type, model_type in enumerate(keys_surrogate_models):
            index = index_model_type if binary else index_var_dep
            model = self._get_model(model_type, index)
            if model_type == "nn":
                if flag_tuning:
                    model.nn_opt_hyperparameters = True

                # Hyperparameters optimized with the same score used to select the models (holdout or CV)
                def func_score(hidden_layers, activation):
                    hyperparameters = model.get_hyperparameters()
                    hyperparameters["nn_hidden_layers"] = hidden_layers
                    hyperparameters["nn_activation"] = activation
                    score, errors = self._get_score(model_type, index, hyperparameters, key_var_dependent, binary)
                    return {"R2": score, "max_error": errors["max_error"], "p95_error": errors["p95_error"],
                            "RMSE": errors["RMSE"]}

                # Stops as soon as a combination reaches r2_min
                model.optimize_hyperparameters(X_training, y_training, X_validation, y_validation,
                                               func_score=func_score, r2_target=self.r2_min)
            hyperparameters = model.get_hyperparameters()
            try:
                score, confusion = self._get_score(model_type, index, hyperparameters, key_var_dependent, binary)
            except Exception as error:
                log_print(f"    {self._get_description(model_type, hyperparameters):<34}failed and was skipped: {error}")
                continue
            self._print_score(model_type, hyperparameters, score_name, score, confusion)
            if best_info is None or score > best_info["score"]:
                best_info = {"model": model_type, "index": index, "hyperparameters": hyperparameters,
                             "score": score, "source": "selection", "details": confusion}
        if best_info is None:
            raise Exception(f"No metamodel could be evaluated for {key_var_dependent}")
        return best_info

    def _get_score(self, model_type, index, hyperparameters, key_var_dependent, binary):

        # R2 (continuous) or balanced accuracy (binary) with validation data (holdout) or k-fold cross-validation
        validation = self.validation_binary if binary else self.validation
        X, y_true, y_predicted = self._get_predictions(model_type, index, hyperparameters, key_var_dependent,
                                                       binary, validation)
        return self._get_metrics(y_true, y_predicted, binary)

    def _get_predictions(self, model_type, index, hyperparameters, key_var_dependent, binary, validation):

        # Predictions of data not used for training: validation data (holdout) or k-fold cross-validation
        X_training, y_training = self._get_Xy(self.dataset_training, key_var_dependent, binary)
        X_validation, y_validation = self._get_Xy(self.dataset_validation, key_var_dependent, binary)
        if validation == "holdout":
            model = self._get_model(model_type, index, hyperparameters)
            model.train(X_training, y_training)
            X = X_validation
            y_true = y_validation[:, 0]
            y_predicted = model.predict(X_validation)
        else:
            X = np.vstack((X_training, X_validation))
            y = np.vstack((y_training, y_validation))
            y_true = y[:, 0]
            y_predicted = np.zeros(len(y_true))
            if binary:
                n_splits = min(self.cv_folds, int(min(np.sum(y_true == 0), np.sum(y_true == 1))))
                folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y_true)
            else:
                folds = KFold(n_splits=self.cv_folds, shuffle=True, random_state=0).split(X)
            for index_train, index_test in folds:
                model = self._get_model(model_type, index, hyperparameters)
                model.train(X[index_train], y[index_train])
                y_predicted[index_test] = model.predict(X[index_test])
        return X, y_true, y_predicted

    def _get_metrics(self, y_true, y_predicted, binary):

        if binary:
            y_predicted = np.round(y_predicted)
            confusion = confusion_matrix(y_true, y_predicted, labels=[0, 1])
            if len(np.unique(y_true)) < 2:
                return np.nan, confusion  # balanced accuracy needs both classes
            return balanced_accuracy_score(y_true, y_predicted), confusion
        errors = np.abs(y_true - y_predicted)
        # Absolute errors: maximum, 95th percentile (95% of the points have a smaller error) and RMSE
        # bias: mean of predicted - data (> 0: the model overestimates)
        return r2_score(y_true, y_predicted), {"max_error": np.amax(errors), "p95_error": np.percentile(errors, 95),
                                               "RMSE": np.sqrt(np.mean(errors ** 2)),
                                               "bias": np.mean(y_predicted - y_true)}

    def _get_description(self, model_type, hyperparameters):

        description = model_type
        if "svm_model" in hyperparameters:
            description += f" ({hyperparameters['svm_model']})"
        if "nn_hidden_layers" in hyperparameters:
            description += f" {hyperparameters['nn_hidden_layers']} {hyperparameters['nn_activation']}"
        return description

    def _print_score(self, model_type, hyperparameters, score_name, score, details):

        # details: confusion matrix (binary) or errors (continuous)
        line = f"    {self._get_description(model_type, hyperparameters):<34}{score_name} = {score:.4f}"
        if isinstance(details, dict):
            line += (f"    max error = {details['max_error']:.4g}    P95 error = {details['p95_error']:.4g}"
                     f"    RMSE = {details['RMSE']:.4g}")
        else:
            line += f"    [[TN FP] [FN TP]] = {details.tolist()}"
        log_print(line)

    def _get_model(self, model_type, index, hyperparameters=None):

        model = SurrogateFactory.create(model_type, self.IN)
        model.init_model(index)
        if hyperparameters:
            model.set_hyperparameters(hyperparameters)
        return model

    def _get_trained_model(self, info, key_var_dependent, binary):

        X_training, y_training = self._get_Xy(self.dataset_training, key_var_dependent, binary)
        model = self._get_model(info["model"], info["index"], info["hyperparameters"])
        model.train(X_training, y_training)
        return model

    def _get_X_from_dataset(self, dataset):
        return np.array([dataset[key_var_independent][0] for key_var_independent in
                               self.keys_var_independent]).transpose()

    def _get_Xy(self, dataset, key_var_dependent, binary):

        # Continuous variables: only converged solutions
        X = self._get_X_from_dataset(dataset)
        y = dataset[key_var_dependent].transpose()
        mask = ~np.isnan(y[:, 0])
        if not binary and 'Convergence' in dataset.keys():
            mask = mask & (dataset['Convergence'][0] == 1)
        return X[mask], y[mask]

    def load_models(self):

        # Restart: train the models of models.cfg again
        self.models_cfg = self.read_models_cfg()
        self.models_info = self.models_cfg
        self.models = {}
        for key_var_dependent, info in self.models_cfg.items():
            binary = key_var_dependent in self.keys_var_dependent_binary
            self.models[key_var_dependent] = self._get_trained_model(info, key_var_dependent, binary)

    def write_classifier_report(self, key_var_dependent):

        # Model/<model>-classifier-<variable>.txt: confusion matrix and metrics of the binary model
        # (data not used for training: validation data or cross-validation)
        info = self.models_info[key_var_dependent]
        [[TN, FP], [FN, TP]] = np.array(info["details"]).tolist()

        def ratio(a, b):
            return a / b if b > 0 else np.nan

        recall_0 = ratio(TN, TN + FP)
        recall_1 = ratio(TP, TP + FN)
        precision_1 = ratio(TP, TP + FP)
        precision_0 = ratio(TN, TN + FN)
        accuracy = ratio(TN + TP, TN + FP + FN + TP)
        balanced_accuracy = (recall_0 + recall_1) / 2
        MCC = ratio(TP * TN - FP * FN, np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))))

        lines = [f"Classifier: {key_var_dependent} (GEN_{self.gen:04d})",
                 f"Model: {self._get_description(info['model'], info['hyperparameters'])} ({info['source']})",
                 f"Validation: {self.validation_binary}" + (f" ({self.cv_folds} folds)" if self.validation_binary == "cross_validation" else ""),
                 "",
                 "Confusion matrix (rows: data, columns: model)",
                 f"{'':<16}{'predicted 0':>14}{'predicted 1':>14}",
                 f"{'data 0':<16}{TN:>14}{FP:>14}",
                 f"{'data 1':<16}{FN:>14}{TP:>14}",
                 "",
                 f"{'balanced accuracy':<36}{balanced_accuracy:.4f}",
                 f"{'accuracy':<36}{accuracy:.4f}",
                 f"{'recall 0 (specificity)':<36}{recall_0:.4f}    TN/(TN+FP): data 0 predicted as 0",
                 f"{'recall 1 (sensitivity)':<36}{recall_1:.4f}    TP/(TP+FN): data 1 predicted as 1",
                 f"{'precision 1':<36}{precision_1:.4f}    TP/(TP+FP): predicted 1 that are 1",
                 f"{'precision 0':<36}{precision_0:.4f}    TN/(TN+FN): predicted 0 that are 0",
                 f"{'Matthews correlation coefficient':<36}{MCC:.4f}    -1 to 1, 0 = random"]
        filename = os.path.join(self.gen_dir, "Model", f"{info['model']}-classifier-{key_var_dependent}.txt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

    def write_models_cfg(self):

        lines = [f"# Metamodels of GEN_{self.gen:04d} (score: R2 with {self.validation}, balanced accuracy with {self.validation_binary})", ""]
        for key_var_dependent, info in self.models_info.items():
            lines.append(f"[{key_var_dependent}]")
            lines.append(f"model = {info['model']}")
            lines.append(f"index = {info['index']}")
            for key, val in info["hyperparameters"].items():
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                lines.append(f"{key} = {val}")
            lines.append(f"source = {info['source']}")
            lines.append(f"score = {info['score']}")
            lines.append("")
        with open(self._file("models.cfg"), "w") as f:
            f.write("\n".join(lines))

    def read_models_cfg(self):

        models_cfg = {}
        for key_var_dependent, section in read_user_input(self._file("models.cfg")).items():
            info = {"model": section["model"][0], "index": section["index"][0]}
            # Values of the config file, replaced by the ones in models.cfg (older files may miss some keys)
            info["hyperparameters"] = self._get_model(info["model"], info["index"]).get_hyperparameters()
            for key in info["hyperparameters"].keys():
                if key in section:
                    info["hyperparameters"][key] = section[key]
            # Values as returned by the model (e.g. 'logistic' instead of ['logistic'])
            model = self._get_model(info["model"], info["index"], info["hyperparameters"])
            info["hyperparameters"] = model.get_hyperparameters()
            models_cfg[key_var_dependent] = info
        return models_cfg

    # -----------------------------------------------------#
    # Parametric optimization
    # -----------------------------------------------------#

    def set_optimization(self):

        # Set bounds
        self.X_min, self.X_max = self.read_ranges("ranges.cfg")
        # Objectives and constraints
        self.opt_obj = self.IN['optimization']['opt_obj']
        self.n_obj = len(self.opt_obj)
        self.opt_constraints = self.IN['optimization']['opt_constraints']
        self.n_constraints = len(self.opt_constraints)

    def _get_performance(self, DV):

        if len(np.shape(DV)) == 1:    DV = np.array([DV])

        obj_val = {}
        for obj_key in self.opt_obj:
            obj_val[obj_key] = self.models[obj_key].predict(DV)

        constraints_val = {}
        for constraints_key in self.opt_constraints:
            constraints_val[constraints_key] = self.models[constraints_key].predict(DV)

        return obj_val, constraints_val

    def _get_objective_and_constraints(self, DV):

        # Evaluates all designs of DV at once: obj (n_designs, n_obj), constraints (n_designs, n_constraints)
        obj_val, constraints_val = self._get_performance(DV)

        obj = []
        for i, obj_option in enumerate(self.opt_obj):
            min_max = self.IN['optimization']['opt_min_max'][i]
            _obj = obj_val[obj_option]
            if 'abs' in min_max:
                _obj = np.abs(_obj)
            if 'max' in min_max:
                _obj = -_obj
            obj.append(_obj)

        constraints = []
        opt_constraints_baseline = self.IN['optimization']['opt_constraints_baseline']
        opt_constraints_above_below_baseline = self.IN['optimization']['opt_constraints_above_below_baseline']
        for i, constraints_key in enumerate(self.opt_constraints):
            _baseline = opt_constraints_baseline[i]
            _above_below = opt_constraints_above_below_baseline[i]
            if _above_below == "above":
                _constraint = _baseline - constraints_val[constraints_key] # < 0
            elif _above_below == "below":
                _constraint = constraints_val[constraints_key] - _baseline # < 0
            else:
                raise Exception(f"Incorrect option: _above_below = {_above_below}")
            constraints.append(_constraint)

        return np.array(obj).transpose(), np.array(constraints).transpose()

    def optimize(self):

        print_section(f"Optimization of the metamodels ({self.infill})")
        if self.infill == "surrogate_optimum":
            out = self.optimize_nsga2_batch(func_batch=self._get_objective_and_constraints,
                                            X_min=self.X_min, X_max=self.X_max,
                                            n_obj=self.n_obj, n_constr=self.n_constraints, IN=self.IN)
            X_opt, objective_opt, constraints_opt, feasible = out
        else:
            raise Exception(f"Infill option not available: {self.infill}")
        if not feasible:
            log_print("WARNING: no feasible design found, using the least-violating one")

        # Predicted performance
        predicted = {}
        for key, model in self.models.items():
            predicted[key] = float(model.predict(np.array([X_opt]))[0])

        # Active bounds
        width = self.X_max - self.X_min
        at_lower = (X_opt - self.X_min) <= self.bound_tolerance * width
        at_upper = (self.X_max - X_opt) <= self.bound_tolerance * width

        # Print optimum
        log_print("")
        log_print(f"    {'design variable':<32}{'optimum':>12}{'lower':>12}{'upper':>12}")
        for i, key_var in enumerate(self.keys_var_independent):
            bound = "  <- lower bound" if at_lower[i] else ("  <- upper bound" if at_upper[i] else "")
            log_print(f"    {key_var:<32}{X_opt[i]:>12.6g}{self.X_min[i]:>12.6g}{self.X_max[i]:>12.6g}{bound}")
        log_print("")
        log_print(f"    {'predicted':<32}")
        for key, val in predicted.items():
            log_print(f"    {key:<32}{val:>12.6g}")

        # Metrics of the metamodels near the optimum
        self.report_local_metrics(X_opt)

        # New ranges
        X_min_new, X_max_new = self.update_ranges(at_lower, at_upper)
        self.write_ranges("ranges_new.cfg", X_min_new, X_max_new)

        # Save optimum
        dvs_doe = {}
        for i, key_var in enumerate(self.keys_var_independent):
            dvs_doe[key_var] = float(X_opt[i])
        optimum = {"dvs_doe": dvs_doe, "dvs_cfd": self._get_dvs_cfd(dvs_doe), "predicted": predicted,
                   "feasible": bool(feasible),
                   "at_lower_bound": at_lower.tolist(), "at_upper_bound": at_upper.tolist()}
        with open(self._file("optimum.json"), "w") as f:
            json.dump(optimum, f, indent=2)

    def report_local_metrics(self, X_opt):

        # Metrics for the n_local points closest to the optimum (distance with each design variable normalized by
        # its range). Predictions by k-fold cross-validation over training + validation data
        print_section(f"Metamodels near the optimum ({self.n_local} closest points, {self.cv_folds}-fold cross-validation)")
        width = np.where(self.X_max > self.X_min, self.X_max - self.X_min, 1.0)
        lines = []
        for key_var_dependent in self.keys_var_dependent + self.keys_var_dependent_binary:
            binary = key_var_dependent in self.keys_var_dependent_binary
            info = self.models_info[key_var_dependent]
            X, y_true, y_predicted = self._get_predictions(info["model"], info["index"], info["hyperparameters"],
                                                           key_var_dependent, binary, "cross_validation")
            distance = np.sqrt(np.sum(((X - X_opt) / width) ** 2, axis=1))
            index_local = np.argsort(distance)[:self.n_local]
            score, details = self._get_metrics(y_true[index_local], y_predicted[index_local], binary)
            line = f"    {key_var_dependent:<26}"
            if binary:
                line += (f"balanced accuracy = {score:.4f}    [[TN FP] [FN TP]] = {details.tolist()}"
                         f"    diverged nearby = {int(np.sum(y_true[index_local] == 0))}")
            else:
                line += (f"R2 = {score:.4f}    max error = {details['max_error']:.4g}    "
                         f"P95 error = {details['p95_error']:.4g}    RMSE = {details['RMSE']:.4g}    "
                         f"bias = {details['bias']:+.4g}")
            line += f"    max distance = {np.amax(distance[index_local]):.3f}"
            lines.append(line)
            log_print(line)
        with open(self._file("local_metrics.txt"), "w") as f:
            f.write(f"Metrics of the metamodels for the {self.n_local} points closest to the optimum\n")
            f.write(f"({self.cv_folds}-fold cross-validation; distance normalized by the ranges of the design variables)\n\n")
            f.write("\n".join(lines) + "\n")

    def _get_dvs_cfd(self, dvs_doe):

        parametrization = rotor_and_diffuser_blades(self.IN)
        parametrization.add_dvs_doe_to_cfd(dvs_doe)
        dvs_cfd = {}
        for key, val in parametrization.dvs_cfd.items():
            dvs_cfd[key] = float(val[-1])
        return dvs_cfd

    # -----------------------------------------------------#
    # Ranges of the design variables
    # -----------------------------------------------------#

    def read_range_expansion(self):

        # [dvs_doe_range_expansion] var = expansion (% of the width), hard delta min, hard delta max (none = no limit)
        self.range_expansion = {}
        for key_var in self.keys_var_independent:
            baseline = self.parametrization.dvs_doe_baseline[key_var]
            percent, hard_min, hard_max = 0.0, -np.inf, np.inf
            if key_var in self.IN.get('dvs_doe_range_expansion', {}):
                val = self.IN['dvs_doe_range_expansion'][key_var]
                percent = val[0]
                if str(val[1]).lower() != 'none': hard_min = baseline + val[1]
                if str(val[2]).lower() != 'none': hard_max = baseline + val[2]
            self.range_expansion[key_var] = [percent, hard_min, hard_max]

    def update_ranges(self, at_lower, at_upper):

        X_min_new, X_max_new = self.X_min.copy(), self.X_max.copy()
        if self.range_update == "fixed":
            return X_min_new, X_max_new
        elif self.range_update != "expand_at_bounds":
            raise Exception(f"Range update option not available: {self.range_update}")

        log_print("")
        for i, key_var in enumerate(self.keys_var_independent):
            percent, hard_min, hard_max = self.range_expansion[key_var]
            delta = percent / 100 * (self.X_max[i] - self.X_min[i])
            if at_lower[i] and delta > 0:
                X_min_new[i] = max(self.X_min[i] - delta, hard_min)
                log_print(f"    Range of {key_var}: lower bound {self.X_min[i]:.6g} -> {X_min_new[i]:.6g}"
                          + ("  (hard limit)" if X_min_new[i] == hard_min else ""))
            if at_upper[i] and delta > 0:
                X_max_new[i] = min(self.X_max[i] + delta, hard_max)
                log_print(f"    Range of {key_var}: upper bound {self.X_max[i]:.6g} -> {X_max_new[i]:.6g}"
                          + ("  (hard limit)" if X_max_new[i] == hard_max else ""))
        return X_min_new, X_max_new

    def write_ranges(self, filename, X_min, X_max):

        lines = ["# Ranges of the design variables (absolute values)", "# variable = lower, upper", "[ranges]"]
        for i, key_var in enumerate(self.keys_var_independent):
            lines.append(f"{key_var} = {X_min[i]}, {X_max[i]}")
        with open(self._file(filename), "w") as f:
            f.write("\n".join(lines) + "\n")

    def read_ranges(self, filename):

        ranges = read_user_input(self._file(filename))['ranges']
        X_min = np.array([ranges[key_var][0] for key_var in self.keys_var_independent], dtype=float)
        X_max = np.array([ranges[key_var][1] for key_var in self.keys_var_independent], dtype=float)
        return X_min, X_max

    # -----------------------------------------------------#
    # CFD
    # -----------------------------------------------------#

    def run_cfd(self, resume=False):

        # Returns False if the loop must stop (Workbench execution disabled)
        print_section("CFD")
        with open(self._file("optimum.json"), "r") as f:
            optimum = json.load(f)
        run_dir = self._file("CFD")

        if not self.wb_settings.run_workbench:
            self.cfx_runner.prepare(run_dir, optimum["dvs_cfd"])
            log_print("run_workbench = false: journal written, Workbench not executed.")
            log_print("Set run_workbench = true and run again to continue.")
            return False

        run_info = {}
        if resume and os.path.isfile(os.path.join(run_dir, "run_info.json")):
            with open(os.path.join(run_dir, "run_info.json"), "r") as f:
                run_info = json.load(f)
        if "t_end" in run_info:
            log_print("Previous run finished: collecting results")
        else:
            if resume:
                log_print("Previous run was interrupted: running again")
            journal_path = self.cfx_runner.prepare(run_dir, optimum["dvs_cfd"])
            if self.wb_settings.breakpoint_before_cfd:
                # 'c' runs CFX, 'q' quits (a restart writes the journal again)
                log_print(f"breakpoint_before_cfd = true: stopping before running {journal_path}")
                breakpoint()
            self.set_status("cfd_submitted")
            self.cfx_runner.execute(run_dir, journal_path)
        result = self.cfx_runner.collect(run_dir)

        # Print results
        log_print("")
        log_print(f"    Convergence = {result['Convergence']}  (predicted {optimum['predicted'].get('Convergence', '-')})")
        log_print(f"    {result['message']}")
        log_print("")
        log_print(f"    {'output':<32}{'CFD':>14}{'predicted':>14}")
        for key, val in result["outputs"].items():
            predicted = f"{optimum['predicted'][key]:>14.6g}" if key in optimum["predicted"] else ""
            log_print(f"    {key:<32}{val:>14.6g}{predicted}")

        with open(self._file("cfd_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        return True

    # -----------------------------------------------------#
    # Post processing
    # -----------------------------------------------------#

    def post_process(self):

        print_section("New datasets")
        with open(self._file("optimum.json"), "r") as f:
            optimum = json.load(f)
        with open(self._file("cfd_result.json"), "r") as f:
            result = json.load(f)

        # New row: design variables, CFD variables, CFD outputs and convergence
        new_row = {}
        new_row.update(optimum["dvs_doe"])
        new_row.update(optimum["dvs_cfd"])
        new_row.update(result["outputs"])
        new_row["Convergence"] = result["Convergence"]

        with open(self._file("training.csv"), "r", newline="") as f:
            rows = [row for row in csv.reader(f) if len(row) > 0]
        header, data = rows[0], rows[1:]
        for key in new_row.keys():
            if key not in header:
                header.append(key)
                for row in data: row.append("nan")
        data.append([str(new_row.get(key, "nan")) for key in header])
        with open(self._file("training_new.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        shutil.copy(self._file("validation.csv"), self._file("validation_new.csv"))
        log_print(f"    training_new.csv: {len(data)} points")

        self.write_history()

    def write_history(self):

        # OPT/history.csv: one row per generation with CFD results
        header, data = None, []
        gen = 0
        while os.path.isfile(self._file("cfd_result.json", gen)):
            with open(self._file("optimum.json", gen), "r") as f:
                optimum = json.load(f)
            with open(self._file("cfd_result.json", gen), "r") as f:
                result = json.load(f)
            row = {"generation": gen, "Convergence_cfd": result["Convergence"]}
            for key in optimum["predicted"].keys():
                row[f"{key}_predicted"] = optimum["predicted"][key]
                if key in result["outputs"]:
                    row[f"{key}_cfd"] = result["outputs"][key]
            for i, key_var in enumerate(self.keys_var_independent):
                row[key_var] = optimum["dvs_doe"][key_var]
                row[f"{key_var}_bound"] = ("lower" if optimum["at_lower_bound"][i] else
                                           ("upper" if optimum["at_upper_bound"][i] else ""))
            if header is None:
                header = list(row.keys())
            data.append([row.get(key, "") for key in header])
            gen += 1
        with open(os.path.join(self.opt_directory, "history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    # -----------------------------------------------------#
    # Plotting
    # -----------------------------------------------------#

    def _plot_model(self, y_model, y_predicted, model, key_var_dependent, r2, dataset_name=""):

        x_label = 'Model data'
        y_label = 'Predicted data'
        title = f"GEN_{self.gen:04d} ({dataset_name})\n{key_var_dependent.replace('_', ' ')}\nModel: {model.name} (R²={'%0.2f' % r2})"
        filename = os.path.join("Model", f"{model.name}-predicted-{key_var_dependent}-{dataset_name}")
        y_45 = np.array([np.linspace(min(np.amin(y_model), np.amin(y_predicted)),
                                     max(np.amax(y_model), np.amax(y_predicted)),
                                     2)]).transpose()
        x_lim = [np.amin(y_model), np.amax(y_model)]
        dx, eps = x_lim[1] - x_lim[0], 0.02
        x_lim = [x_lim[0] - dx * eps, x_lim[1] + dx * eps]

        plot_SBO(x_label, y_label, title, x_lim=x_lim,
                 X0=y_model, Y0=y_predicted, x0y0_kind="points", labels_list_x0=["data"],
                 X1=y_45, Y1=y_45, x1y1_kind="dotted", labels_list_x1=["y=x"],
                 flag_aspect_ratio=False,
                 output_directory=self.gen_dir, loc='upper left',
                 filename=filename, extensions=['png'])
