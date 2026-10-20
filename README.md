# SBO – Surrogate Based Optimization

Surrogate based optimization of compressor stages using CFD data (ANSYS CFX).
Metamodels (neural networks, kriging, RBF) are trained on CFD results and optimized with NSGA-II.
A binary classifier (random forest, SVM) predicts whether the CFD converges and keeps the optimizer away from diverging designs.

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set SBO_HOME=C:\path\to\SBO
```

## Usage

Run from the folder of the case, passing the config file:

```
cd test_cases\StageH2
python %SBO_HOME%\bin\SBO.py config.cfg
```

The type of problem is set in `[general_info] problem`:

| problem | Driver | Description |
|---|---|---|
| `SBO` | `src/driver_SBO.py` | Offline: trains metamodels on a dataset and optimizes them |
| `SBO_with_classification` | `src/driver_SBO_with_classification.py` | Offline, with convergence classifier as constraint |
| `online_SBO` | `src/driver_online_SBO.py` | Online: runs the optimum in CFX and updates the metamodels at every generation |
| `multistage_compressor` | `src/driver_multistage.py` | Multistage compressor design (sCO2) |

## Online SBO

Inputs: `config.cfg`, training and validation datasets (CSV with design variables, CFD outputs and a `Convergence` column, 1 = converged), and a Workbench project with the CFX case parameterized.

Each generation `OPT/GEN_XXXX`:

1. **pre_process** – copies the datasets and ranges of the previous generation
2. **train_models** – selects and trains the metamodels (`models.cfg`); continuous models use converged points only
3. **optimize** – NSGA-II on the metamodels; if the optimum is at a bound, that side of the range is expanded (limited by hard limits)
4. **run_cfd** – writes a Workbench journal, runs `runwb2` in batch mode, reads outputs and convergence (solver `.out`)
5. **post_process** – adds the CFD result to `training_new.csv`, the training set of the next generation

The run can be interrupted and restarted: each generation stores its completed stages in `status.json`.
To stop between generations, create an empty file `OPT/STOP`.

### Main config sections

| Section | Content |
|---|---|
| `[general_info]` | variables, datasets, models, `validation` / `validation_binary` (`holdout` or `cross_validation`), `r2_min` |
| `[neural_networks]` | network settings and hyperparameter optimization (`nn_opt_*`, `nn_opt_metric = r2 \| max_error \| p95_error`) |
| `[random_forest]`, `[support_vector_machine]` | classifier settings |
| `[optimization]`, `[nsga2]` | objectives, constraints, NSGA-II settings |
| `[dvs_cfd_baseline]`, `[dvs_doe_baseline_and_ranges]` | baseline geometry and initial ranges |
| `[dvs_doe_range_expansion]` | expansion (% of range width) and hard limits per variable |
| `[online]` | `max_generations`, `selection_every`, `bound_tolerance`, `n_local`, `infill`, `range_update` |
| `[workbench]` | project, mode, executable, `timeout_s`, `run_workbench`, `breakpoint_before_cfd` |
| `[workbench_inputs]`, `[workbench_outputs]` | mapping between variable names and Workbench parameters (`Pxx`) |
| `[cfd_convergence]` | `out_file` (solver termination summary) or `max_iterations` |

Workbench modes (`[workbench] mode`):

- `copy_new_dp` – copy of the project in each generation, new design point
- `copy_modify_dp` – copy of the project in each generation, base design point modified
- `base_new_dp` – design points added to the base project itself
- `shared_new_dp` – project copied once to `OPT/Workbench`, design points of all generations added to it

For debugging: `run_workbench = false` writes the journal and stops; `breakpoint_before_cfd = true` stops in the debugger right before running CFX.

### Outputs

| File | Content |
|---|---|
| `OPT/log.txt` | log of all runs |
| `OPT/history.csv` | per generation: optimum, predicted and CFD values, active bounds |
| `GEN_XXXX/models.cfg` | selected metamodels and hyperparameters |
| `GEN_XXXX/Model/` | predicted vs data plots, classifier report (confusion matrix and metrics) |
| `GEN_XXXX/local_metrics.txt` | metamodel metrics for the `n_local` points closest to the optimum |
| `GEN_XXXX/optimum.json`, `cfd_result.json` | optimum (design and CFD variables, predictions) and CFD result |
| `GEN_XXXX/ranges.cfg`, `ranges_new.cfg` | ranges used and ranges for the next generation |
| `GEN_XXXX/CFD/` | journal, exported design point table, Workbench log |

## Structure

```
bin/SBO.py              entry point
common/                 config reader, logger, plotting, directories
src/models.py           metamodels (nn, kriging, rbf, rf, svm)
src/opt.py              NSGA-II
src/parametrization.py  design variables (DoE) -> CFD/BladeGen variables
src/cfx_runner.py       Workbench journal, execution, results and convergence
src/driver_*.py         drivers of each problem type
test_cases/             example cases
```
