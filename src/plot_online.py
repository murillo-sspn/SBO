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
#   Plots of the online SBO history: design variables #
#   with their ranges, objective and constraints of   #
#   each design. Left: baseline and initial DoE,      #
#   right: optimization (shared y axis). Each plot    #
#   is saved as .pdf, .png and .csv                   #
#######################################################

#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#-----------------------------------------------------#
# Definitions
#-----------------------------------------------------#

# Times New Roman, LaTeX style math symbols (STIX)
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}

# Axis label and scale factor of each variable (plotted value = value * scale)
LABELS = {
    "rotor_hub_midstream_angle":    (r"Rotor hub midstream angle [°]", 1.0),
    "rotor_shroud_midstream_angle": (r"Rotor shroud midstream angle [°]", 1.0),
    "rotor_hub_wrap_perturbation":  (r"Rotor hub wrap perturbation [°]", 1.0),
    "rotor_TE_rake_angle":          (r"Rotor TE rake angle [°]", 1.0),
    "diffuser_midstream_angle":     (r"Diffuser midstream angle [°]", 1.0),
    "diffuser_wrap_perturbation":   (r"Diffuser wrap perturbation [°]", 1.0),
    "diffuser_tn2":                 (r"Diffuser thickness $t_{n,2}$ [mm]", 1000.0),
    "diffuser_tn3":                 (r"Diffuser thickness $t_{n,3}$ [mm]", 1000.0),
    "Eff_Isentropic_Vanes_TE":      (r"Isentropic efficiency, $\eta_{is}$ [%]", 1.0),
    "Ratio_p0_Vanes_TE":            (r"Total pressure ratio, $p_{0}/p_{01}$ [-]", 1.0),
    "Convergence":                  (r"Convergence [-]", 1.0),
    # Monitors
    "Eff_Isentropic_Outlet":        (r"Isentropic efficiency (outlet), $\eta_{is}$ [%]", 1.0),
    "Eff_Polytropic_Vanes_TE":      (r"Polytropic efficiency, $\eta_{p}$ [%]", 1.0),
    "Ratio_p0_Outlet":              (r"Total pressure ratio (outlet), $p_{0}/p_{01}$ [-]", 1.0),
    "Diffuser_Cp":                  (r"Diffuser pressure recovery, $C_{p}$ [-]", 1.0),
    "Power":                        (r"Power [kW]", 1.0e-3),
    "S_Gen":                        (r"Entropy generation, $\dot{S}_{gen}$ [W/K]", 1.0),
}

# Colors: baseline, initial DoE, optimization, best design
COLOR_BASELINE = "tab:red"
COLOR_DOE = "0.6"
COLOR_OPT = "k"
COLOR_BEST = "blue"


def get_label(key):
    if key in LABELS:
        return LABELS[key]
    return key.replace("_", " "), 1.0


# -----------------------------------------------------#
# Common
# -----------------------------------------------------#

def _create_figure(history):
    """ Left: baseline and initial DoE (x = design), right: optimization (x = generation), shared y axis """
    n_doe = int(np.sum(history["generation"] < 0))
    n_gen = int(np.sum(history["generation"] >= 0))
    fig, (ax_doe, ax_opt) = plt.subplots(1, 2, sharey=True, figsize=(10, 5.2),
                                         gridspec_kw={"width_ratios": [2, 1], "wspace": 0.06})
    ax_doe.set_title("Exploration (initial DoE)")
    ax_opt.set_title("Optimization")
    ax_doe.set_xlabel("Design")
    ax_opt.set_xlabel("Generation")
    ax_doe.set_xlim(-1, n_doe)
    ax_opt.set_xlim(-0.5, n_gen - 0.5)
    ax_opt.set_xticks(range(0, n_gen, max(1, int(np.ceil(n_gen / 10)))))
    ax_opt.tick_params(axis="y", left=False)
    for ax in (ax_doe, ax_opt):
        ax.grid(True, linewidth=0.4, alpha=0.5)
    return fig, ax_doe, ax_opt


def _split(history, values):
    """ x and values of the DoE panel and of the optimization panel """
    doe = history["generation"] < 0
    return history["design"][doe], values[doe], history["generation"][~doe], values[~doe]


def _plot_designs(ax_doe, ax_opt, history, y, label_not_converged="not converged"):
    """ Designs (black): circles converged, crosses not converged; baseline (red), new best designs (blue)
    The panel tells the phase (initial DoE or optimization) """
    doe = history["generation"] < 0
    converged = history["convergence"] == 1
    phase = history["phase"]
    for ax, mask_phase, x, size in [(ax_doe, phase == "DoE", history["design"], 24),
                                    (ax_opt, phase == "optimization", history["generation"], 40)]:
        mask = mask_phase & converged
        ax.scatter(x[mask], y[mask], s=size, color=COLOR_OPT, zorder=3, label="converged")
        mask = mask_phase & ~converged
        if np.any(mask):
            ax.scatter(x[mask], y[mask], s=size + 14, color=COLOR_OPT, marker="x", linewidth=1.2, zorder=3,
                       label=label_not_converged)
    mask = history["new_best"] & ~doe
    if np.any(mask):
        ax_opt.scatter(history["generation"][mask], y[mask], s=75, color=COLOR_BEST, zorder=4,
                       label="new best design")
    mask = phase == "baseline"
    ax_doe.scatter(history["design"][mask], y[mask], s=65, color=COLOR_BASELINE, marker="D", zorder=5,
                   label="baseline")


def _plot_baseline_value(ax_doe, ax_opt, history, y, label="baseline value"):
    i_baseline = np.where(history["phase"] == "baseline")[0]
    if len(i_baseline) == 0:
        return
    for ax in (ax_doe, ax_opt):
        ax.axhline(y[i_baseline[0]], color=COLOR_BASELINE, linestyle="--", linewidth=1.3, zorder=2,
                   label=label)


def _save(fig, ax_doe, ax_opt, output_directory, filename, header, rows):
    # Legend with the entries of both panels, below the axes
    handles, labels = [], []
    for ax in (ax_doe, ax_opt):
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    n_columns = 3
    fig.legend(handles, labels, loc="lower center", ncol=n_columns, frameon=False, bbox_to_anchor=(0.5, 0.0))
    n_rows = int(np.ceil(len(labels) / n_columns))
    fig.subplots_adjust(bottom=0.15 + 0.065 * n_rows, left=0.10, right=0.98, top=0.93)
    os.makedirs(output_directory, exist_ok=True)
    for extension in ["pdf", "png"]:
        fig.savefig(os.path.join(output_directory, f"{filename}.{extension}"), dpi=300)
    plt.close(fig)
    with open(os.path.join(output_directory, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# -----------------------------------------------------#
# Plots
# -----------------------------------------------------#

def plot_design_variable(output_directory, key, history, value, lower, upper, hard_min=-np.inf, hard_max=np.inf):
    """ Design variable of each design with the lower and upper bounds of its range and the hard limits
    (maximum extension of the range; infinite = no limit)

    history: design, generation (-1: initial DoE), phase, convergence, new_best (see plot_history of the driver)
    """
    label, scale = get_label(key)
    with plt.rc_context(STYLE):
        fig, ax_doe, ax_opt = _create_figure(history)
        for bound in (lower, upper):
            x_doe, b_doe, x_opt, b_opt = _split(history, bound * scale)
            ax_doe.plot([-1, len(x_doe)], [b_doe[0]] * 2, color="k", linewidth=1.0, label="range bounds")
            ax_opt.step(x_opt, b_opt, where="mid", color="k", linewidth=1.0)
        for hard_limit in (hard_min, hard_max):
            if np.isfinite(hard_limit):
                for ax in (ax_doe, ax_opt):
                    ax.axhline(hard_limit * scale, color="k", linestyle=":", linewidth=1.4, zorder=2,
                               label="hard limits")
        _plot_baseline_value(ax_doe, ax_opt, history, value * scale)
        _plot_designs(ax_doe, ax_opt, history, value * scale)
        ax_doe.set_ylabel(label)

        header = ["design", "generation", "phase", f"{key} (x{scale:g})", "lower_bound", "upper_bound",
                  "hard_min", "hard_max", "Convergence", "new_best"]
        rows = []
        for i in range(len(value)):
            rows.append([history["design"][i], history["generation"][i], history["phase"][i], value[i] * scale,
                         lower[i] * scale, upper[i] * scale, hard_min * scale, hard_max * scale,
                         history["convergence"][i], int(history["new_best"][i])])
        _save(fig, ax_doe, ax_opt, output_directory, f"design_variable-{key}", header, rows)


def plot_output(output_directory, key, history, value, predicted, best=None, limit=None, limit_label="",
                binary=False, kind="objective"):
    """ Objective, constraint or monitor of each design: CFD value and value predicted by the metamodels

    best:   best feasible value so far (objective), one value per design
    limit:  constraint limit (horizontal line)
    kind:   objective | constraint | monitor (file name)
    """
    label, scale = get_label(key)
    with plt.rc_context(STYLE):
        fig, ax_doe, ax_opt = _create_figure(history)
        if not binary:
            _plot_baseline_value(ax_doe, ax_opt, history, value * scale)
        if limit is not None:
            for ax in (ax_doe, ax_opt):
                ax.axhline(limit * scale, color="k", linestyle="--", linewidth=1.0, zorder=2, label=limit_label)
        if best is not None:
            x_doe, best_doe, x_opt, best_opt = _split(history, best * scale)
            ax_doe.step(x_doe, best_doe, where="post", color=COLOR_BEST, linewidth=1.6, zorder=2,
                        label="best feasible so far")
            # Optimization panel starts with the best design of the DoE
            ax_opt.step(np.concatenate(([-0.5], x_opt, [x_opt[-1] + 0.5])),
                        np.concatenate(([best_doe[-1]], best_opt, [best_opt[-1]])),
                        where="post", color=COLOR_BEST, linewidth=1.6, zorder=2)
        # Values of designs not converged are meaningless: shown on the baseline value
        value_plot = value.copy()
        label_not_converged = "not converged"
        i_baseline = np.where(history["phase"] == "baseline")[0]
        if not binary and len(i_baseline) > 0:
            value_plot[history["convergence"] != 1] = value[i_baseline[0]]
            label_not_converged = "not converged (on baseline value)"
        _plot_designs(ax_doe, ax_opt, history, value_plot * scale, label_not_converged)
        x_doe, p_doe, x_opt, p_opt = _split(history, predicted * scale)
        if np.any(np.isfinite(p_opt)):
            ax_opt.scatter(x_opt, p_opt, s=48, marker="^", facecolors="none", edgecolors="k", linewidth=1.0,
                           zorder=4, label="predicted (metamodel)")
        if binary:
            ax_doe.set_yticks([0, 1])
            ax_doe.set_ylim(-0.25, 1.25)
        ax_doe.set_ylabel(label)

        header = ["design", "generation", "phase", f"{key}_cfd (x{scale:g})", f"{key}_plotted (x{scale:g})",
                  f"{key}_predicted (x{scale:g})", "Convergence", "new_best"]
        if best is not None:
            header.append("best_feasible_so_far")
        rows = []
        for i in range(len(value)):
            row = [history["design"][i], history["generation"][i], history["phase"][i], value[i] * scale,
                   value_plot[i] * scale, "" if np.isnan(predicted[i]) else predicted[i] * scale,
                   history["convergence"][i], int(history["new_best"][i])]
            if best is not None:
                row.append(best[i] * scale)
            rows.append(row)
        _save(fig, ax_doe, ax_opt, output_directory, f"{kind}-{key}", header, rows)
