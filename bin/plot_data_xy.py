#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# GLOBAL STYLE
# ============================================================
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "lines.linewidth": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# CHECK INPUT
# ============================================================
if len(sys.argv) != 2:
    print("Usage: python plot_data_xy.py case_dir/plot.cfg")
    sys.exit(1)

config_path = sys.argv[1]
case_dir = os.path.dirname(config_path)

# ============================================================
# READ CONFIG
# ============================================================
config = {}
with open(config_path, "r") as f:
    exec(f.read(), {}, config)

files = config["files"]
x_label = config["x_label"]
y_label = config["y_label"]
title = config["title"]
legend_labels = config["legend_labels"]
colors = config["colors"]

# Optional limits
x_lim = config.get("x_lim", None)
y_lim = config.get("y_lim", None)
# Optional ticks
xticks = config.get("xticks", None)
yticks = config.get("yticks", None)
# Offset
x_offset = config.get("x_offset", 0.0)
y_offset = config.get("y_offset", 0.0)
# Multiplicative offset
x_mult = config.get("x_mult", 1.0)
y_mult = config.get("y_mult", 1.0)
# Linear interp
flag_lin_interp_x = config.get("lin_interp_x", False)
flag_lin_interp_y = config.get("lin_interp_y", False)
n_lin_interp = config.get("n_lin_interp", 10000)
# Insets
flag_inset = config.get("flag_inset", False)
x_lim_inset = config.get("x_lim_inset", None)
y_lim_inset = config.get("y_lim_inset", None)

# Adjust 180
flag_adjust_180 = config.get("adjust_180", False)

# X and Y scales
x_scale = config.get("x_scale", "linear")
y_scale = config.get("y_scale", "linear")

# NUmber of periods
n_periods = config.get("n_periods", 1)

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(10,8))

plt.xscale(x_scale)
plt.yscale(y_scale)

if flag_inset:
    ax_inset = inset_axes(
                    ax,
                    width="35%",
                    height="35%",
                    loc="lower right",
                    bbox_to_anchor=(-0.03, 0.06, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0
                )

for file_name, label, color in zip(files, legend_labels, colors):

    file_path = os.path.join(case_dir, file_name)

    try:
        data = np.genfromtxt(file_path, delimiter="\t", names=True)
    except:
        data = np.genfromtxt(file_path, delimiter=",", names=True)

    col_names = data.dtype.names
    x = x_mult * data[col_names[0]] + x_offset
    y = y_mult * data[col_names[1]] + y_offset

    if flag_lin_interp_x:
        _x = np.linspace(x.min(), x.max(), n_lin_interp)
        _y = np.interp(_x, x, y)
    elif flag_lin_interp_y:
        _y = np.linspace(y.min(), y.max(), n_lin_interp)
        _x = np.interp(_y, y, x)
    else:
        _x = x + 0.0
        _y = y + 0.0

    if flag_adjust_180:

        for i, __x in enumerate(_x):
            if __x > 180:
                _x[i] = __x - 360

        theta = _x
        theta_fixed = theta.copy()
        jumps = np.abs(np.diff(theta)) > 180
        theta_fixed[1:][jumps] = np.nan
        _x = theta_fixed + 0.0

    period_length = _x.max() - _x.min()

    for k in range(n_periods):
        x_shifted = _x + k * period_length
        ax.plot(x_shifted, _y, label=label if k == 0 else None, color=color)

    if flag_inset:
        ax_inset.plot(_x, _y, label=label, color=color)

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_title(title)

# Apply limits only if provided
if xticks is not None:  ax.set_xticks(xticks)
if yticks is not None:  ax.set_yticks(yticks)
if x_lim is not None:   ax.set_xlim(x_lim)
if y_lim is not None:   ax.set_ylim(y_lim)
#ax.ticklabel_format(style='plain', useOffset=False)
if flag_inset:
    if x_lim_inset is not None:   ax_inset.set_xlim(x_lim_inset)
    if y_lim_inset is not None:   ax_inset.set_ylim(y_lim_inset)
    ax_inset.tick_params(labelsize=16)
    ax_inset.grid(True)
    #ax_inset.ticklabel_format(style='plain', useOffset=False)






# Boxed legend (default style)
ax.legend(loc="best", frameon=True)

ax.grid(True)

plt.tight_layout()

output_path = os.path.join(case_dir, f"plot_output-{case_dir}.pdf")
plt.savefig(output_path)
plt.close()

print(f"Plot saved to: {output_path}")