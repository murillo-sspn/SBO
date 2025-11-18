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
import os
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms as transforms
from matplotlib import rc
rc('font', family='serif', size=14)
rc('lines', linewidth=1.5)
from mpl_toolkits.axes_grid1 import make_axes_locatable

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#

#-----------------------------------------------------#
# Definitions
#-----------------------------------------------------#
# Colormap: https://matplotlib.org/stable/gallery/color/colormap_reference.html
#colormap_cmap='gist_rainbow'
#colormap_cmap='rainbow'
#colormap_cmap='jet'
colormap_cmap = 'viridis'


def plot_SBO(*args, **kwargs):
    plot = Plotting(*args, **kwargs)

class Plotting():

    def __init__(self, x_label: str, y_label: str, title: str, x_lim=None, y_lim=None,
                 z_label=None, X_contour=None, Y_contour=None, Z_contour=None, levels_contour=20, cmap='viridis',
                 X0=None, Y0=None, labels_list_x0=None, colors_0=None, x0y0_kind="evalpts", ha_0=None, va_0=None,
                 X1=None, Y1=None, labels_list_x1=None, colors_1=None, x1y1_kind="ctrlpts", ha_1=None, va_1=None,
                 X2=None, Y2=None, labels_list_x2=None, colors_2=None, x2y2_kind="gridpts", ha_2=None, va_2=None,
                 X3=None, Y3=None, labels_list_x3=None, colors_3=None, x3y3_kind="xy",      ha_3=None, va_3=None,
                 X4=None, Y4=None, labels_list_x4=None, colors_4=None, x4y4_kind="pareto",  ha_4=None, va_4=None,
                 X5=None, Y5=None, labels_list_x5=None, colors_5=None, x5y5_kind="history", ha_5=None, va_5=None,
                 X6=None, Y6=None, labels_list_x6=None, colors_6=None, x6y6_kind="opt",     ha_6=None, va_6=None,
                 flag_ranges=False, index_ranges=0, X_L=None, X_U=None, Y_L=None, Y_U=None,
                 flag_rotate_second_and_penultimate_ranges=False,
                 flag_multiple_colors=False,
                 fig=None, ax=None, loc='lower right',
                 flag_grid=True, flag_aspect_ratio=False, flag_colorbar=False,
                 flag_show=False, output_directory=None, filename='figure', extensions=['png', 'pdf']):

        # Chart
        self.x_label, self.y_label, self.title, self.x_lim, self.y_lim = x_label, y_label, title, x_lim, y_lim
        # Contour
        self.z_label, self.X_contour, self.Y_contour, self.Z_contour = z_label, X_contour, Y_contour, Z_contour
        self.levels_contour, self.cmap = levels_contour, cmap

        # Summary lists
        self.X              = [X0, X1, X2, X3, X4, X5, X6]
        self.Y              = [Y0, Y1, Y2, Y3, Y4, Y5, Y6]
        self.labels_list    = [labels_list_x0, labels_list_x1, labels_list_x2,
                               labels_list_x3, labels_list_x4, labels_list_x5,
                               labels_list_x6]
        self.colors         = [colors_0, colors_1, colors_2, colors_3,
                               colors_4, colors_5, colors_6]
        self.xy_kind        = [x0y0_kind, x1y1_kind, x2y2_kind, x3y3_kind,
                               x4y4_kind, x5y5_kind, x6y6_kind]
        self.ha             = [ha_0, ha_1, ha_2, ha_3, ha_4, ha_5, ha_6]
        self.va             = [va_0, va_1, va_2, va_3, va_4, va_5, va_6]
        # Maximum number of plots
        self.N_plots_max    = len(self.X)
        # Initializing flag
        self.flag_labels_exist = False

        # Annotations
        for i in range(self.N_plots_max):
            if not(isinstance(self.ha[i], np.ndarray) or isinstance(self.ha[i], list)):
                self.ha[i] = ['center', 'center', 'center', 'center']
                self.va[i] = ['center', 'center', 'center', 'center']

        self.flag_ranges, self.index_ranges, self.X_L, self.X_U, self.Y_L, self.Y_U = flag_ranges, index_ranges, X_L, X_U, Y_L, Y_U
        self.flag_ranges_list = [(self.flag_ranges and i == self.index_ranges) for i in range(self.N_plots_max)]
        self.flag_rotate_second_and_penultimate_ranges = flag_rotate_second_and_penultimate_ranges
        self.flag_multiple_colors = flag_multiple_colors
        self.fig, self.ax = fig, ax
        self.flag_fig_is_none = False
        self.loc = loc
        self.flag_grid = flag_grid
        self.flag_aspect_ratio = flag_aspect_ratio
        self.flag_colorbar = flag_colorbar
        self.flag_show = flag_show

        # Saving information
        self.output_directory = output_directory
        self.filename = filename
        self.extensions = extensions

        self._plot_contour_multiple_xy_and_ranges()

    def _plot_contour_multiple_xy_and_ranges(self):

        """

        Parameters
        ----------


        Returns
        -------


        """


        if self.fig == None or self.ax == None:
            self.flag_fig_is_none = True
            self.fig, self.ax = plt.subplots(num=1)

        self.ax.ticklabel_format(useOffset=False)

        if self.flag_grid: self.ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

        # -----------------------------------------------------#
        # Contours
        # -----------------------------------------------------#

        if not(self.z_label == None):
            contour = self.ax.tricontourf(self.X_contour, self.Y_contour, self.Z_contour,
                                      levels=self.levels_contour, cmap=self.cmap)
            cbar = self.fig.colorbar(contour, ax=self.ax)
            cbar.set_label(self.z_label)

        # -----------------------------------------------------#
        # Points
        # -----------------------------------------------------#

        for i in range(self.N_plots_max):
            self._plot_XiYi(self.X[i], self.Y[i], self.labels_list[i],
                            self.colors[i], self.xy_kind[i], self.flag_ranges_list[i])

        # -----------------------------------------------------#
        # Annotations
        # -----------------------------------------------------#

        for i, _xy_kind in enumerate(self.xy_kind):
            if _xy_kind == "annotate":
                n = len(self.X[i])
                for j in range(n):
                    self.ax.text(self.X[i][j], self.Y[i][j], self.labels_list[i][j],
                                 fontsize=10, color='black',
                                 ha=self.ha[i][j], va=self.va[i][j],
                                 bbox=dict(
                                     boxstyle='round,pad=0.3',   # rounded box with small padding
                                     facecolor='white',          # white background
                                     edgecolor='black',          # outline color
                                     linewidth=0.8,              # thin black border
                                     alpha=0.9                   # slight transparency (optional)
                                 ))

        # -----------------------------------------------------#
        # Chart
        # -----------------------------------------------------#

        self._set_axis_attribute(self.fig, self.ax, self.x_label, self.y_label, self.title, self.x_lim, self.y_lim,
                            self.flag_aspect_ratio)

        if self.flag_labels_exist:  self.ax.legend(loc=self.loc)

        self._save_chart(self.fig, self.extensions, self.filename, self.output_directory)

        self._show_chart(self.flag_show)

        self.ax.cla()
        if self.flag_fig_is_none:
            self.fig.clf()
            plt.close(self.fig)

    def _plot_XiYi(self, Xi, Yi, labels_list_xi, colors_i, xiyi_kind, flag_ranges_i):

        if (isinstance(Xi, np.ndarray)) and (isinstance(Yi, np.ndarray)):

            # Number of points and curves
            try:
                n_pts, n_curves = np.shape(Xi)
            except:
                n_pts, = np.shape(Xi)
                n_curves = 1

            # Labels
            if (labels_list_xi == None):
                labels_list_xi = n_curves * [""]
            if not (self.flag_labels_exist):
                self.flag_labels_exist = not (((np.array(labels_list_xi)) == "").all())

            # Colors
            if colors_i == None and self.flag_multiple_colors:
                cmap = cm.get_cmap(colormap_cmap, n_curves)  # sample N discrete colors from viridis
                colors_i = [cmap(i) for i in range(n_curves)]

                if self.flag_colorbar and "history" in xiyi_kind:
                    norm = mcolors.Normalize(vmin=1, vmax=n_curves)
                    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax)
                    cbar.set_label('Iter')
                    cbar.set_ticks(np.linspace(1, n_curves, 2))
                    cbar.set_ticklabels([f"{int(t)}" for t in np.linspace(1, n_curves, 2)])

            # Set rotation angles array if ranges are plotted
            if flag_ranges_i:
                angles = np.zeros((n_pts, n_curves))
                # Adjut second and penultimate
                if self.flag_rotate_second_and_penultimate_ranges:
                    for i in range(n_curves):
                        angles[1, i] = 180 / np.pi * np.arctan((Yi[1, i] - Yi[0, i]) / (Xi[1, i] - Xi[0, i]))
                        angles[-2, i] = 180 / np.pi * np.arctan((Yi[-1, i] - Yi[-2, i]) / (Xi[-1, i] - Xi[-2, i]))

            for i in range(n_curves):

                if flag_ranges_i:

                    # Get x, y components of evaluated points
                    x_cp_l, y_cp_l = self.X_L[:, i], self.Y_L[:, i]
                    x_cp_u, y_cp_u = self.X_U[:, i], self.Y_U[:, i]

                    for j, _ in enumerate(self.X_L):

                        rect = Rectangle((x_cp_l[j], y_cp_l[j]),
                                         x_cp_u[j] - x_cp_l[j],
                                         y_cp_u[j] - y_cp_l[j],
                                         linewidth=2.0,
                                         edgecolor='darkblue',
                                         facecolor='blue',
                                         linestyle='-',
                                         alpha=0.6)
                        if self.flag_rotate_second_and_penultimate_ranges:
                            _angle = angles[j, i]
                            rect.set_transform(
                                transforms.Affine2D().rotate_deg_around(Xi[j, i], Yi[j, i], _angle) + self.ax.transData)
                        self.ax.add_patch(rect)

                # Get x, y components of evaluated points
                x, y = Xi[:, i], Yi[:, i]
                # Plot
                points, = self.ax.plot(x, y)
                # Kwargs for plot setup
                if "ctrlpts" in xiyi_kind:
                    if n_pts > 2:
                        _kwargs = {"kind": "ctrlpts"}
                    else:
                        _kwargs = {"kind": "ctrlpts2"}
                else:
                    _kwargs = {"kind": xiyi_kind}
                label = labels_list_xi[i]
                if label != "":                 _kwargs["label"] = label
                if self.flag_multiple_colors:   _kwargs["color"] = colors_i[i]
                self._plot_setup(points, **_kwargs)

    def _export_data(self):

        txt = ""
        txt += "\n-----------------------------------------------------\n"
        for i in range(self.N_plots_max):
            txt += f"i = {i}\n"
            txt += self._get_txt_data_XiYi(self.X[i], self.Y[i], self.labels_list[i])
            txt += "\n-----------------------------------------------------\n"
        txt += "Contour\n"
        txt += self._get_txt_data_XYZcontour()
        txt += "\n-----------------------------------------------------\n"
        filepath = os.path.join(self.output_directory, f"{self.filename}.dat")
        with open(filepath,'w') as f: f.write(txt)

    def _get_txt_data_XiYi(self, Xi, Yi, labels_list_xi):

        txt = ""

        if (isinstance(Xi, np.ndarray)) and (isinstance(Yi, np.ndarray)):

            try:
                n_pts, n_curves = np.shape(Xi)
            except:
                n_pts, = np.shape(Xi)
                n_curves = 1

            for i in range(n_curves):
                txt += f"{labels_list_xi[i]}\n"
                txt += f"{self.x_label}, {self.y_label}\n"
                # Get x, y components of evaluated points
                x, y = Xi[:, i], Yi[:, i]
                for j in range(n_pts):
                    txt += f"{'%.8f'%x[j]}, {'%.8f'%y[j]}\n"
                txt += "\n"

        return txt

    def _get_txt_data_XYZcontour(self):

        txt = ""
        if (isinstance(self.X_contour, np.ndarray)):
            n_pts, = np.shape(self.X_contour)
            txt += f"{self.x_label}, {self.y_label}, {self.z_label}\n"
            for i in range(n_pts):
                x, y, z = self.X_contour[i], self.Y_contour[i], self.Z_contour[i]
                txt += f"{'%.8f'%x}, {'%.8f'%y}, {'%.8f'%z}\n"
            txt += "\n"
        return txt


    # -----------------------------------------------------#
    # Auxiliary
    # -----------------------------------------------------#

    def _plot_setup(self, points, kind, label=None, color=None):
        if "evalpts" in kind:
            points.set_marker("")
            points.set_markersize(4.0)
            points.set_markeredgewidth(0.35)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_linestyle("-")
            self._set_func(points.set_color, color, "k")
            points.set_linewidth(2)
            points.set_label(label)

        elif "ctrlpts" in kind:
            points.set_marker("o")
            points.set_markersize(5)
            points.set_markeredgewidth(1)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("r")
            if kind == "ctrlpts":
                points.set_linestyle('dotted')
            elif kind == "ctrlpts2":
                points.set_linestyle('')
            self._set_func(points.set_color, color, "r")
            points.set_linewidth(1.5)
            points.set_label(label)

        elif "gridpts" in kind:
            points.set_marker("o")
            points.set_markersize(4.0)
            points.set_markeredgewidth(0.35)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_linestyle("-")
            self._set_func(points.set_color, color, "k")
            points.set_linewidth(2)
            points.set_label(label)

        elif "xy" in kind:
            points.set_marker("")
            points.set_markersize(1)
            points.set_markeredgewidth(0.35)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_linestyle("-")
            self._set_func(points.set_color, color, "k")
            points.set_linewidth(2)
            points.set_label(label)

        elif "pareto" in kind:

            points.set_marker("o")
            points.set_markersize(6)
            points.set_markeredgewidth(1)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("k")
            points.set_linestyle('dotted')
            points.set_color("k")
            points.set_linewidth(2)
            points.set_label(label)

        elif "history" in kind:
            points.set_marker("o")
            points.set_markersize(4)
            points.set_markeredgewidth(1)
            self._set_func(points.set_markeredgecolor, color, "b")
            self._set_func(points.set_markerfacecolor, color, "w")
            points.set_linestyle(" ")
            points.set_color("k")
            points.set_linewidth(0.50)
            points.set_label(label)

        elif "opt" in kind:
            points.set_marker("*")
            points.set_markersize(14)
            points.set_markeredgewidth(1.5)
            points.set_markeredgecolor('darkred')
            points.set_markerfacecolor('red')
            points.set_linestyle(" ")
            points.set_color("r")
            points.set_linewidth(0.50)
            points.set_label(label)

        elif "training" in kind:
            points.set_marker(".")
            points.set_markersize(10)
            points.set_markeredgewidth(1.5)
            points.set_markeredgecolor('k')
            points.set_markerfacecolor('k')
            points.set_linestyle(" ")
            points.set_color("k")
            points.set_linewidth(0.50)
            points.set_label(label)

        elif "validation" in kind:
            points.set_marker("*")
            points.set_markersize(10)
            points.set_markeredgewidth(1.5)
            points.set_markeredgecolor('k')
            points.set_markerfacecolor('k')
            points.set_linestyle(" ")
            points.set_color("k")
            points.set_linewidth(0.50)
            points.set_label(label)

        elif "dotted" in kind:
            points.set_marker("")
            points.set_markersize(1)
            points.set_markeredgewidth(0.35)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_linestyle("dotted")
            self._set_func(points.set_color, color, "k")
            points.set_linewidth(2)
            points.set_label(label)

        elif "points" in kind:
            points.set_marker("o")
            points.set_markersize(5)
            points.set_markeredgewidth(1)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_linestyle("")
            self._set_func(points.set_color, color, "k")
            points.set_linewidth(2)
            points.set_label(label)

    def _set_func(self, func, input, default):
        func(default) if input == None else func(input)

    def _set_axis_attribute(self, fig, ax, x_label, y_label, title, x_lim, y_lim, flag_aspect_ratio, flag_rotate_xticks=True):
        ax.set_xlabel(x_label, fontsize=12, color='k')
        ax.set_ylabel(y_label, fontsize=12, color='k')
        ax.set_title(title, fontsize=14, color='k')
        if not (x_lim is None): ax.set_xlim(x_lim)
        if not (y_lim is None): ax.set_ylim(y_lim)
        if flag_rotate_xticks: plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_aspect(1.00) if flag_aspect_ratio else fig.tight_layout()
        if flag_rotate_xticks: fig.subplots_adjust(bottom=0.20)

    def _save_chart(self, fig, extensions, filename, output_directory):
        for ext in extensions:
            filename_with_extension = f"{filename}.{ext}"
            filepath = os.path.join(output_directory, filename_with_extension)
            filedir = os.path.dirname(filepath)
            if not(os.path.exists(filedir)): os.mkdir(filedir)
            fig.savefig(filepath)
        self._export_data()

    def _show_chart(self, flag_show):
        if flag_show: plt.show()


