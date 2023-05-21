import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
import matplotlib.patches as mpatches


class PlotHelper:

    # Constant variables
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'burlywood', 'chartreuse', '0.5', '0.9', '0.2']
    line_styles = ['solid', 'dashed', 'dotted', 'dashdot', '-', '--', '-.', ':', 'None', ' ', '']

    fig = None
    ax = None
    x_axis_range = 0
    curves = []
    areas = []
    scatter = []
    # curves is a list of tuples for now.
    # Might have to make it a list of objects if more sophistication is to be added.
    vlines = []

    def __init__(self, id, title, x_label, y_label,  x_log=True, y_log=True):
        self.fig = plt.figure(id, figsize=(9, 9))
        self.ax = self.fig.add_subplot()
        self.fig.suptitle(title)
        self.ax.grid(True)

        # plt.xlabel(x_label)
        # plt.ylabel(y_label)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

        if x_log:
            self.ax.set_xscale('log')

        if y_log:
            self.ax.set_yscale('log')

    def clear_curves(self):
        self.curves = []
        self.vlines = []
        self.scatter = []

    def add_vline(self, x_value, y_range, label, label_y, color_id, linestyle=0):
        vline_element = (x_value, y_range, label, label_y, color_id, linestyle)
        self.vlines.append(vline_element)

    def add_scatter(self, y_values, label, color_id, linestyle=0):
        scatter_element = (y_values, label, color_id, linestyle)
        self.scatter.append(scatter_element)
        # Update x-Axis length
        size = len(y_values)
        if size > self.x_axis_range:
            self.x_axis_range = size
    # end def

    def add_curve(self, y_values, label, color_id, linestyle=0):
        curve_element = (y_values, label, color_id, linestyle)
        self.curves.append(curve_element)
        # Update x-Axis length
        size = len(y_values)
        if size > self.x_axis_range:
            self.x_axis_range = size
    # end def

    def add_area(self, y_1, y_2, color_id, transparency):
        area = (y_1, y_2, color_id, transparency)
        self.areas.append(area)

    def plot_curves(self):
        # TODO: Check if y_values of all curves are of same length?
        x_axis = range(self.x_axis_range)

        i = 0
        for curve in self.curves:
            xrange = len(curve[0])
            self.ax.plot(range(xrange), curve[0], self.colours[curve[2]],
                         label=curve[1], linestyle=self.line_styles[curve[3]])
            i = i + 1

        for scat in self.scatter:
            xrange = len(scat[0])
            self.ax.scatter(range(xrange), scat[0], s=1, c=self.colours[scat[2]])
            # , label=scat[1]) Omitting adding label as we create legend separately.
            i = i + 1

        for area in self.areas:
            y_1, y_2, color_id, transparency = area
            col = self.colours[color_id]
            xrange = range(len(y_1))
            self.ax.fill_between(xrange, y_1, y_2, facecolor=col, alpha=transparency)

        for vline in self.vlines:
            x_val = vline[0]
            label = vline[2]
            y_val = vline[3]
            col = self.colours[vline[4]]
            ls = self.line_styles[vline[5]]
            self.ax.vlines(vline[0], 0, 1, linestyles=ls, colors=col)
            self.ax.text(x_val + 100, y_val, label, rotation=90, verticalalignment='center')

        # self.ax.legend(loc='upper left') Omitting positioning legend as we create legend separately.
    # end def

    def set_title(self, tit_text):
        self.fig.suptitle(tit_text)

    # Stack overflow warns. https://stackoverflow.com/questions/43392623/get-the-title-of-a-given-figure
    def get_title(self):
        return self.fig._suptitle.get_text()

    @staticmethod
    def show_plots():
        plt.show()

    def set_reward_y_axis_limits(self):
        self.ax.set_ylim([-0.05, 1.05])

    def set_exponential_x_axis(self):
        def sqr(x):
            return x**2

        def root(x):
            return x**(1/2)

        self.ax.set_xscale('function', functions=(sqr, root))

    def set_patch_hadles(self):
        handles, labels = self.ax.get_legend_handles_labels()

        patches = []
        for scat in self.scatter:
            y_val = scat[0]
            label = scat[1]
            col_id = scat[2]
            patch = mpatches.Patch(color=PlotHelper.colours[col_id], label=label)
            patches.append(patch)

        eps_start = mpatches.Patch(color=PlotHelper.colours[0], label="Episode start time")  # Blue
        stat_test = mpatches.Patch(color=PlotHelper.colours[4], label="Statistical test pass time")     # Magenta

        patches.append(eps_start)
        patches.append(stat_test)

        handles = patches
        # [patch1, patch2, patch3, patch4]

        # self.fig.legend(handles, labels, loc='upper center', ncol=4)
        self.fig.legend(handles=handles, loc='upper center', ncol=3)


