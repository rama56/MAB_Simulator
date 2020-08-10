import matplotlib.pyplot as plt


class PlotHelper:

    # TODO: Make this a singleton class

    # Constant variables
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'burlywood', 'chartreuse', '0.5', '0.9', '0.2']

    is_x_axis_logarithmic = False
    fig = None
    ax = None
    x_axis_range = 0
    curves = []
    # curves is a list of tuples for now.
    # Might have to make it a list of objects if more sophistication is to be added.

    def __init__(self):
        # self.fig, self.ax = plt.subplots(figsize=(5, 5))
        pass

    def clear_curves(self):
        self.curves = []

    def add_curve(self, y_values, label, color_id, linestyle='solid'):
        curve_element = (y_values, label, color_id, linestyle)

        self.curves.append(curve_element)

        # Update x-Axis length
        size = len(y_values)
        if size > self.x_axis_range:
            self.x_axis_range = size

    # end def

    def plot_curves(self):
        # TODO: Check if y_values of all curves are of same length?
        x_axis = range(self.x_axis_range)

        i = 0
        for curve in self.curves:
            plt.plot(x_axis, curve[0], self.colours[curve[2]], label=curve[1], linestyle=curve[3])
            i = i + 1

        plt.legend(loc='upper left')
    # end def

    def set_logarithmic_x_axis(self, flag):
        self.is_x_axis_logarithmic = flag

    @staticmethod
    def initiate_figure(title, x_label, y_label,  x_log=True, y_log=True):
        fig = plt.figure(figsize=(9, 9))
        fig.suptitle(title)
        plt.grid(True)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if x_log:
            plt.xscale('log')

        if y_log:
            plt.yscale('log')

    @staticmethod
    def show_plots():
        plt.show()
