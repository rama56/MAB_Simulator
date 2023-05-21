'''
This script runs lots of instances of Slowly Varying Bandits with different algorithms
and plots the average regret of different algorithms.
'''
# from Algorithms.ucb1 import UCB1
# from Algorithms.ucb_doubling import UCBDoubling
# from Algorithms.ucb_incremental import UCBIncremental
import datetime
import math
import numpy as np

from BanditInstance.multi_armed_bandit import MultiArmedBandit
from BanditInstance.slowly_varying_mab import SlowlyVaryingBandits
from Helpers.plot_helper import PlotHelper
from Helpers.log_helper import LogHelper
import Helpers.file_helper as fh
import time

from NonStationaryAlgorithms import our_algo, slidwind_ucb_hash, rexp3, exp3s


class NonStatOurAlgoExperimentRunner:
    logger = None

    def __init__(self, repetitions, inst_file_names):
        self.repetitions = repetitions
        self.inst_file_names = inst_file_names


        self.logger = LogHelper.get_logger(__name__)
        # self.logger.info("Created experiment_runner for {0} repetitions with time horizon {1}".format(repetitions, self.T))

        self.instances = []
        self.regrets = []


    # end __init__

    def run(self):

        for inst_file_name in self.inst_file_names:

            fw = fh.FileHelper()
            data = fw.read_json_from_file(file_name=inst_file_name)
            starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = data
            # inst_name = self.file_name_to_instance(inst_file_name)
            inst_name = self.delta_to_inst_name(delta)

            self.save_instance(arm1, arm2, inst_name)
            instance_id = 0
            self.T = math.ceil(math.e ** logT)

            emp_reg_runs = [0] * self.repetitions

            for i in range(self.repetitions):
                self.logger.debug("MAB instance: " + inst_file_name + ", Run: " + str(i))
                starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = data

                T = math.ceil(math.e ** logT)
                arm1.refill_tape()
                arm2.refill_tape()
                svb = SlowlyVaryingBandits(k=2, t=T, delta=delta, start_means=starting_means,arms=[arm1, arm2], Deltas=Delta, Lambdas=Lambdas)
                svb.set_algorithms(algorithms_to_run)

                # start_time = time.time()
                svb.run_bandit_algorithm(verbose=True)
                # end_time = time.time()
                # print('Vectorized exec-time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

                svb.fill_instance_specific_info()
                svb.analyse_regret()
                emp_reg_runs[i] = np.asarray(svb.cum_regret_empirical[0])  # Index 0 as there's only 1 algo.
                svb = None

            # End a repetition.

            # After repeating the instance multiple times, compute average performance for this instance.
            emp_reg_ave, emp_reg_stddev = self.compute_regret_ave_dev(emp_reg_runs)
            self.save_regrets(emp_reg_ave, emp_reg_stddev, inst_name)

            instance_id += 1
        # End an instance.

    def compute_regret_ave_dev(self, emp_reg_runs):
        regret_sumed_over_reps = np.zeros(self.T)
        for run_id in range(self.repetitions):
            regret_sumed_over_reps += emp_reg_runs[run_id]

        emp_reg_ave = regret_sumed_over_reps/self.repetitions

        deviation_summed_over_reps = np.zeros(self.T)
        for run_id in range(self.repetitions):
            deviation_summed_over_reps += np.square(emp_reg_runs[run_id]-emp_reg_ave)

        emp_reg_stddev = np.sqrt(deviation_summed_over_reps/self.repetitions)

        return emp_reg_ave, emp_reg_stddev

    def save_instance(self, arm1, arm2, inst_name):
        self.instances.append((arm1.means, arm2.means, inst_name))

    def save_regrets(self, emp_reg_ave, emp_reg_stddev, inst_name):
        self.regrets.append((emp_reg_ave, emp_reg_stddev, inst_name))

    def plot_instance(self):
        instances_plot = PlotHelper(1, "Instances", "Time", "True reward mean", x_log=False, y_log=False)
        instances_plot.clear_curves()
        color = 0
        for inst_obj in self.instances:
            mu_1, mu_2, inst_name = inst_obj
            instances_plot.add_curve(np.asarray(mu_1), inst_name , color, linestyle=0)
            instances_plot.add_curve(np.asarray(mu_2), '_' + inst_name + " (Arm 2)", color, linestyle=1)
            color += 1

        instances_plot.set_reward_y_axis_limits()
        instances_plot.plot_curves()

    def plot_regret(self):
        color = 0
        regret_plot = PlotHelper(2, "Average Regret", "Time", "Regret", x_log=False, y_log=False)
        regret_plot.clear_curves()
        for reg_obj in self.regrets:
            emp_reg_ave, emp_reg_stddev, inst_name = reg_obj
            regret_plot.add_curve(emp_reg_ave, inst_name, color_id=color)
            regret_plot.add_area(emp_reg_ave - emp_reg_stddev / 2,
                                  emp_reg_ave + emp_reg_stddev / 2, color_id=color, transparency=0.5)
            color += 1

        regret_plot.plot_curves()

    def file_name_to_instance(self, name):
        if name == "oscillation1.json":
            return r'$\delta = 0.44$ '
        elif name == "oscillation1.json":
            return r'$\delta = 0.5$ '

    def delta_to_inst_name(self, delta):
        return r'$\delta = $ ' + '{0:6f}'.format(delta)


if __name__ == "__main__":
    # algorithms_to_run = [("Our algorithm", our_algo.OurAlgo)]
    algorithms_to_run = [
        ("SnoozeIt", our_algo.OurAlgo),
        # ("SW-UCB#", slidwind_ucb_hash.SlidWind_UCB_Hash),
        # ("Rexp3", rexp3.RExp3),
        # ("Exp3.S", exp3s.Exp3s)
    ]

    repetitions = 10
    instance_file_name = ["oscillation1.json", "oscillation2.json", "oscillation3.json", "oscillation4.json"]
    # instance_file_name = ["instance_5.json"]
    er = NonStatOurAlgoExperimentRunner(repetitions, instance_file_name)
    er.run()
    er.plot_instance()
    er.plot_regret()
    PlotHelper.show_plots()
# end
