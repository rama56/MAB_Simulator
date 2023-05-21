'''
This script runs lots of instances of Slowly Varying Bandits with different algorithms
and plots the average regret of different algorithms.
'''
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


class NonStatExperimentRunner:
    logger = None

    def __init__(self, repetitions, inst_file_name, algorithms_to_run):
        self.repetitions = repetitions
        self.exp_id = 0

        self.algorithms_to_run = algorithms_to_run
        self.algo_count = len(algorithms_to_run)

        fw = fh.FileHelper()
        self.data = fw.read_json_from_file(file_name=inst_file_name)
        starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = self.data

        self.T = math.ceil(math.e ** logT)

        inst_svb = SlowlyVaryingBandits(k=2, t=self.T, delta=delta, start_means=starting_means, arms=[arm1, arm2],
                                        Deltas=Delta, Lambdas=Lambdas)
        inst_svb.fill_instance_specific_info()
        inst_svb.plot_instance()
        self.optimal_reward = inst_svb.cum_optimal_reward

        self.logger = LogHelper.get_logger(__name__)
        self.logger.info("Created experiment_runner for {0} repetitions with time horizon {1}".format(repetitions, self.T))

        self.emp_reg_runs = [[0] * self.algo_count for x in range(self.repetitions)]
        self.emp_reg_ave = [0] * self.algo_count
        self.emp_reg_stddev = [0] * self.algo_count
        # self.exp_regret_emp = np.zeros((self.algo_count, self.repetitions, self.T))
        self.average_arm_pulls_0 = [0] * self.algo_count
        self.average_arm_pulls_1 = [0] * self.algo_count

    # end __init__

    def run(self):
        for i in range(self.repetitions):
            self.logger.debug("MAB instance {0}".format(i))
            starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = self.data

            T = math.ceil(math.e ** logT)
            arm1.refill_tape()
            arm2.refill_tape()
            svb = SlowlyVaryingBandits(k=2, t=T, delta=delta, start_means=starting_means,arms=[arm1, arm2], Deltas=Delta, Lambdas=Lambdas)
            svb.set_algorithms(self.algorithms_to_run)

            start_time = time.time()
            svb.run_bandit_algorithm(verbose=True)
            end_time = time.time()
            print('Vectorized exec-time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

            svb.fill_instance_specific_info()
            svb.analyse_regret()
            # svb.plot_instance()
            # svb.plot_regret()
            # svb.plot_arm_pulls()
            # svb.show_plots()

            # self.update_average_regrets(svb)
            self.store_regret(svb, self.exp_id)
            self.exp_id = self.exp_id + 1

            self.update_average_arm_pulls(svb)

            svb = None
        self.compute_regret_ave_dev()

    def update_average_arm_pulls(self, svb):
        i=0
        for alg_obj in svb.algo_objs:
            alg = alg_obj[1]
            self.average_arm_pulls_0[i] = ((self.average_arm_pulls_0[i] * self.exp_id) +
                                          np.cumsum(alg.pull_0)) / (self.exp_id + 1)
            self.average_arm_pulls_1[i] = ((self.average_arm_pulls_1[i] * self.exp_id) +
                                           np.cumsum(alg.pull_1)) / (self.exp_id + 1)
            i=i+1

    def store_regret(self, svb, run_id):
        for i in range(self.algo_count):
            self.emp_reg_runs[run_id][i] = svb.cum_regret_empirical[i]

    def compute_regret_ave_dev(self):
        for i in range(self.algo_count):
            regret_sumed_over_reps = np.zeros(self.T)
            for run_id in range(self.repetitions):
                regret_sumed_over_reps += self.emp_reg_runs[run_id][i]

            self.emp_reg_ave[i] = regret_sumed_over_reps/self.repetitions

            deviation_summed_over_reps = np.zeros(self.T)
            for run_id in range(self.repetitions):
                deviation_summed_over_reps += np.square(self.emp_reg_runs[run_id][i]-self.emp_reg_ave[i])

            self.emp_reg_stddev[i] = np.sqrt(deviation_summed_over_reps/self.repetitions)
            x=5

    def update_average_regrets(self, svb):
        for i in range(self.algo_count):
            self.emp_reg_ave[i] = ((self.emp_reg_ave[i] * self.exp_id) +
                                       svb.cum_regret_empirical[i]) / (self.exp_id + 1)
    # end def

    def plot_regret_ave(self):
        plot_2 = PlotHelper(2, "Average Regret vs Time\n", "Time", "Average Regret",
                            x_log=False, y_log=False)
        # plot_2.set_exponential_x_axis()
        plot_2.clear_curves()
        # plot_2.add_curve(self.optimal_reward, "Optimal Reward", 0)
        # self.ph.add_curve(self.cum_regret_theo_bound, "Theoretical Upper Bound", 4)

        for i in range(self.algo_count):  # For each bandit algorithm,
            # plot_2.add_curve(self.cum_reward_empirical[i], "Empirical Reward " + self.algorithms_to_run[i][0], 2*i + 1)
            plot_2.add_curve(self.emp_reg_ave[i], self.algorithms_to_run[i][0], color_id=i + 1)
            plot_2.add_area(self.emp_reg_ave[i]-self.emp_reg_stddev[i]/2,
                            self.emp_reg_ave[i]+self.emp_reg_stddev[i]/2, color_id=i+1, transparency=0.5)

        plot_2.plot_curves()

    def plot_arm_pull_ave(self):
        plot_3 = PlotHelper(3, "Arm pulls Average\n", "Horizon t", "Pull Count",
                            x_log=False, y_log=False)
        plot_3.clear_curves()

        for i in range(self.algo_count):  # For each bandit algorithm,
            plot_3.add_curve(self.average_arm_pulls_0[i], "Arm 0 - " + self.algorithms_to_run[i][0], i + 4, linestyle=0)
            plot_3.add_curve(self.average_arm_pulls_1[i], "Arm 1 - " + self.algorithms_to_run[i][0], i + 4, linestyle=1)

        plot_3.plot_curves()

# end class


if __name__ == "__main__":
    # algorithms_to_run = [("Our algorithm", our_algo.OurAlgo)]
    algorithms_to_run = [
        ("SnoozeIt", our_algo.OurAlgo),
        ("SW-UCB\#", slidwind_ucb_hash.SlidWind_UCB_Hash),
        ("Rexp3", rexp3.RExp3),
        ("Exp3.S", exp3s.Exp3s)
    ]

    repetitions = 1
    instance_file_name = "instance_15.json"
    er = NonStatExperimentRunner(repetitions, instance_file_name, algorithms_to_run)

    er.run()

    # er.plot_graph(message)
    er.plot_regret_ave()
    # er.plot_arm_pull_ave()

    PlotHelper.show_plots()

# end
