'''
This script runs lots of instances of MABs with different algorithms and plots the average regret of
different algorithms
'''
from Algorithms.ucb1 import UCB1
from Algorithms.ucb_doubling import UCBDoubling
from Algorithms.ucb_incremental import UCBIncremental
from multi_armed_bandit import MultiArmedBandit
from Helpers.plot_helper import PlotHelper
from Helpers.log_helper import LogHelper
from Helpers.misc_helper import MiscellaneousHelper as mh


class ExperimentRunner:
    total_experiment_count = 0
    arms_count_k = 0
    arm_pulls_n = 0

    current_experiment_count = 0

    algorithms_to_run = None
    algo_count = 0

    average_regret_theo = 0
    average_regret_emp = 0

    logger = None

    def __init__(self, exp_cnt, k, n, algorithms_to_run):
        self.total_experiment_count = exp_cnt
        self.arms_count_k = k
        self.arm_pulls_n = n

        self.logger = LogHelper.get_logger(__name__)
        self.logger.info("Created experiment_runner for {0} repetitions, {1} arms, {2} arm pulls".format(exp_cnt, k, n))

        self.algorithms_to_run = algorithms_to_run
        self.algo_count = len(algorithms_to_run)

        self.average_regret_emp = [0] * self.algo_count

    # end __init__

    def run(self):

        for i in range(self.total_experiment_count):
            self.logger.debug("MAB instance {0}".format(i))

            mab = None
            mab = MultiArmedBandit(k=self.arms_count_k, t=self.arm_pulls_n)
            mab.set_algorithms(self.algorithms_to_run)
            mab.run_bandit_algorithm()
            mab.analyse_regret()

            self.update_average_regrets(mab)
            self.current_experiment_count = self.current_experiment_count + 1

    def update_average_regrets(self, mab):

        self.average_regret_theo = ((self.average_regret_theo * self.current_experiment_count) +
                                    mab.cum_regret_theo_bound) / (self.current_experiment_count + 1)

        for i in range(self.algo_count):
            self.average_regret_emp[i] = ((self.average_regret_emp[i] * self.current_experiment_count) +
                                       mab.cum_regret_empirical[i]) / (self.current_experiment_count + 1)

    # end def

    def plot_graph(self):
        ph = PlotHelper()
        ph.initiate_figure("Average regret vs Time", "Time", "Regret", x_log=False, y_log=False)

        ph.add_curve(self.average_regret_theo, "Theoretical Upper Bound", 4)

        for i in range(self.algo_count):  # For each bandit algorithm,
            ph.add_curve(self.average_regret_emp[i], self.algorithms_to_run[i][0], 5+i)

        ph.plot_curves()
        ph.show_plots()


# end class


if __name__ == "__main__":

    algorithms_to_run = [("UCB1", UCB1), ("UCB-Inc", UCBIncremental), ("UCB-Doub", UCBDoubling)]
        # ,("UCB-Doub-TR", UCBDoubling, mh.ucb_doubling_radius)]

    er = ExperimentRunner(2, 5, 10 ** 5, algorithms_to_run)

    er.run()

    er.plot_graph()

# end
