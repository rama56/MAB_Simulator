'''
This script runs lots of instances of MABs with different algorithms and plots the average regret of
different algorithms
'''

from multi_armed_bandit import MultiArmedBandit
from Helpers.plot_helper import PlotHelper
from Helpers.log_helper import LogHelper


class ExperimentRunner:
    total_experiment_count = 0
    arms_count_k = 0
    arm_pulls_n = 0

    current_experiment_count = 0

    average_regret_theo = 0
    average_regret_ucb1 = 0
    average_regret_ucb_inc = 0
    average_regret_ucb_doub = 0
    average_regret_ucb_doub_tr = 0

    logger = None

    def __init__(self, exp_cnt, k, n):
        self.total_experiment_count = exp_cnt
        self.arms_count_k = k
        self.arm_pulls_n = n

        self.logger = LogHelper.get_logger(__name__)
        self.logger.info("Created experiment_runner for {0} repetitions, {1} arms, {2} arm pulls".format(exp_cnt, k, n))
    # end __init__

    def run(self):
        for i in range(self.total_experiment_count):
            self.logger.debug("MAB instance {0}".format(i))

            mab = None
            mab = MultiArmedBandit(k=self.arms_count_k, t=self.arm_pulls_n)
            mab.run_bandit_algorithm()
            mab.analyse_regret()

            self.update_average_regrets(mab)
            self.current_experiment_count = self.current_experiment_count + 1

    def update_average_regrets(self, mab):
        self.average_regret_ucb1 = ((self.average_regret_ucb1 * self.current_experiment_count) +
                                    mab.cum_regret_empirical) / (self.current_experiment_count + 1)

        self.average_regret_ucb_inc = ((self.average_regret_ucb_inc * self.current_experiment_count) +
                                       mab.cum_regret_empirical_incremental) / (self.current_experiment_count + 1)

        self.average_regret_ucb_doub = ((self.average_regret_ucb_doub * self.current_experiment_count) +
                                        mab.cum_regret_empirical_doubling) / (self.current_experiment_count + 1)

        self.average_regret_ucb_doub_tr = ((self.average_regret_ucb_doub_tr * self.current_experiment_count) +
                                        mab.cum_regret_empirical_doubling_tr) / (self.current_experiment_count + 1)

        self.average_regret_theo = ((self.average_regret_theo * self.current_experiment_count) +
                                    mab.cum_regret_theo_bound) / (self.current_experiment_count + 1)


    # end def

    def plot_graph(self):
        ph = PlotHelper()
        ph.initiate_figure("Average regret vs Time", "Time", "Regret", x_log=False, y_log=False)

        ph.add_curve(self.average_regret_theo, "Theoretical Upper Bound", 4)
        ph.add_curve(self.average_regret_ucb1, "UCB1", 5)
        ph.add_curve(self.average_regret_ucb_inc, "UCB-Inc", 6)
        ph.add_curve(self.average_regret_ucb_doub, "UCB-Doub", 7)
        ph.add_curve(self.average_regret_ucb_doub_tr, "UCB-Doub-TR", 8)

        ph.plot_curves()
        ph.show_plots()

# end class


if __name__ == "__main__":
    er = ExperimentRunner(5, 3, 10**4)
    er.run()

    er.plot_graph()

# end
