import numpy as np

from Helpers.log_helper import LogHelper
from Helpers.math_helper import MathHelper as rvh
from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.plot_helper import PlotHelper


class MultiArmedBandit:

    ph = None
    logger = None

    """ Objects that are a part of the multi-armed-bandit setting. """
    arms = []
    true_means = None
    K = 0
    T = 0

    algorithms_to_run = None
    algo_count = 0

    algo_ran = False
    cum_reward_empirical = None

    """ Objects that are a part of the analysis of the performance. """

    best_arm = None
    Deltas = []

    # Cumulative regret, rewards are computed for all time-steps, as a list.
    cum_optimal_reward = None

    cum_regret_empirical = None
    cum_regret_theo_bound = None

    cum_pulls = None
    theoretical_bounds_arm_pulls = None

    """ Functions to create, run, and analyse MABs. """
    def __init__(self, k=10, t=10**6):
        # self.ph = PlotHelper()
        self.logger = LogHelper.get_logger(__name__)

        # Set the parameters of number of arms, time horizon arbitrarily.
        self.K = k
        self.T = t
        self.Deltas = [0] * self.K

        # Commenting out as I'm anyway overwriting in the slowly_varying_mab.py anyway.
        # # Create the arms
        # self.true_means, self.arms = mh.get_arms(self.K, self.T)

    def set_algorithms(self, algorithms_to_run):
        self.algorithms_to_run = algorithms_to_run
        self.algo_count = len(algorithms_to_run)

        self.cum_pulls = [None] * self.algo_count
        self.cum_regret_empirical = [None] * self.algo_count
        self.cum_reward_empirical = [None] * self.algo_count

    def run_bandit_algorithm(self, verbose=False):

        for i in range(self.algo_count):
            algo = self.algorithms_to_run[i]

            algo_name = algo[0]
            algo_class = algo[1]

            algorithm = algo_class(arms=self.arms, t=self.T, n=self.T, verbose=verbose)

            if len(algo) > 2:
                algorithm = algo_class(arms=self.arms, t=self.T, n=self.T, radius_function=algo[2])

            self.logger.debug("To start " + algo_name)

            rewards_list = algorithm.play_arms()

            rewards_np_array = np.array(rewards_list)
            self.cum_reward_empirical[i] = np.cumsum(rewards_np_array)

            pull_matrix = algorithm.arm_pulled_at_time
            pull_ndarray = np.array(pull_matrix)

            self.cum_pulls[i] = np.cumsum(pull_ndarray, axis=0)

            for arm in self.arms:
                arm.reset_counts()
            # end for
        # end for
    # end run_bandit_algorithm

    def analyse_suboptimal_arm_pulls(self):
        # Compute deltas and theoretical upper bound of playing each sub-optimal arm.
        self.best_arm = mh.get_maximum_index(self.true_means)
        mean_of_best_arm = self.true_means[self.best_arm]

        for i in range(self.K):
            self.Deltas[i] = mean_of_best_arm - self.true_means[i]

        del_sq_invs = mh.get_instance_dependent_square_inverses(self.Deltas, self.best_arm)

        addi_constant = rvh.func_of_pi(add=1, power=2, mult=1 / 3)

        time_series = np.arange(self.T + 1)

        logarithmic_time_series = rvh.natural_logarithm(time_series)

        a = np.array(del_sq_invs)
        del_sq_inv_row_matrix = np.reshape(a, (1, -1))
        logarithmic_time_series_column_matrix = np.reshape(logarithmic_time_series, (-1, 1))

        matrix = np.dot(logarithmic_time_series_column_matrix, del_sq_inv_row_matrix)

        self.theoretical_bounds_arm_pulls = matrix + addi_constant

    def analyse_regret(self):

        # Compute basic stats
        self.analyse_common_stats()

        # Compute cumulative empirical rewards

        for i in range(self.algo_count):
            # Compute cumulative empirical regret
            self.cum_regret_empirical[i] = self.cum_optimal_reward - self.cum_reward_empirical[i]
    # end def analyse_regret

    def plot_suboptimal_arm(self):

        self.ph.initiate_figure("#Pulls of sub-optimal arms vs Time", "Rounds n", "#Pulls", x_log=False, y_log=True)

        for col in range(self.K):   # For each arm,
            theoretical_bound = self.theoretical_bounds_arm_pulls[:, col]

            if col != self.best_arm:
                self.ph.add_curve(theoretical_bound, mh.stringify(self.true_means[col]) + " Theo", col, 0)

            for i in range(self.algo_count):  # For each bandit algorithm,

                empirical_pulls = self.cum_pulls[i][:, col]
                self.ph.add_curve(empirical_pulls, mh.stringify(self.true_means[col]) + self.algorithms_to_run[i][0],
                                  col, i+1)

        self.ph.plot_curves()

    def plot_regret(self):

        self.ph.clear_curves()
        true_means_string = "True means of arms: " + mh.stringify_list(self.true_means)

        self.ph.initiate_figure("Regret of algorithms vs Time\n" + true_means_string, "Rounds n", "Regret",
                                x_log=False, y_log=False)

        # ph.add_curve(self.cum_optimal_reward, "Optimal Reward", 1)
        # ph.add_curve(self.cum_reward_empirical, "Empirical Reward", 2)
        # ph.add_curve(self.cum_reward_empirical_incremental, "Empirical Reward" incremental, 3)

        self.ph.add_curve(self.cum_regret_theo_bound, "Theoretical Upper Bound", 4)

        for i in range(self.algo_count):  # For each bandit algorithm,
            self.ph.add_curve(self.cum_regret_empirical[i], self.algorithms_to_run[i][0], 5+i)

        self.ph.plot_curves()

    # end def

    """ Do something to print nicely 
        This is a temporary way to see the results."""
    def print_performance(self):

        print("Printing the cumulative values of different metrics..")

        print("Theoretical optimal reward ")
        print(self.cum_optimal_reward[-1])

        print("Empirical reward ")
        print(self.cum_reward_empirical[-1])

        print("Theoretical regret bound ")
        print(self.cum_regret_theo_bound[-1])

        print("Empirical regret ")
        print(self.cum_regret_empirical[-1])

        # Commented out
        # self.logger.info("Printing the cumulative values of different metrics..")
        #
        # self.logger.info("Theoretical optimal reward = {0}".format(self.cum_regret_theo_bound[-1]))
        #
        # self.logger.info("UCB1 Empirical reward = {0}".format(self.cum_optimal_reward[-1]))
        #
        # self.logger.info("UCB-Inc Empirical reward = {0}".format(self.cum_optimal_reward[-1]))
        #
        # self.logger.info("UCB-Doub Empirical reward = {0}".format(self.cum_optimal_reward[-1]))
        #
        # self.logger.info("UCB-Doub-TR Empirical reward = {0}".format(self.cum_optimal_reward[-1]))
    # end def

    def analyse_common_stats(self):
        # Compute deltas and theoretical upper bound of regret of UCB1.
        self.best_arm = mh.get_maximum_index(self.true_means)
        mean_of_best_arm = self.true_means[self.best_arm]

        for i in range(self.K):
            self.Deltas[i] = mean_of_best_arm - self.true_means[i]

        sum_del_inv, sum_del = mh.get_instance_dependent_values(self.best_arm, self.Deltas)

        mult_constant, addi_constant = mh.get_theoretical_constants(sum_del_inv, sum_del)

        time_series = np.arange(self.T + 1)
        self.cum_regret_theo_bound = mult_constant * rvh.natural_logarithm(time_series) + addi_constant
        self.cum_optimal_reward = time_series * mean_of_best_arm
