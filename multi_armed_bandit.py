import numpy as np

from Algorithms.ucb_urgent import UCBUrgent
from arm import Arm
from Algorithms.ucb1 import UCB1
from Algorithms.ucb_urgent import UCBUrgent
from math_helper import MathHelper as rvh
from misc_helper import MiscellaneousHelper as mh
from plot_helper import PlotHelper


class MultiArmedBandit:

    ph = None

    """ Objects that are a part of the multi-armed-bandit setting. """
    arms = []
    true_means = None
    K = 0
    T = 0
    rewards = []
    rewards_urgent = []
    cum_reward_empirical = None
    cum_reward_empirical_urgent = None
    """ Objects that are a part of the analysis of the performance. """

    best_arm = None
    deltas = []

    # Cumulative regret, rewards are computed for all time-steps, as a list.
    cum_optimal_reward = None
    cum_regret_theo_bound = None

    cum_pulls_urgent = None
    cum_pulls = None

    cum_regret_empirical = None
    cum_regret_empirical_urgent = None

    theoretical_bounds_arm_pulls = None

    """ Functions to create, run, and analyse MABs. """

    def __init__(self, k=10, t=10**6):
        self.ph = PlotHelper()

        # Set the parameters of number of arms, time horizon arbitrarily.
        self.K = k
        self.T = t

        self.deltas = [0] * self.K

        # Create the arms
        self.true_means = rvh.get_uniform_sample(0, 1, self.K)
        for i in range(self.K):
            arm = Arm(self.true_means[i], size=self.T)
            self.arms.append(arm)

        # Create the algorithm object
        self.algorithm = UCB1(arms=self.arms, t=self.T, n=self.T)

    def run_bandit_algorithm(self):
        # Use the algorithm to pull the required number of arms in the required time.
        rewards_list = self.algorithm.play_arms()

        self.rewards = np.array(rewards_list)

        pull_matrix = self.algorithm.arm_pulls_by_time
        pull_ndarray = np.array(pull_matrix)

        self.cum_pulls = np.cumsum(pull_ndarray, axis=0)

        for arm in self.arms:
            arm.reset_counts()

        self.algorithm = UCBUrgent(arms=self.arms, t=self.T, n=self.T)
        rewards_urgent_list = self.algorithm.play_arms()
        self.rewards_urgent = np.array(rewards_urgent_list)

        pull_matrix = self.algorithm.arm_pulls_by_time
        pull_ndarray = np.array(pull_matrix)

        self.cum_pulls_urgent = np.cumsum(pull_ndarray, axis=0)

    def analyse_suboptimal_arm_pulls(self):
        # Compute deltas and theoretical upper bound of playing each sub-optimal arm.
        self.best_arm = mh.get_maximum_index(self.true_means)
        mean_of_best_arm = self.true_means[self.best_arm]

        for i in range(self.K):
            self.deltas[i] = mean_of_best_arm - self.true_means[i]

        del_sq_invs = mh.get_instance_dependent_square_inverses(self.deltas, self.best_arm)

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
        self.cum_reward_empirical = np.cumsum(self.rewards)
        self.cum_reward_empirical_urgent = np.cumsum(self.rewards_urgent)

        # Compute cumulative empirical regret
        self.cum_regret_empirical = self.cum_optimal_reward - self.cum_reward_empirical
        self.cum_regret_empirical_urgent = self.cum_optimal_reward - self.cum_reward_empirical_urgent

    # end def analyse_regret

    def plot_suboptimal_arm(self):

        self.ph.initiate_figure("#Pulls of sub-optimal arms vs Time", "Time T", "#Pulls", x_log=False, y_log=True)

        for col in range(self.K):
            theoretical_bound = self.theoretical_bounds_arm_pulls[:, col]
            empirical_pulls = self.cum_pulls[:, col]
            empirical_pulls_urgent = self.cum_pulls_urgent[:, col]
            # empirical_pulls_urgent

            # ['solid' | 'dashed', 'dashdot', 'dotted' | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':'
            # | 'None' | ' ' | '' ]

            if col != self.best_arm:
                self.ph.add_curve(theoretical_bound, mh.stringify(self.true_means[col]) + " theo", col, 'solid')

            self.ph.add_curve(empirical_pulls, mh.stringify(self.true_means[col]) + " UCB1", col, 'dotted')
            self.ph.add_curve(empirical_pulls_urgent, mh.stringify(self.true_means[col]) + " UCB-Inc", col, 'dashed')

        self.ph.plot_curves()

    def plot_regret(self):

        self.ph.clear_curves()
        self.ph.initiate_figure("Regret of algorithms vs Time", "Time T", "Regret", x_log=True, y_log=False)

        # ph.add_curve(self.cum_optimal_reward, "Optimal Reward", 1)
        # ph.add_curve(self.cum_reward_empirical, "Empirical Reward", 2)
        # ph.add_curve(self.cum_reward_empirical_urgent, "Empirical Reward" Urgent, 3)

        self.ph.add_curve(self.cum_regret_theo_bound, "Theoretical Upper Bound", 4)
        self.ph.add_curve(self.cum_regret_empirical, "UCB1", 5)
        self.ph.add_curve(self.cum_regret_empirical_urgent, "UCB-Inc", 6)

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

    # end def

    def analyse_common_stats(self):
        # Compute deltas and theoretical upper bound of regret of UCB1.
        self.best_arm = mh.get_maximum_index(self.true_means)
        mean_of_best_arm = self.true_means[self.best_arm]

        for i in range(self.K):
            self.deltas[i] = mean_of_best_arm - self.true_means[i]

        sum_del_inv, sum_del = mh.get_instance_dependent_values(self.best_arm, self.deltas)

        mult_constant, addi_constant = mh.get_theoretical_constants(sum_del_inv, sum_del)

        time_series = np.arange(self.T + 1)
        self.cum_regret_theo_bound = mult_constant * rvh.natural_logarithm(time_series) + addi_constant
        self.cum_optimal_reward = time_series * mean_of_best_arm
