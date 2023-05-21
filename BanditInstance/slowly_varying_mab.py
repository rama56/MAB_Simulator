# This file contains the SlowlyVaryingBandits class that characterizes
# an instance of slowly varying bandits.
# It can be initialized using \delta and T, and it generates it's own \mu.
# Has setting specific analysis functions.

from BanditInstance import multi_armed_bandit as mab
from Helpers.non_stationary_helper import NonStationaryHelper as nsh
import numpy as np
from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.math_helper import MathHelper
from Helpers.plot_helper import PlotHelper
from NonStationaryAlgorithms.our_algo import OurAlgo


class SlowlyVaryingBandits(mab.MultiArmedBandit):

    # ph = None
    # logger = None

    """ Objects that are a part of the multi-armed-bandit setting. """
    # arms = []
    # true_means = None
    # K = 0
    # T = 0
    #
    # algorithms_to_run = None
    # algo_count = 0
    #
    # cum_reward_empirical = None

    """ Objects that are a part of the analysis of the performance. """

    mean_of_best_arm = []
    Deltas = []

    # # Cumulative regret, rewards are computed for all time-steps, as a list.
    # cum_optimal_reward = None
    #
    # cum_regret_empirical = None
    # cum_regret_theo_bound = None
    #
    # cum_pulls = None
    # theoretical_bounds_arm_pulls = None

    """ Functions to create, run, and analyse MABs. """

    def __init__(self, k=10, t=10**6, delta=0.1, start_means=None, arms=None, Deltas=None, Lambdas=None):
        super().__init__(k, t)

        self.delta = delta

        if arms is None:
            self.true_means, self.arms = nsh.get_slowly_varying_arms(self.K, self.T, delta, start_means)
        else:
            self.arms = arms
            self.true_means = [np.asarray(x.means) for x in self.arms]

        self.mean_of_best_arm = []
        self.Deltas = Deltas
        self.Lambdas_vect = Lambdas
        self.algo_objs = []
        self.our_algo = None

    def run_bandit_algorithm(self, verbose=False):
        self.algo_ran = True
        for i in range(self.algo_count):
            algo = self.algorithms_to_run[i]

            algo_name = algo[0]
            algo_class = algo[1]

            algorithm = algo_class(arms=self.arms, t=self.T, delta=self.delta, verbose=verbose)
            self.algo_objs.append((algo_name, algorithm))

            self.logger.debug("To start " + algo_name)

            algorithm.play_arms()

            algorithm.print_results()
            # algorithm.plot_results(self.ph)

            rewards_np_array = np.array(algorithm.rewards)
            self.cum_reward_empirical[i] = np.cumsum(rewards_np_array)

            # pull_matrix = algorithm.arm_pulled_at_time
            # pull_ndarray = np.array(pull_matrix)
            #
            # self.cum_pulls[i] = np.cumsum(pull_ndarray, axis=0)

            if type(algorithm) is OurAlgo:
                self.our_algo = algorithm

            for arm in self.arms:
                arm.reset_counts()
            # end for
        # end for
    # end run_bandit_algorithm

    def analyse_regret(self):
        # Compute cumulative empirical rewards
        for i in range(self.algo_count):
            # Compute cumulative empirical regret
            self.cum_regret_empirical[i] = self.cum_optimal_reward - self.cum_reward_empirical[i]
    # end def analyse_regret

    def plot_instance(self):

        # Instance params T and delta
        inst_details = "Time horizon, T =  " + str(self.T) + r', drift limit $\delta=$ ' + '{0:6f}'.format(self.delta)
        inst_details = None # Omitting title as we place legend there.
        plot_1 = PlotHelper(1, inst_details, "Time", "Reward", x_log=False, y_log=False)

        # True Means.
        plot_1.add_scatter(self.true_means[0], "Arm 1 - True reward mean", 1, linestyle=0)
        plot_1.add_scatter(self.true_means[1], "Arm 2 - True reward mean", 2, linestyle=0)

        # Delta & Lambda
        if self.Deltas is not None:
            plot_1.add_scatter(self.Deltas, "Gap", 3, linestyle=1)
        if self.Lambdas_vect is not None:
            plot_1.add_scatter(self.Lambdas_vect, "Detectable Gap", 7, linestyle=1)

        if self.our_algo is not None:
            self.our_algo.plot_episodic(plot_1)

        plot_1.set_reward_y_axis_limits()

        plot_1.set_patch_hadles()

        plot_1.plot_curves()

    def plot_regret(self):
        plot_2 = PlotHelper(2, "Regret vs Time\n", "Time", "Regret",
                                x_log=False, y_log=False)
        plot_2.clear_curves()
        # plot_2.add_curve(self.cum_optimal_reward, "Optimal Reward", 0)
        # self.ph.add_curve(self.cum_regret_theo_bound, "Theoretical Upper Bound", 4)

        for i in range(self.algo_count):  # For each bandit algorithm,
            # plot_2.add_curve(self.cum_reward_empirical[i], "Empirical Reward " + self.algorithms_to_run[i][0], 2*i + 1)
            plot_2.add_curve(self.cum_regret_empirical[i], self.algorithms_to_run[i][0], i + 1)

        plot_2.plot_curves()
    # end def

    def plot_arm_pulls(self, start_colour=1):
        plot_3 = PlotHelper(3, "Arm pulls vs Time\n", "Time", "Cumulative Pulls",
                                x_log=False, y_log=False)
        plot_3.clear_curves()

        c = start_colour
        for alg_obj in self.algo_objs:
            name = alg_obj[0]
            alg = alg_obj[1]
            plot_3.add_curve(np.cumsum(alg.pull_0), name + "- Arm 1", c, linestyle=0)
            plot_3.add_curve(np.cumsum(alg.pull_1), name + "- Arm 2", c, linestyle=1)
            c = c+1

        plot_3.plot_curves()

    """ Do something to print nicely 
        This is a temporary way to see the results."""
    def print_performance(self):

        print("Printing the cumulative values of different metrics..")

        print("Theoretical optimal reward ")
        print(self.cum_optimal_reward[-1])

        print("Empirical reward ")
        print(self.cum_reward_empirical[0][-1])

        # print("Theoretical regret bound ")
        # print(self.cum_regret_theo_bound[-1])

        print("Empirical regret ")
        print(self.cum_regret_empirical[0][-1])
    # end def

    # ANALYSE INSTANCE SPECIFIC STUFF, NOT ALGO SPECIFIC.
    def fill_instance_specific_info(self):
        # MU_1, MU_2 is known. Quantities for Regret Analysis.
        # self.mean_of_best_arm = [max(self.true_means[0][i], self.true_means[1][i]) for i in range(self.T)]
        self.mean_of_best_arm = np.maximum(self.true_means[0], self.true_means[1])
        self.cum_optimal_reward = np.cumsum(self.mean_of_best_arm)

        # Quantities for Instance hardness
        # self.fill_Deltas()
        # self.fill_Lambdas()
        # self.fill_Lambdas_vectorized()    # TODO - Commenting as it's taking long, and not so useful for analysis anyway.

    def show_plots(self):
        PlotHelper.show_plots()

    def fill_Deltas(self):
        # DELTA.
        self.Deltas = nsh.get_Delta_from_mu(self.true_means[0], self.true_means[1])

    def fill_Lambdas(self):
        if self.Deltas is None:
            self.Lambdas = nsh.get_lambda_from_mu(self.true_means[0], self.true_means[1])

    def fill_Lambdas_vectorized(self):
        if self.Lambdas_vect is None:
            self.Lambdas_vect = nsh.get_lambda_from_mu_vectorized(self.true_means[0], self.true_means[1])

        for i in range(len(self.Lambdas_vect)):
            if self.Lambdas_vect[i] == 0:
                if i!=0:
                    self.Lambdas_vect[i] = np.sqrt(144*12/i)


