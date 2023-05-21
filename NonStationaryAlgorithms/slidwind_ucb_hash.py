from NonStationaryAlgorithms.bandit_algorithm import BanditAlgorithm

from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.math_helper import MathHelper
import numpy as np


class SlidWind_UCB_Hash(BanditAlgorithm):

    def __init__(self, arms, t, delta, verbose=False):

        super().__init__(arms, t, delta, verbose=verbose)
        # upper_confidence_bound = None
        # radius_function = None

        self.rewards = []

        # Fix params
        self.l = 4.3     # Lambda parameter in the paper.
        self.kappa = - MathHelper.custom_log(delta, t)
        self.alph = min(1, 3*self.kappa/4)    # Alpha parameter in the paper.

    def play_arms(self):

        for t in range(self.K):
            arm_to_pull = t
            reward = super().pull_arm(arm_to_pull, t)
            self.rewards.append(reward)

            self.record_pull_vect(arm_to_pull, t, reward)

        for t in range(self.K, self.T):

            # Sliding Window size
            w = min(MathHelper.ceil(self.l*(t**self.alph)), t)

            w_reward_0 = self.reward_0[t-w:t-1]
            w_reward_1 = self.reward_1[t-w:t-1]
            w_cum_reward_0 = np.sum(w_reward_0)
            w_cum_reward_1 = np.sum(w_reward_1)

            w_pull_0 = self.pull_0[t-w:t-1]
            w_pull_1 = self.pull_1[t-w:t-1]
            w_cnt_0 = np.sum(w_pull_0)
            w_cnt_1 = np.sum(w_pull_1)

            w_muhat_0 = w_cum_reward_0/w_cnt_0
            w_muhat_1 = w_cum_reward_1/w_cnt_1

            logt = MathHelper.natural_logarithm(t)
            w_radius_0 = np.sqrt((1+self.alph)*logt / w_cnt_0)
            w_radius_1 = np.sqrt((1+self.alph)*logt / w_cnt_1)

            ucb_0 = w_muhat_0 + w_radius_0
            ucb_1 = w_muhat_1 + w_radius_1

            arm_to_pull = None
            if ucb_0 > ucb_1:
                arm_to_pull = 0
            else:
                arm_to_pull = 1

            reward = super().pull_arm(arm_to_pull, t)
            self.rewards.append(reward)
            self.record_pull_vect(arm_to_pull, t, reward)

    def plot_episodic(self, plot_obj):
        pass

    def print_results(self):
        print("Kappa = " + str(self.kappa))
        print("Alpha = " + str(self.alph))
        time_junctions = [self.T/4, self.T/2, 3*self.T/4, self.T]

        print("Max window (ell * t^alpha) at different time-steps.")
        for junction in time_junctions:
            print("Time: " + str(junction))
            w = MathHelper.ceil(self.l*(junction**self.alph))
            print("Window: " + str(w))








