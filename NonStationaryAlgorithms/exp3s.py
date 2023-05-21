from NonStationaryAlgorithms.bandit_algorithm import BanditAlgorithm

from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.math_helper import MathHelper
import numpy as np


class Exp3s(BanditAlgorithm):

    def __init__(self, arms, t, delta, verbose=False):

        super().__init__(arms, t, delta, verbose=verbose)
        self.rewards = []

        TwoLogTwo = 2*MathHelper.natural_logarithm(2)
        # Fix params
        self.v_budget = delta * t    # Variation budget V_T
        self.alph = 1/t
        self.gamma = min(1, (4*self.v_budget*2 * MathHelper.natural_logarithm(2*t) / ((np.e-1)**2 * t))*(1/3))  # Woof!

    def play_arms(self):
        # Initialize
        w_0 = 1
        w_1 = 1
        for t in range(self.T):
            p_0 = (1-self.gamma)*(w_0/(w_0+w_1)) + self.gamma * (1/2)
            p_1 = (1-self.gamma)*(w_1/(w_0+w_1)) + self.gamma * (1/2)

            arm_to_pull = MathHelper.get_bernoulli_sample(p_1)

            reward = super().pull_arm(arm_to_pull, t)
            self.rewards.append(reward)
            self.record_pull_vect(arm_to_pull, t, reward)

            x_0 = 0
            x_1 = 0
            if arm_to_pull == 0:
                x_0 = reward/p_0
            else:
                x_1 = reward/p_1

            # ADJUST WEIGHTS
            w_0 = w_0 * (np.e**(self.gamma*x_0/2)) + (np.e * self.alph/2)*(w_0+w_1)
            w_1 = w_1 * (np.e**(self.gamma*x_1/2)) + (np.e * self.alph/2)*(w_0+w_1)

    def plot_episodic(self, plot_obj):
        pass

    def print_results(self):
        print("Variation Budget = " + str(self.v_budget))
        print("Gamma = " + str(self.gamma))







