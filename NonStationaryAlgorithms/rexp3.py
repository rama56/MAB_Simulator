from NonStationaryAlgorithms.bandit_algorithm import BanditAlgorithm

from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.math_helper import MathHelper
import numpy as np


class RExp3(BanditAlgorithm):

    def __init__(self, arms, t, delta, verbose=False):

        super().__init__(arms, t, delta, verbose=verbose)
        self.rewards = []

        TwoLogTwo = 2*MathHelper.natural_logarithm(2)
        # Fix params
        self.v_budget = delta * t    # Variation budget V_T
        self.batch_size = MathHelper.ceil(
            (TwoLogTwo)**(1/3) * (t/self.v_budget)**(2/3))
        self.num_batches = MathHelper.ceil(t/self.batch_size)

        self.gamma = min(1, np.sqrt(TwoLogTwo/((np.e - 1)*self.batch_size)))

    def play_arms(self):
        # Initialize
        w_0 = 1
        w_1 = 1
        batch = 1
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
            w_0 = w_0 * (np.e ** (self.gamma*x_0/2))
            w_1 = w_1 * (np.e ** (self.gamma*x_1/2))

            # RESET BATCH
            if t >= batch * self.batch_size - 1:
                batch = batch + 1
                w_0 = 1
                w_1 = 1

    def plot_episodic(self, plot_obj):
        pass

    def print_results(self):
        print("Variation Budget = " + str(self.v_budget))
        print("Batch size = " + str(self.batch_size))
        print("Num of batches = " + str(self.num_batches))
        print("Gamma = " + str(self.gamma))







