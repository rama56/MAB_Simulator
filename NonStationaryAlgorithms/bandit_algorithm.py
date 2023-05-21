import abc  # Abstract Base Class
from Helpers.misc_helper import MiscellaneousHelper as mh
import numpy as np


class BanditAlgorithm(abc.ABC):
    arms = 0
    T = 0   # Horizon limit of time.
    K = 0   # Number of arms
    delta = 0   # Step-wise drift
    arm_pulled_at_time = None
    verbose = False
    # The time variable that the play_arms() function should iterate.
    t = 0

    def __init__(self, arms, t, delta, verbose=False):

        self.arms = arms
        self.T = t
        self.delta = delta

        self.K = len(self.arms)

        self.arm_pulled_at_time = []
        self.rewards = [-1] * self.T

        # For vectorized statistical test
        self.reward_0 = np.zeros(t)
        self.reward_1 = np.zeros(t)
        # For vectorized statistical test - End
        self.pull_0 = np.zeros(t)
        self.pull_1 = np.zeros(t)

        self.verbose = verbose

    @abc.abstractmethod
    def play_arms(self):
        pass

    @abc.abstractmethod
    def print_results(self):
        pass

    def print_verbose(self, text):
        if self.verbose:
            print(text)

    def record_pull(self, arm_number):
        # assert len(zeros) == self.K
        self.arm_pulled_at_time.append(arm_number)

    def record_pull_vect(self, arm_to_pull, t, reward):
        if arm_to_pull == 0:
            self.reward_0[t] = reward
            self.pull_0[t] = 1
            # self.cnt_0 = self.cnt_0 + 1
        else:
            self.reward_1[t] = reward
            self.pull_1[t] = 1
            # self.cnt_1 = self.cnt_1 + 1

    def pull_arm(self, arm_number, t):
        # Get the reward from the arm.
        reward = self.arms[arm_number].pull(t)

        self.record_pull(arm_number)

        # Return the reward
        return reward








