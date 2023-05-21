import abc  # Abstract Base Class
from Helpers.misc_helper import MiscellaneousHelper as mh

from Helpers.math_helper import MathHelper as rvh

#  DELETE THIS CLASS.

class NonStat_MultiArmedBandits:
    _tape = [-1]
    _reward_means = [-1]
    delta = None

    def __init__(self, T, K):
        self.T = T  # Time Horizon
        self.time_step = 1
        self.K = K  # Number of arms

        # self.rewardGeneration = Taped
        # (the other being generating rewards iid when arm is pulled)

    def create_means(self, delta):
        self.delta = delta
        for i in range(1, self.K):
            mu_i = [-1]
            mu_i_1 = rvh.get_uniform_sample(0, 1)
            mu_i.append(mu_i_1)
            for t in range(2, self.T+1):
                mu_i_t = mu_i[t-1] + rvh.get_uniform_sample(-delta, delta)
                mu_i.append(mu_i_t)

            self._reward_means.append(mu_i)

    def fill_tapes(self):
        for i in range(1, self.K):
            muhat_i = [-1]
            for t in range(1, self.T+1):
                muhat_i_t = rvh.get_bernoulli_sample(self._reward_means[i][t])
                muhat_i.append(muhat_i_t)

            self._tape.append(muhat_i)

    def pull_arm(self, arm):
        # Get the reward from the arm (implicitly at current self.time_step).
        reward = self._tape[arm][self.time_step]
        self.time_step = self.time_step+1

        return reward






