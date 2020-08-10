from Algorithms.ucb import UCB

from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.math_helper import MathHelper as rvh


class UCBDoubling(UCB):

    radius_function = None

    def __init__(self, arms, t, n, radius_function=None):

        super().__init__(arms, t, n)

        if radius_function is not None:
            self.radius_function = radius_function

    def play_arms(self):
        rewards = [0]
        n = 0

        # At time t = 0,
        for i in range(1, self.K + 1):
            arm_number = i-1
            reward = super().pull_arm(arm_number)
            rewards.append(reward)

            n = n + 1

        # From time t = 1
        for t in range(1, rvh.ceiled_log_base_2(self.N) + 1):
            self.revise_ucbs(n)

            # pull the arm with highest UCB 2 power t times
            pulls_this_iteration = 2 ** t

            arm_with_highest_ucb = mh.get_maximum_index(self.upper_confidence_bound)

            for i in range(pulls_this_iteration):
                if n >= self.N:
                    break

                reward = super().pull_arm(arm_with_highest_ucb)
                rewards.append(reward)
                n = n + 1
            # end for
        # end for

        return rewards

    def revise_ucbs(self, t):
        for i in range(self.K):
            self.upper_confidence_bound[i] = self.radius_function(t, self.arms[i].pull_count) + \
                                             self.arms[i].empirical_mean
        # end for
    # end def






