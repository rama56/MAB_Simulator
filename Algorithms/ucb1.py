from Algorithms.ucb import UCB

from Helpers.misc_helper import MiscellaneousHelper as mh


class UCB1(UCB):

    def __init__(self, arms, t, n):

        super().__init__(arms, t, n)

    def play_arms(self):
        rewards = [0]

        for t in range(1, self.K + 1):
            arm_number = t-1

            reward = super().pull_arm(arm_number)

            rewards.append(reward)

        for t in range(self.K + 1, self.T + 1):
            self.revise_ucbs(t)

            # pull the arm with highest UCB
            arm_with_highest_ucb = mh.get_maximum_index(self.upper_confidence_bound)

            reward = super().pull_arm(arm_with_highest_ucb)

            rewards.append(reward)

        return rewards

    def revise_ucbs(self, t):
        for i in range(self.K):
            self.upper_confidence_bound[i] = mh.textbook_radius(t, self.arms[i].pull_count) + \
                                             self.arms[i].empirical_mean
        # end for
    # end def






