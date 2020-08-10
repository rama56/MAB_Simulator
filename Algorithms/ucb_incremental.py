from Algorithms.ucb import UCB

from Helpers.misc_helper import MiscellaneousHelper as mh


class UCBIncremental(UCB):

    def __init__(self, arms, t, n):

        super().__init__(arms, t, n)

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
        for t in range(1, mh.ciel_root(self.N) + 1):
            self.revise_ucbs(n)

            # pull the arm with highest UCB 2t-1 times
            pulls_this_iteration = 2 * t - 1

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
            self.upper_confidence_bound[i] = mh.textbook_radius(t, self.arms[i].pull_count) + \
                                             self.arms[i].empirical_mean
        # end for
    # end def






