
from unittest import TestCase

from arm import Arm
from multi_armed_bandit import MultiArmedBandit


class Test(TestCase):

    def test_arm(self):
        arm = Arm(0.3)
        rewards = []
        T = 20

        for t in range(1, T+1):
            reward = arm.pull()
            rewards.append(reward)

        total = sum(rewards)

        assert arm.pull_count == 20

        x = 5
    # end def

    def test_mab_run(self):

        million = 10 ** 6
        thousand = 10 ** 3

        mab = MultiArmedBandit(k=5, t=million)

        mab.run_bandit_algorithm()

        mab.analyse_regret()
        mab.analyse_suboptimal_arm_pulls()

        # mab.print_performance()
        mab.plot_suboptimal_arm()
        mab.plot_regret()

        mab.ph.show_plots()

        a = 5

