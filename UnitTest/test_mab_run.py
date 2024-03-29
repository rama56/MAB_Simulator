
from unittest import TestCase

from UrgentBandits.ucb1 import UCB1
from UrgentBandits.ucb_doubling import UCBDoubling
from UrgentBandits.ucb_incremental import UCBIncremental
from BanditElements.arm import Arm
from BanditInstance.multi_armed_bandit import MultiArmedBandit


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
    # end def

    def test_mab_run(self):
        mab = MultiArmedBandit(k=2, t=10 ** 4)

        algorithms_to_run = [("UCB1", UCB1), ("UCB-Inc", UCBIncremental),
                             # ("UCB-Doub-TR", UCBDoubling, mh.ucb_doubling_radius),
                             ("UCB-Doub", UCBDoubling)]

        mab.set_algorithms(algorithms_to_run)

        mab.run_bandit_algorithm()

        mab.analyse_regret()
        mab.analyse_suboptimal_arm_pulls()
        # mab.print_performance()
        mab.plot_suboptimal_arm()
        mab.plot_regret()

        mab.ph.show_plots()

    # end def

    def test_mab_run_many_arms(self):

        mab = MultiArmedBandit(k=100, t=10**6)

        mab.run_bandit_algorithm()

        # With many arms, we don't analyse pulls of sub-optimal arms.
        mab.analyse_regret()
        mab.print_performance()
        mab.plot_regret()

        mab.ph.show_plots()

    # end def

