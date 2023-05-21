# This class contains the algorithm that we have come up with for our paper.
#
from NonStationaryAlgorithms.bandit_algorithm import BanditAlgorithm
from Helpers.misc_helper import MiscellaneousHelper as mh
from Helpers.math_helper import MathHelper
from Helpers.plot_helper import PlotHelper as ph
import numpy as np


def buffer(lambTil, delta):
    return lambTil/(3*delta)
    # return lambTil/delta


def conservative_buffer(logT, tau_i, delta):
    return (4/delta)*MathHelper.sqrt(logT/tau_i)


class OurAlgo(BanditAlgorithm):

    def __init__(self, arms, t, delta, verbose=False):
        super().__init__(arms, t, delta, verbose=verbose)
        self.lambTils = []
        self.g_i = []
        self.tau_i = []
        self.t_i = []
        self.episode_count = 0
        self.logT = MathHelper.natural_logarithm(self.T)

        self.epis_muhat_0 = []
        self.epis_muhat_1 = []
        self.epis_cnt_0 = 0
        self.epis_cnt_1 = 0

    def play_arms(self):

        # Episode
        i = 0
        self.t_i.append(-1)
        self.print_verbose("Episode " + str(i) + " starts after time " + str(self.t_i[i]))

        # ACTIVE ARMS
        A = [0, 1]
        # SNOOZED ARMS
        S = None

        for t in range(self.T):
            least_recently_played_arm = A.pop(0)
            A.append(least_recently_played_arm)

            # PULL ARMS ALTERNATIVELY.
            reward = super().pull_arm(least_recently_played_arm, t)
            self.rewards[t] = reward

            self.record_pull_vect(least_recently_played_arm, t, reward)

            # Monitor empirical means.
            self.update_empirical_means(least_recently_played_arm, reward)

            better_arm = -1
            if len(A) > 1 and (t-self.t_i[i]) % 2 == 0:  # If we are in active phase, and even count
                # PERFORM STATISTICAL TEST
                max_window = int((t-self.t_i[i])/2)
                # better_arm, lambTil, w = self.find_better_arm(t, max_window)
                better_arm, lambTil, w = self.find_better_arm_vect(t, self.t_i[i])

            if better_arm != -1:
                assert len(self.g_i) == i
                self.g_i.append(t)
                # For illustration - revert to lambTil
                self.lambTils.append(lambTil)
                self.tau_i.append(self.g_i[i]-self.t_i[i])

                self.print_verbose("Episode " + str(i) + " gauges at time "
                                   + str(self.g_i[i]) + ". Better arm " + str(better_arm))

                self.print_verbose("Window " + str(w) + " for a emp gap of " + str(lambTil))

                self.print_verbose("Gauge period " + str(self.tau_i[i]))
                # buf = buffer(lambTil, self.delta)
                logT = MathHelper.natural_logarithm(self.T)
                # buf = conservative_buffer(logT, self.tau_i[i], self.delta)
                buf = buffer(lambTil, self.delta)
                self.print_verbose("Buffer period " + str(buf))

                # if buf > self.tau_i[i]:
                if buf > 2*w:
                    # SNOOZE SUB-OPTIMAL ARM
                    subopt_arm = 1 - better_arm
                    snooze_end = self.g_i[i] - 2*w + buf
                    # snooze_end = self.t_i[i] + buf

                    self.print_verbose("Arm " + str(subopt_arm) + " snoozed till " + str(snooze_end))
                    S = subopt_arm, snooze_end
                    A.remove(subopt_arm)
                else:
                    # NO PASSIVE PHASE
                    # MOVE TO NEXT EPISODE
                    self.reset_counts()
                    i = i+1
                    assert len(self.t_i) == i
                    self.t_i.append(t)
                    self.print_verbose("Episode " + str(i) + " starts after time " + str(self.t_i[i]))

            if S is not None and t >= S[1]:
                # RESPAWN ARM
                arm_to_respawn = S[0]
                A.append(arm_to_respawn)
                S = None
                self.print_verbose("Arm " + str(arm_to_respawn) + " respawned at " + str(t))
                self.reset_counts()
                # MOVE TO NEXT EPISODE
                i = i+1
                assert len(self.t_i) == i
                self.t_i.append(t)
                self.print_verbose("Episode " + str(i) + " starts after time " + str(self.t_i[i]))
            #end if
        #end for

        i = i + 1
        assert len(self.t_i) == i
        self.t_i.append(self.T)
        self.episode_count = i
        # if len(self.tau_i) == i-1:
        #     self.tau_i.append(self.g_i[i] - self.t_i[i])

        self.algo_ran = True
    #end play_arms

    def find_better_arm(self, t, max_window):
        musum_0w = 0
        musum_1w = 0
        tt = t

        for w in range(1, max_window):
            # Empirical means of arms 0 and 1, in the last w samples.
            reward_tt = self.rewards[tt]    #t'
            reward_ttt = self.rewards[tt-1] #t''
            if self.arm_pulled_at_time[tt] == 0:
                musum_0w = musum_0w + reward_tt
                musum_1w = musum_1w + reward_ttt
            else:
                musum_1w = musum_1w + reward_tt
                musum_0w = musum_0w + reward_ttt

            muhat_0 = musum_0w/w
            muhat_1 = musum_1w/w

            radius = MathHelper.sqrt(2 * self.logT / w)
            lambTil = MathHelper.sqrt(72*self.logT / w)

            ucb_0 = muhat_0 + radius
            lcb_0 = muhat_0 - radius
            ucb_1 = muhat_1 + radius
            lcb_1 = muhat_1 - radius

            if lcb_1 - ucb_0 > 2*radius:
                return 1, lambTil, w
            elif lcb_0 - ucb_1 > 2*radius:
                return 0, lambTil, w

            tt = tt-2
        #end for

        return -1, None, None
    #end find_better_arm()

    def find_better_arm_vect(self, t, eps_start):
        #Rewards from current episode.
        e_reward_0 = self.reward_0[eps_start+1:t+1]
        e_reward_1 = self.reward_1[eps_start+1:t+1]

        # Sum from here (index) so far (to t). "sfhsf"
        sfhsf_0 = np.cumsum(e_reward_0[::-1])[::-1]
        sfhsf_1 = np.cumsum(e_reward_1[::-1])[::-1]
        # Delt = sfhsf_0 - sfhsf

        # Active period is guaranteed to be even from caller method.
        act = t-eps_start
        iw = np.asarray(range(act))
        wind_size = act - iw

        # Using wind_size/2 at that's the actual sample count. -> FLAG 1

        # Mean from here (index) so far (to t). "mfhsf"
        mfhsf_0 = 2*sfhsf_0 / wind_size     # FLAG 1
        mfhsf_1 = 2*sfhsf_1 / wind_size

        # For illustration - revert to 72 * self.logT / wind_size
        lam_to_detect = np.sqrt(72 * self.logT / wind_size)    # FLAG 1
        # radius = MathHelper.sqrt(self.logT / wind_size)
        radius = MathHelper.sqrt(4*self.logT / wind_size)
        # For illustration - revert to 4 * self.logT / wind_size

        ucb_0 = mfhsf_0 + radius
        lcb_0 = mfhsf_0 - radius
        ucb_1 = mfhsf_1 + radius
        lcb_1 = mfhsf_1 - radius

        is_1_better = lcb_1 - ucb_0 > 2 * radius
        is_0_better = lcb_0 - ucb_1 > 2 * radius

        if (is_0_better==False).all() and (is_1_better==False).all():
            return -1, None, None

        if (is_0_better==False).all():
            idx_0_better = -1
        else:
            idx_0_better = np.max(np.where(is_0_better == True))

        if (is_1_better==False).all():
            idx_1_better = -1
        else:
            idx_1_better = np.max(np.where(is_1_better == True))

        if idx_0_better > idx_1_better:
            lambTil = lam_to_detect[idx_0_better]
            w = wind_size[idx_0_better]     # Window size, not sample count.
            return 0, lambTil, w/2
        else:
            lambTil = lam_to_detect[idx_1_better]
            w = wind_size[idx_1_better]  # Window size, not sample count.
            return 1, lambTil, w/2
    #end find_better_arm()

    def print_results(self):
        print("Total episodes = ", self.episode_count)
        print("Start times = ", self.t_i)
        print("Gauge times = ", self.g_i)
        print("Gauge durations = ", self.tau_i)
        print("Empirical gaps = ", self.lambTils)

    def plot_episodic(self, plot_obj):
        # Minimum episode size, and maximum average regret.
        logT = MathHelper.natural_logarithm(self.T)
        min_episode_size = (4 ** (2 / 3)) * (self.delta ** (-2 / 3)) * (logT ** (1 / 3))
        r_ave = 3 * (4 ** (2 / 3)) * (self.delta ** (1 / 3)) * (logT ** (1 / 3))

        # inst_details = "\nMin episode size = " + mh.stringify(min_episode_size) + \
        #                ", Max average regret = " + mh.stringify(r_ave)

        # cur_title = plot_obj.get_title()
        # new_title = cur_title + inst_details
        # plot_obj.set_title(new_title)

        # Empirical rewards (within current episode)
        plot_obj.add_scatter(self.epis_muhat_0, "Arm 1 - Episodic emp. reward mean", 5)
        plot_obj.add_scatter(self.epis_muhat_1, "Arm 2 - Episodic emp. reward mean", 6)

        # Episodic starts, gauges.
        for i in range(self.episode_count):
            # For every episode,
            # The start - Blue solid lines
            # plot_obj.add_vline(self.t_i[i], None, "Episode " + str(i), 0.1, 0, linestyle=0)
            plot_obj.add_vline(self.t_i[i], None, r'$t_{0}$ '.format('{'+str(i+1)+'}'), 0.9, 0, linestyle=0)

            # The gauge, the lambtil - Magenta dotted lines
            if len(self.g_i) > i:
                # plot_obj.add_vline(self.g_i[i], None, "Gauge " + str(i) + ", Lambtil " + mh.stringify(self.lambTils[i]), 0.5, 4, linestyle=1)
                plot_obj.add_vline(self.g_i[i], None, r'$g_{0}$ '.format('{'+str(i+1)+'}') +
                                   r", $\widehat{\lambda}=$ " + mh.stringify(self.lambTils[i]), 0.1, 4, linestyle=1)

    def update_empirical_means(self, arm, reward):

        if arm == 0:
            if self.epis_cnt_0 == 0:
                self.epis_muhat_0.append(reward)
            else:
                muhat_0t = (self.epis_muhat_0[-1] * self.epis_cnt_0 + reward)/(self.epis_cnt_0 + 1)
                self.epis_muhat_0.append(muhat_0t)

            self.epis_cnt_0 = self.epis_cnt_0+1

            if self.epis_cnt_1 == 0:
                self.epis_muhat_1.append(0)
            else:
                self.epis_muhat_1.append(self.epis_muhat_1[-1])
        elif arm == 1:
            if self.epis_cnt_1 == 0:
                self.epis_muhat_1.append(reward)
            else:
                muhat_1t = (self.epis_muhat_1[-1] * self.epis_cnt_1 + reward)/(self.epis_cnt_1 + 1)
                self.epis_muhat_1.append(muhat_1t)

            self.epis_cnt_1 = self.epis_cnt_1+1

            if self.epis_cnt_0 == 0:
                self.epis_muhat_0.append(0)
            else:
                self.epis_muhat_0.append(self.epis_muhat_0[-1])

    def reset_counts(self):
        self.epis_cnt_0 = 0
        self.epis_cnt_1 = 0







