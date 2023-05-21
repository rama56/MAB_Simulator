import math
from unittest import TestCase
import datetime
import time

from Helpers.misc_helper import MiscellaneousHelper as mh
# from BanditElements.arm import Arm
from BanditElements.slowly_varying_arm import SlowlyVaryingArm
from BanditInstance.multi_armed_bandit import MultiArmedBandit
from BanditInstance.slowly_varying_mab import SlowlyVaryingBandits
from Helpers.plot_helper import PlotHelper
from NonStationaryAlgorithms import our_algo, slidwind_ucb_hash, rexp3, exp3s
import Helpers.file_helper as fh
from Helpers.non_stationary_helper import NonStationaryHelper as nsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Test(TestCase):

    def test_slowly_varying_arm(self):
        starting_mean = 0.3
        delta = 0.01
        size = 1000

        arm = SlowlyVaryingArm(starting_mean, delta, size)
        rewards = []
        T = 20

        for t in range(1, T+1):
            reward = arm.pull()
            rewards.append(reward)

        print("True means = ", arm.means[:T])
        print("Empirical rewards = ", rewards)

        assert arm.pull_count == 20
    # end def

    def test_mab_run(self):
        starting_means = [0.2, 0.5]

        T = math.ceil(math.e ** 12)
        delta = 1/(432*15*10)

        arm1 = SlowlyVaryingArm(starting_means[0], delta, T)
        arm1.make_move_reverse_arm(0.01)

        arm2 = SlowlyVaryingArm(starting_means[1], delta, T)
        arm2.make_move_reverse_arm(0.00001)

        svb = SlowlyVaryingBandits(k=2, t=T, delta=delta, start_means=starting_means, arms=[arm1, arm2])

        # algorithms_to_run = [("UCB1", UCB1), ("UCB-Inc", UCBIncremental),
        #                      ("UCB-Doub-TR", UCBDoubling, mh.ucb_doubling_radius), ("UCB-Doub", UCBDoubling)]

        algorithms_to_run = [("Our algorithm", our_algo.OurAlgo)]
        svb.set_algorithms(algorithms_to_run)

        svb.run_bandit_algorithm(verbose=True)
        svb.fill_instance_specific_info()

        svb.plot_our_algo_trajectory()

        # svb.analyse_regret()

        # svb.plot_regret()

        svb.show_plots()
    # end def

    def test_create_problem_instances(self):
        fw = fh.FileHelper()
        file_name = "instance_9.json"
        starting_means = [0.5, 0.5]
        # rev_probs = [0.0001, 0.01]
        logT = 12
        T = math.ceil(math.e ** logT)
        const = 432
        delta = 1 / (const * logT)

        # Instance 7.
        # cp1 = [(T / 4, "up"), (3 * T / 4, "flat"), (T, "down")]
        # cp2 = [(T / 3, "flat"), (4 * T / 5, "down"), (T, "up")]

        # Instance 8
        cp1 = [(0.33*T, "flat"), (0.5*T, "up"), (0.75*T, "down"), (T, "flat")]
        cp2 = [(0.33*T, "flat"), (0.6*T, "down"), (0.8*T, "up"), (T, "flat")]
        checkpoints = [cp1, cp2]

        print("File = " + file_name +
              " Mu start = " + str(starting_means) +
              " logT = " + str(logT) +
              # " Rev probs = " + str(rev_probs) +
              " Checkpoints = " + str(checkpoints) +
              " delta = " + str(delta))

        arm1 = SlowlyVaryingArm(starting_means[0], delta, T)
        # arm1.make_move_reverse_arm(rev_probs[0])
        arm1.make_arms_turn_checkpoints(cp1)

        arm2 = SlowlyVaryingArm(starting_means[1], delta, T)
        # arm2.make_move_reverse_arm(rev_probs[1])
        arm2.make_arms_turn_checkpoints(cp2)

        mu_1 = np.asarray(arm1.means)
        mu_2 = np.asarray(arm2.means)

        Delta = nsh.get_Delta_from_mu(mu_1, mu_2)
        Lambdas = None
        # nsh.get_lambda_from_mu_vectorized(mu_1, mu_2)

        instance = starting_means, checkpoints, logT, delta, arm1, arm2, Delta, Lambdas

        fw.write_json_to_file(instance, file_name)

    def test_create_deterministic_problem_instances(self):
        fw = fh.FileHelper()
        file_name = "instance_17.json"
        # starting_means = [0.9, 0.2]
        logT = 12
        T = math.ceil(math.e ** logT)
        const = 400

        # Instance 11
        # delta = 1 / (const * logT * 10)
        # cp1 = [(0.5 * T, "up3"), (0.7 * T, "flat"), (T, "down3")]
        # cp2 = [(0.5 * T, "down3"), (0.7 * T, "flat"), (T, "up3")]

        # # Instance 12
        # delta = 1 / (const * logT)
        # cp1 = [(0.1 * T, "flat"), (0.2 * T, "flat"), (0.3 * T, "up3"), (0.4 * T, "flat"), (0.5 * T, "down3"),
        #        (0.6 * T, "flat"), (0.7 * T, "up3"), (0.8 * T, "flat"), (0.9 * T, "down3"), (T, "flat")]
        #
        # cp2 = [(0.1 * T, "flat"), (0.2 * T, "down3"), (0.3 * T, "flat"), (0.4 * T, "flat"), (0.5 * T, "up3"),
        #        (0.6 * T, "flat"), (0.7 * T, "flat"), (0.8 * T, "flat"), (0.9 * T, "down3"), (T, "flat")]

        # # Instance 13
        delta = 1 / (const * logT * 10)
        starting_means = [0.9, 0.2]
        cp1 = [(0.1 * T, "flat"), (0.2 * T, "down3"), (0.3 * T, "flat"), (0.42 * T, "up3"), (0.5 * T, "up3"),
               (0.6 * T, "flat"), (0.7 * T, "flat"), (0.8 * T, "flat"), (0.9 * T, "down3"), (T, "up3")]

        cp2 = [(0.1 * T, "flat"), (0.2 * T, "flat"), (0.3 * T, "flat"), (0.4 * T, "flat"), (0.5 * T, "flat"),
               (0.6 * T, "flat"), (0.7 * T, "flat"), (0.8 * T, "flat"), (0.9 * T, "flat"), (T, "flat")]

        # In 13, to not hit mu=1 (for arm 1), consider changing
        # (0.4 * T, "up1"), (0.5 * T, "up1") to (0.42 * T, "up1"), (0.5 * T, "flat")

        # Instance 14
        # delta = 1 / (const * logT * 2)
        # starting_means = [0.7, 0.3]
        # cp1 = [(0.1 * T, "flat"), (0.15 * T, "down3"), (0.3 * T, "flat"), (0.33 * T, "up3"), (0.45 * T, "flat"), (0.48 * T, "down3"),
        #        (0.6 * T, "flat"), (0.7 * T, "up1"), (0.8 * T, "flat"), (0.82 * T, "down3"), (T, "flat")]
        #
        # cp2 = [(0.05 * T, "flat"), (0.08 * T, "up3"), (0.2 * T, "flat"), (0.25 * T, "down3"), (0.35 * T, "flat"), (0.39 * T, "up3"),
        #        (0.5 * T, "flat"), (0.51 * T, "up3"), (0.6 * T, "flat"),  (0.7 * T, "down1"), (0.9 * T, "flat"), (T, "flat")]

        # checkpoints = [cp1, cp2]
        checkpoints = None

        # # Instance 15
        # delta = 1 / (const * logT)
        # starting_means = [0.9, 0.1]
        # flat_duration = 3500
        # drift_duration = 3500

        print("File = " + file_name +
              " Mu start = " + str(starting_means) +
              " logT = " + str(logT) +
              # " Flat duration = " + str(flat_duration) + ", Drift duration = " + str(drift_duration) +
              " Det. Checkpoints = " + str(checkpoints) +
              " delta = " + str(delta))

        arm1 = SlowlyVaryingArm(starting_means[0], delta, T)
        arm1.make_arms_turn_det_checkpoints(cp1)
        # arm1.make_arms_flip_rapidly(-1, flat_duration, drift_duration)

        arm2 = SlowlyVaryingArm(starting_means[1], delta, T)
        # arm2.make_move_reverse_arm(rev_probs[1])
        arm2.make_arms_turn_det_checkpoints(cp2)
        # arm2.make_arms_flip_rapidly(1, flat_duration, drift_duration)
        mu_1 = np.asarray(arm1.means)
        mu_2 = np.asarray(arm2.means)

        # Delta = nsh.get_Delta_from_mu(mu_1, mu_2)
        Delta = None
        Lambdas = None
        # nsh.get_lambda_from_mu_vectorized(mu_1, mu_2)

        instance = starting_means, checkpoints, logT, delta, arm1, arm2, Delta, Lambdas

        fw.write_json_to_file(instance, file_name)

    def test_display_problem_instance_from_file(self):
        file_name = "oscillation1.json"
        fw = fh.FileHelper()

        start_time = time.time()
        data = fw.read_json_from_file(file_name)
        # arm1, arm2 = data   # For instance_1,2,3.
        starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = data
        end_time = time.time()
        print('Read instance from file-time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

        # logT = 12
        T = math.ceil(math.e ** logT)
        # const = 432
        # delta = 1 / (const * logT)

        svb = SlowlyVaryingBandits\
            (k=2, t=T, delta=delta, start_means=None,
             arms=[arm1, arm2], Deltas=None, Lambdas=None)

        # svb.fill_Lambdas_vectorized()
        svb.plot_instance()

        svb.show_plots()

    def test_run_problem_once_from_file(self):
        file_name = "oscillation1.json"
        fw = fh.FileHelper()

        start_time = time.time()
        data = fw.read_json_from_file(file_name)
        # For more recent file instances.
        starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = data
        # For older file instances.
        # arm1, arm2 = data   # For instance_1,2,3.

        end_time = time.time()
        print('Read instance from file-time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
        # logT = 10
        T = math.ceil(math.e ** logT)
        # const = 432
        # delta = 1 / (const * logT)

        arm1.refill_tape()
        arm2.refill_tape()

        svb = SlowlyVaryingBandits\
            (k=2, t=T, delta=delta, start_means=None,
             arms=[arm1, arm2], Deltas=None, Lambdas=None)

        # algorithms_to_run = [("Our algorithm", our_algo.OurAlgo)]
        algorithms_to_run = [
            ("SnoozeIt", our_algo.OurAlgo),
            # ("SW-UCB\#", slidwind_ucb_hash.SlidWind_UCB_Hash),
            # ("Rexp3", rexp3.RExp3),
            # ("Exp3.S", exp3s.Exp3s)
            ]

        svb.set_algorithms(algorithms_to_run)

        start_time = time.time()
        svb.run_bandit_algorithm(verbose=True)
        end_time = time.time()
        print('Vectorized exec-time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

        svb.fill_instance_specific_info()
        svb.analyse_regret()
        svb.plot_instance()
        # svb.plot_regret()
        # svb.plot_arm_pulls()

        svb.show_plots()

    def test_create_oscillating_arms(self):
        fw = fh.FileHelper()
        file_name = "oscillation8.json"
        logT = 12
        T = math.ceil(math.e ** logT)
        const = 400

        # Oscill 1, delta = 1/const logT 10
        # Oscill 2, delta = 1.33/const logT 10
        # Oscill 3, delta = 0.66/const logT 10
        # Oscill 4, delta = 0.33/const logT 10

        # Oscill 5, 1 / (const * logT * 10) (smallest drift)
        # cp1 = [(0.1 * T, "flat"), (0.2 * T, "up3"), (0.3 * T, "flat"), (0.42 * T, "down3"), (0.5 * T, "down3"),
        #        (0.6 * T, "flat"), (0.7 * T, "up3"), (0.8 * T, "up3"), (0.9 * T, "flat"), (T, "down3")]
        #
        # cp2 = [(0.1 * T, "flat"), (0.2 * T, "down3"), (0.3 * T, "flat"), (0.4 * T, "up3"), (0.5 * T, "up3"),
        #        (0.6 * T, "flat"), (0.7 * T, "down3"), (0.8 * T, "down3"), (0.9 * T, "flat"), (T, "up3")]

        # Oscill 6, 2 / (const * logT * 10) (quicker drift)
        # cp1 = [(0.1 * T, "flat"), (0.15 * T, "up3"), (0.35 * T, "flat"), (0.4 * T, "down3"), (0.45 * T, "down3"),
        #        (0.65 * T, "flat"), (0.7 * T, "up3"), (0.75 * T, "up3"), (0.95 * T, "flat"), (T, "down3")]
        #
        # cp2 = [(0.1 * T, "flat"), (0.15 * T, "down3"), (0.35 * T, "flat"), (0.4 * T, "up3"), (0.45 * T, "up3"),
        #        (0.65 * T, "flat"), (0.7 * T, "down3"), (0.75 * T, "down3"), (0.95 * T, "flat"), (T, "up3")]

        delta = 4 / (const * logT * 10)
        starting_means = [0.5, 0.5]

        lag = 0.1 * T / 4

        cp1 = [(0.1 * T, "flat"), (0.1 * T + lag, "up3"), (0.4 * T - lag, "flat"), (0.4 * T, "down3"), (0.4 * T + lag, "down3"),
               (0.7 * T - lag, "flat"), (0.7 * T, "up3"), (0.7 * T + lag, "up3"), (1 * T - lag, "flat"), (T, "down3")]

        cp2 = [(0.1 * T, "flat"), (0.1 * T + lag, "down3"), (0.4 * T - lag, "flat"), (0.4 * T, "up3"), (0.4 * T + lag, "up3"),
               (0.7 * T - lag, "flat"), (0.7 * T, "down3"), (0.7 * T + lag, "down3"), (1 * T - lag, "flat"), (T, "up3")]

        checkpoints = [cp1, cp2]
        # checkpoints = None

        print("File = " + file_name +
              " Mu start = " + str(starting_means) +
              " logT = " + str(logT) +
              # " Flat duration = " + str(flat_duration) + ", Drift duration = " + str(drift_duration) +
              " Det. Checkpoints = " + str(checkpoints) +
              " delta = " + str(delta))

        arm1 = SlowlyVaryingArm(starting_means[0], delta, T)
        arm1.make_arms_turn_det_checkpoints(cp1)

        arm2 = SlowlyVaryingArm(starting_means[1], delta, T)
        arm2.make_arms_turn_det_checkpoints(cp2)

        Delta = None
        Lambdas = None

        instance = starting_means, checkpoints, logT, delta, arm1, arm2, Delta, Lambdas

        fw.write_json_to_file(instance, file_name)

    def test_compute_lambda_and_write(self):
        # Instance - Lambda
        # 7 - 1
        # 4 - 2, 2a (some great snaps and idea), actually 2b

        input_file_name = "instance_4.json"
        fw = fh.FileHelper()

        start_time = time.time()
        data = fw.read_json_from_file(input_file_name)
        # arm1, arm2 = data # For instance_1,2,3.
        starting_means, rev_probs, logT, delta, arm1, arm2, Delta, Lambdas = data
        end_time = time.time()
        print('Read instance from file-time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

        T = math.ceil(math.e ** logT)

        svb = SlowlyVaryingBandits \
            (k=2, t=T, delta=delta, start_means=starting_means,
             arms=[arm1, arm2], Deltas=Delta, Lambdas=None)

        # svb.fill_Deltas()
        svb.fill_Lambdas_vectorized()

        instance = logT, delta, arm1, arm2, svb.Deltas, svb.Lambdas_vect
        output_file_name = "Lambda_2b.json"
        fw.write_json_to_file(instance, output_file_name)

        svb.plot_instance()
        svb.show_plots()

    def test_vectorized_lambda_computation(self):
        file_names = ["instance_5.json"]

        # This is the only test that builds plots by itself. Because we need sub-plots, and PlotHelper doesn't
        # support that as of now.

        fig = plt.figure(1, figsize=(9, 9))
        ax = None
        plot_id = 1
        for file_name in file_names:
            ax = fig.add_subplot(2, 2, plot_id)
            fw = fh.FileHelper()
            data = fw.read_json_from_file(file_name)
            starting_means, checkpoints, logT, delta, arm1, arm2, Delta, Lambdas = data

            T = math.ceil(math.e ** logT)
            const = 432
            delta = 1 / (const * logT * 10)

            svb = SlowlyVaryingBandits\
                (k=2, t=T, delta=delta, start_means=None,
                 arms=[arm1, arm2])

            svb.fill_Deltas()

            start_time = time.time()
            svb.fill_Lambdas_vectorized()
            end_time = time.time()
            print('Execution time Vectorized code: ' + str(datetime.timedelta(seconds=end_time - start_time)))

            mu_1 = svb.true_means[0]
            mu_2 = svb.true_means[1]
            xrange = len(mu_1)
            ax.scatter(range(xrange), mu_1, s=1, c=PlotHelper.colours[1], label="Arm 1")
            ax.scatter(range(xrange), mu_2, s=1, c=PlotHelper.colours[2], label="Arm 2")
            ax.scatter(range(xrange), svb.Deltas, s=1, c=PlotHelper.colours[3], label="Gap, " + r'$\Delta$')
            ax.scatter(range(xrange), svb.Lambdas_vect, s=1, c=PlotHelper.colours[4], label="Detectable Gap, " + r'$\lambda$')

            plot_id += 1

        # End for loop over instances.

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        # lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.show()

    def test_sub_plot_printing(self):
        file_name = ["Lambda_2b.json"]
        fw = fh.FileHelper()
        data = fw.read_json_from_file(file_name[0])
        logT, delta, arm1, arm2, Delta, Lambdas = data

        # This is the second test that builds plots by itself. Because we need sub-plots, and PlotHelper doesn't
        # support that as of now.

        limits = [[(None, None), (-0.05, 1.05)],
                  [(46000, 66000), (0.65, 0.783)],
                  [(56500, 60000), (0.733, 0.749)],
                  [(57801, 58049), (0.73976, 0.74124)]]

        fig = plt.figure(1, figsize=(9, 3))
        ax = None
        plot_id = 1
        for xlimit, ylimit in limits:
            ax = fig.add_subplot(1, 4, plot_id)
            ax.grid(True)
            xs, xe = xlimit
            ys, ye = ylimit
            if xs is not None:
                ax.set_xlim([xs, xe])
            if ys is not None:
                ax.set_ylim([ys, ye])

            ax.set_xlabel("Time, " + r'$t$')
            if plot_id == 1:
                ax.set_ylabel("Reward")

            T = math.ceil(math.e ** logT)
            # const = 432
            # delta = 1 / (const * logT * 10)

            # svb = SlowlyVaryingBandits\
            #     (k=2, t=T, delta=delta, start_means=None,
            #      arms=[arm1, arm2])

            # mu_1 = svb.true_means[0]
            # mu_2 = svb.true_means[1]
            mu_1 = arm1.means
            mu_2 = arm2.means
            xrange = len(mu_1)
            ax.scatter(range(xrange), mu_1, s=1, c=PlotHelper.colours[1], label="Arm 1, " + r'$\mu_{1,.}$')
            ax.scatter(range(xrange), mu_2, s=1, c=PlotHelper.colours[2], label="Arm 2, " + r'$\mu_{2,.}$')
            ax.scatter(range(xrange), Delta, s=1, c=PlotHelper.colours[3], label="Gap, " + r'$\Delta_{2,.}$')
            ax.scatter(range(xrange), Lambdas, s=1, c=PlotHelper.colours[7], label="Detectable Gap, " + r'$\lambda_{2,.}$')

            plot_id += 1

        # End for loop over xlimits, ylimits.

        handles, labels = ax.get_legend_handles_labels()
        patch1 = mpatches.Patch(color=PlotHelper.colours[1], label="Arm 1, " + r'$\mu_{1,.}$')
        patch2 = mpatches.Patch(color=PlotHelper.colours[2], label="Arm 2, " + r'$\mu_{2,.}$')
        patch3 = mpatches.Patch(color=PlotHelper.colours[3], label="Gap, " + r'$\Delta_{2,.}$')
        patch4 = mpatches.Patch(color=PlotHelper.colours[7], label="Detectable Gap, " + r'$\lambda_{2,.}$')
        handles = [patch1, patch2, patch3, patch4]

        fig.legend(handles, labels, loc='upper center', ncol=4)

        # lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.show()
