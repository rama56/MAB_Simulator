import numpy as np
import math

from arm import Arm
from Helpers.math_helper import MathHelper as rvh


class MiscellaneousHelper:

    @staticmethod
    def get_maximum_index(list_):
        a = np.array(list_)
        idx = np.argmax(a)
        return idx

    @staticmethod
    def get_arms(arm_count, tape_size):
        true_means = rvh.get_uniform_sample(0, 1, arm_count)
        arms = []
        for i in range(arm_count):
            arm = Arm(true_means[i], size=tape_size)
            arms.append(arm)

        return true_means, arms

    @staticmethod
    def ucb_doubling_radius(n, pull_count):
        radius = math.sqrt((2 * math.log(math.log(n))) / pull_count)
        return radius

    @staticmethod
    def textbook_radius(t, pull_count):
        radius = math.sqrt((2 * math.log(t)) / pull_count)
        return radius

    ''' Compute the summation of deltas of all arms, and summation of delta inverse of all non-optimal arms. '''
    @staticmethod
    def get_instance_dependent_values(best_arm, deltas):
        sum_del_inv = 0
        sum_del = 0

        k = len(deltas)

        for i in range(k):
            if i == best_arm:
                continue

            del_inv = 1 / deltas[i]
            sum_del_inv = sum_del_inv + del_inv
            sum_del = sum_del + deltas[i]

        return sum_del_inv, sum_del

    @staticmethod
    def ciel_root(n):
        x = math.sqrt(n)
        y = math.ceil(x)
        return y

    @staticmethod
    def get_theoretical_constants(sum_del_inv, sum_del):
        mult_constant = 2 * sum_del_inv
        addi_constant = rvh.func_of_pi(add=1, power=2, mult=1 / 3) * sum_del

        return mult_constant, addi_constant

    @staticmethod
    def get_instance_dependent_square_inverses(deltas, best_arm):
        k = len(deltas)
        del_sq_invs = [0] * k

        for i in range(k):
            if i == best_arm:
                del_sq_invs[i] = -1
            else:
                del_sq_inv = 1 / pow(deltas[i], 2)
                del_sq_invs[i] = del_sq_inv

        return 8 * del_sq_invs

    @staticmethod
    def stringify(fl):
        return str(round(fl, 2))

    @staticmethod
    def stringify_list(fl_list):
        readable = ""
        for fl in fl_list:
            readable = readable + "  " + MiscellaneousHelper.stringify(fl)

        return readable
