import numpy as np
import math

from math_helper import MathHelper as rvh

class MiscellaneousHelper:

    @staticmethod
    def get_maximum_index(list_):
        a = np.array(list_)
        idx = np.argmax(a)
        return idx


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

        return del_sq_invs

    @staticmethod
    def stringify(fl):
        return str(round(fl, 2))
