from scipy.stats import bernoulli
import numpy as np
import math


class MathHelper:

    @staticmethod
    def get_uniform_sample(lower, upper, size=None):
        return np.random.uniform(lower, upper, size)

    @staticmethod
    def get_bernoulli_sample(p, size=None):
        # Returns 0 with probability 1-p,
        #         1 with probability p

        return bernoulli.rvs(p=p, size=size, )

    @staticmethod
    def get_gaussian_sample(mean, variance):
        st_d = variance**(1/2)
        return np.random.normal(loc=mean, scale=st_d)

    ''' Deliberate choice to not overload (instead, redefine) as it 
    adds instruction cycles to a repeatedly called method'''
    @staticmethod
    def get_bernoulli_samples(p, size):
        # Returns 0 with probability 1-p,
        #         1 with probability p

        return bernoulli.rvs(p=p, size=size)

    @staticmethod
    def something():
        return 1

    @staticmethod
    def func_of_pi(add, power, mult):
        return add + pow(np.pi, power) * mult

    @staticmethod
    def natural_logarithm(x):
        # return math.log(x)
        # if x < 1:
        #     return 0
        return np.log(x)

    @staticmethod
    def custom_log(x, base):
        return math.log(x, base)

    @staticmethod
    def ceiled_log_base_2(x):
        a = np.log2(x)
        return math.ceil(a)
        pass

    @staticmethod
    def sqrt(x):
        a = np.sqrt(x)
        return a

    @staticmethod
    def absolute(x):
        if x > 0:
            return x
        else:
            return -x

    @staticmethod
    def ceil(x):
        return math.ceil(x)

    @staticmethod
    def isinf(x):
        return math.isinf(x)


