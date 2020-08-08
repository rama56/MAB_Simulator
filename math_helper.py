from scipy.stats import bernoulli
import numpy as np


class MathHelper:

    @staticmethod
    def get_uniform_sample(lower, upper, size=1):
        return np.random.uniform(lower, upper, size)

    @staticmethod
    def get_bernoulli_sample(p, size=1):
        # Returns 0 with probability 1-p,
        #         1 with probability p

        return bernoulli.rvs(p=p, size=size)

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
