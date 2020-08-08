
from unittest import TestCase
from math_helper import MathHelper as rvh


class Test(TestCase):

    def test_library_random_variables(self):

        # Uniform distribution
        result_1 = rvh.get_uniform_sample(0, 1, 10)

        # Bernoulli distribution
        result_2 = rvh.get_bernoulli_sample(0.8)

        result_3 = rvh.get_bernoulli_sample(p=0.5, size=10)

        a = 5

