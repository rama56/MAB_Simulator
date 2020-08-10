from Helpers.math_helper import MathHelper as rvh


class Arm:
    _mean = 0
    _tape = []
    _tape_index = 0
    # Ideally, move the data stuff outside here and put into ucb.py
    pull_count = 0
    reward_sum = 0
    empirical_mean = 0

    def __init__(self, mean, size=10 ** 7):
        self._mean = mean

        # Create a tape of values to return.
        self._tape = rvh.get_bernoulli_sample(p=self._mean, size=size)
        self._tape_index = 0

    def reset_counts(self):
        self.pull_count = 0
        self.reward_sum = 0
        self.empirical_mean = 0
        self._tape_index = 0

    def pull(self):
        reward_sample = self._tape[self._tape_index]

        self._tape_index = self._tape_index + 1
        self.pull_count = self.pull_count + 1
        self.reward_sum = self.reward_sum + reward_sample
        self.empirical_mean = self.reward_sum / self.pull_count

        return reward_sample
