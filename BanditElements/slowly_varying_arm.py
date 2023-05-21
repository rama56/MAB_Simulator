import enum

from Helpers.math_helper import MathHelper as rvh
from BanditElements import arm


def probably_reverse(prob):
    rand_val = rvh.get_uniform_sample(0,1)
    if rand_val < prob:
        return -1
    return 1


class SlowlyVaryingArm(arm.Arm):
    _means = []     # Ditch the _mean of the super class.
    _tape_size = 0
    _delta = 0

    drift_values = {
        "up3": 1,
        "up2": 0.66,
        "up1": 0.33,
        "flat": 0,
        "down1": -0.33,
        "down2": -0.66,
        "down3": -1
    }

    def __init__(self, starting_mean, delta, size=10 ** 7):
        self._means = [starting_mean]
        self._tape_size = size
        self._delta = delta

    def make_normal_arm(self):
        for t in range(1, self._tape_size):
            mu_i_t = self._means[t - 1] + rvh.get_uniform_sample(-self._delta, self._delta)
            if mu_i_t < 0:
                mu_i_t = -mu_i_t
            if mu_i_t > 1:
                mu_i_t = 2 - mu_i_t

            self._means.append(mu_i_t)

        # Generate stochastic arm pull outputs.
        self.fill_tape()

    def make_move_reverse_arm(self, reverse_chance=0.1):
        sign = 1
        for t in range(1, self._tape_size):
            mu_i_t = self._means[t - 1] + rvh.get_uniform_sample(0, self._delta) * sign
            if mu_i_t < 0:
                mu_i_t = -mu_i_t
            if mu_i_t > 1:
                mu_i_t = 2 - mu_i_t

            self._means.append(mu_i_t)
            sign = probably_reverse(reverse_chance) * sign

        # Generate stochastic arm pull outputs.
        self.fill_tape()

    def make_arms_turn_checkpoints(self, checkpoints):
        i=0
        limit, behaviour = checkpoints[i]
        for t in range(1, self._tape_size):
            mu_i_t = self._means[t - 1] + self.get_drift_from_behaviour(behaviour)
            if mu_i_t < 0:
                mu_i_t = -mu_i_t
            if mu_i_t > 1:
                mu_i_t = 2 - mu_i_t

            self._means.append(mu_i_t)

            if t >= limit:
                i = i+1
                limit, behaviour = checkpoints[i]

        # Generate stochastic arm pull outputs.
        self.fill_tape()

    def make_arms_turn_det_checkpoints(self, checkpoints):
        i = 0
        limit, behaviour = checkpoints[i]
        for t in range(1, self._tape_size):
            mu_i_t = self._means[t - 1] + self._delta*self.drift_values[behaviour]
            if mu_i_t < 0:
                mu_i_t = -mu_i_t
            if mu_i_t > 1:
                mu_i_t = 2 - mu_i_t

            self._means.append(mu_i_t)

            if t >= limit:
                i = i + 1
                limit, behaviour = checkpoints[i]

        # Generate stochastic arm pull outputs.
        self.fill_tape()

    def make_arms_flip_rapidly(self, start_sign, flat_duration, drift_duration):
        f_d = flat_duration
        d_d = drift_duration
        for t in range(1, self._tape_size):
            mu_i_t = None
            if f_d > 0:
                mu_i_t = self._means[t - 1]
                f_d -= 1
            elif d_d > 0:
                mu_i_t = self.means[t-1] + start_sign * self._delta
                d_d -= 1
            else:
                mu_i_t = self._means[t-1]

            if f_d == 0 and d_d == 0:
                f_d = flat_duration
                d_d = drift_duration
                start_sign = - start_sign

            self.means.append(mu_i_t)

        # Generate stochastic arm pull outputs.
        self.fill_tape()

    def fill_tape(self):
        self._tape = []
        for t in range(self._tape_size):
            # muhat_t = rvh.get_bernoulli_sample(self._means[t])
            muhat_t = rvh.get_gaussian_sample(mean=self._means[t], variance=1/4)
            self._tape.append(muhat_t)

        self._tape_index = 0

    def refill_tape(self):
        self.fill_tape()

    def get_drift_from_behaviour(self, beh):
        if beh == "up":
            s = rvh.get_uniform_sample(0, self._delta)
        elif beh == "flat":
            s = rvh.get_uniform_sample(-self._delta, self._delta)
        else:
            s = -rvh.get_uniform_sample(0, self._delta)
        return s

    @property
    def means(self):
        return self._means




