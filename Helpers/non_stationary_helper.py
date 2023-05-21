from BanditElements.arm import Arm
from Helpers.math_helper import MathHelper as rvh
from BanditElements.slowly_varying_arm import SlowlyVaryingArm
import numpy as np


class NonStationaryHelper:

    @staticmethod
    def get_slowly_varying_arms(arm_count, tape_size, delta, start_means):
        true_starting_means = start_means
        if start_means is None:
            true_starting_means = rvh.get_uniform_sample(0, 1, arm_count)
        arms = []
        for i in range(arm_count):
            # Creates an arm and the varying reward profile
            arm = SlowlyVaryingArm(true_starting_means[i], delta, size=tape_size)
            arm.make_normal_arm()
            # arm.fill_tape()

            arms.append(arm)

        true_means = [x.means for x in arms]
        return true_means, arms

    @staticmethod
    def get_lambda_from_mu(means1, means2):
        T = len(means1)
        logT = rvh.natural_logarithm(T)

        lambdas = [0]*T

        for t in range(T):
            cum_dif = means1[t] - means2[t]
            for w in range(2, t):
                cum_dif = cum_dif + (means1[t-w+1] - means2[t-w+1])

                lamb = 12* rvh.sqrt(logT/w)

                if cum_dif > w*lamb or cum_dif < -w*lamb:
                    lambdas[t] = lamb
                    break
                # end if
            # end for
        # end for

        return lambdas

    @staticmethod
    def get_lambda_from_mu_vectorized(mu_1, mu_2):
        T = len(mu_1)
        logT = rvh.natural_logarithm(T)
        Del = mu_1 - mu_2
        #lambdas = np.zeros(T)
        lambdas = []

        # Sum from here so far. (sfhsf) (From index of list to t)
        sfhsf = np.zeros(T)

        for t in range(T):
            Del_t = mu_1[t] - mu_2[t]
            sfhsf[:(t+1)] += Del_t

            iw = np.asarray(range(t+1))
            wind_size = t+1 - iw

            # Mean from here so far (mfhsf) (From index of list to t)
            mfhsf = sfhsf[:(t+1)] / wind_size

            # \lamba = sqrt(logT/ windowsize)
            lam_to_detect = 12 * np.sqrt(logT/wind_size)

            is_lam_detectable = np.bitwise_or(mfhsf > lam_to_detect, -mfhsf > lam_to_detect)

            if(is_lam_detectable == False).all():
                lambdas.append(0)
            else:
                reqd_idx = np.max(np.where(is_lam_detectable == True))
                lambdas.append(lam_to_detect[reqd_idx])
        # end for

        return lambdas

    @staticmethod
    def get_Delta_from_mu(mu_1, mu_2):
        return np.absolute(mu_1 - mu_2)