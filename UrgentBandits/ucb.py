import abc  # Abstract Base Class
from Helpers.misc_helper import MiscellaneousHelper as mh


class UCB(abc.ABC):
    arms = 0
    T = 0   # Horizon limit of time.
    N = 0   # Number of arm pulls to be made.
    K = 0   # Number of arms

    upper_confidence_bound = None
    arm_pulls_by_time = None
    radius_function = None

    def __init__(self, arms, t, n, radius_function=mh.textbook_radius):
        self.arms = arms
        self.T = t
        self.N = n

        self.K = len(self.arms)
        self.upper_confidence_bound = [0] * self.K

        self.arm_pulls_by_time = [[0] * self.K]

        self.radius_function = radius_function

        a = 5

    @abc.abstractmethod
    def play_arms(self):
        pass

    @abc.abstractmethod
    def revise_ucbs(self, t):
        pass

    def record_pull(self, zeros):
        assert len(zeros) == self.K
        self.arm_pulls_by_time.append(zeros)

    def pull_arm(self, arm_number):
        # Get the reward from the arm.
        reward = self.arms[arm_number].pull()

        # Record the pull.
        zeros = [0] * self.K    # TODO: Rename zeros
        zeros[arm_number] = 1
        self.record_pull(zeros)

        # Return the reward
        return reward






