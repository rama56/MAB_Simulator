
from unittest import TestCase

from Helpers.misc_helper import MiscellaneousHelper as mh

from Helpers.log_helper import LogHelper


class Test(TestCase):

    def test_get_arms(self):
        true_means, arms = mh.get_arms(10, 100)

        best_arm = mh.get_maximum_index(true_means)
    # end def

    def test_logging(self):
        logger = LogHelper.get_logger(__name__)
        x = logger.hasHandlers()

        logger.debug("xxx")
        logger.info("yyy")

