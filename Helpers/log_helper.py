import logging


class LogHelper:

    def __init__(self):
        pass

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
            logger.propagate = False

        return logger

    # TODO: Implement logging into files.
