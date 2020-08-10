import logging


class LogHelper:

    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
            logger.propagate = False

        return logger

    # TODO: Implement logging into files.
