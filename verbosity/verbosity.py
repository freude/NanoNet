import logging
import warnings


VERBOSITY = 1


def set_verbosity(level):
    global VERBOSITY
    if level > 0:
        logging.getLogger().setLevel(logging.INFO)
        VERBOSITY = level
    else:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.CRITICAL)
        VERBOSITY = level
