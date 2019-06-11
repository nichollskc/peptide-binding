"""Utils"""
import logging

verbosity_to_logging_level = {
    0:  logging.CRITICAL,
    1:  logging.ERROR,
    2:  logging.WARNING,
    3:  logging.INFO,
    4:  logging.DEBUG,
}


def setup_logging(verbosity):
    """Initialises logging with custom formatting and a given verbosity level."""
    format_string = "%(asctime)s %(levelname)s | %(module)s - %(funcName)s: %(message)s"
    level = verbosity_to_logging_level[verbosity]
    logging.basicConfig(level=level,
                        format=format_string,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"Set up logging with verbosity level {logging.getLevelName(level)}")
