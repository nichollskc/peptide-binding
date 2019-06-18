"""Utils"""
import logging
import subprocess

verbosity_to_logging_level = {
    0:  logging.CRITICAL,
    1:  logging.ERROR,
    2:  logging.WARNING,
    3:  logging.INFO,
    4:  logging.DEBUG,
}


def setup_logging(verbosity, logfile=None):
    """Initialises logging with custom formatting and a given verbosity level."""
    format_string = "%(asctime)s %(levelname)s | %(module)s - %(funcName)s: %(message)s"
    level = verbosity_to_logging_level[verbosity]

    log_formatter = logging.Formatter(format_string)
    root_logger = logging.getLogger()

    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Set up logging to console with verbosity level {logging.getLevelName(level)}")

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Set up logging to file {logfile} "
                     f"with verbosity level {logging.getLevelName(level)}")


def log_git_info(gitlogfile):
    """Print the git information necessary to identify the code used. Save to a file."""
    cmd = f"(set -x; git log -n 1; git status --untracked-files=no; git diff) > {gitlogfile}"
    subprocess.run(cmd, shell=True)
