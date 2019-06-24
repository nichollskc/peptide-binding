"""Given a list of CDR-like fragments and the target fragments they interact with,
generate an equal number of negative samples by permuting dissimilar CDR fragments.
Dissimilarity is judged by sequence alignment."""
# pylint: disable=wrong-import-position
import argparse
import logging
import os
import sys
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

import pandas as pd

import scripts.helper.construct_database as con_dat
import scripts.helper.log_utils as log_utils
import scripts.helper.utils as utils

parser = argparse.ArgumentParser(description="Given a list of CDR-like fragments and the "
                                             "target fragments they interact with, generate "
                                             "an equal number of negative samples.")
parser.add_argument("--verbosity",
                    help="verbosity level for logging",
                    default=2,
                    type=int,
                    choices=[0, 1, 2, 3, 4])

parser.add_argument('positive_samples',
                    help="file containing positive examples",
                    type=argparse.FileType('r'))
parser.add_argument('output_file',
                    help="file to write table including labelled positives and negatives")

args = parser.parse_args()

log_utils.setup_logging(args.verbosity)

gitlogfile = "logs/git/generate_simple_negatives.gitlog"
logging.info(f"Saving git information to file {gitlogfile}")
log_utils.log_git_info(gitlogfile)

logging.info(f"Reading in table of positive samples from '{args.positive_samples}'.")

positives_df = pd.read_csv(args.positive_samples, header=0)

logging.info("Generating negative examples")

combined_df = con_dat.generate_negatives_alignment_threshold(positives_df)

logging.info(f"Writing positive and negative examples to file '{args.output_file}'.")

utils.save_df_csv_quoted(combined_df, args.output_file)

logging.info("Done")
