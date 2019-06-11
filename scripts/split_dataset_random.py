"""Given a list of CDR-like fragments and the target fragments they interact with,
split the list into train, test and validate."""
# pylint: disable=wrong-import-position
import argparse
import logging
import os
import sys
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

import numpy as np

import scripts.helper.construct_database as con_dat
import scripts.helper.log_utils as log_utils
import scripts.helper.utils as utils

parser = argparse.ArgumentParser(description="Given a list of CDR-like fragments and the "
                                             "target fragments they interact with, split the "
                                             "list into separate sets.")
parser.add_argument("--verbosity",
                    help="verbosity level for logging",
                    default=2,
                    type=int,
                    choices=[0, 1, 2, 3, 4])

required_named = parser.add_argument_group('required named arguments')
required_named.add_argument("--seed",
                            help="seed to allow reproducible results",
                            required=True,
                            type=int)
required_named.add_argument("--input",
                            required=True,
                            type=argparse.FileType('r'),
                            help="csv file where each row is a bound pair")
required_named.add_argument("--data_filenames",
                            nargs='+',
                            help="list of .csv filenames for data e.g. "
                                 "'--data_filenames training.csv validation.csv test.csv'")
required_named.add_argument("--label_filenames",
                            nargs='+',
                            help="list of .npy filenames for labels e.g. "
                                 "'--label_filenames training_y.npy validation_y.npy test_y.npy'")
required_named.add_argument("--group_proportions",
                            type=int,
                            nargs='+',
                            help="list of proportions each group should be assigned e.g. "
                                 "'--group_proportions 50 25 25'. Must add up to 100.")

args = parser.parse_args()

log_utils.setup_logging(args.verbosity)

label_filenames = args.label_filenames
group_proportions = args.group_proportions
data_filenames = args.data_filenames

print(args)

# Validate input
assert len(label_filenames) == len(data_filenames), \
    "--label_filenames and --data_filenames must be comma-separated lists of the same length."
assert len(label_filenames) == len(data_filenames), \
    "--label_filenames and --group_proportions must be comma-separated lists of the same length."
assert sum(group_proportions) == 100, \
    "--group_proportions must be a comma-separated list of percentages that add up to 100."
logging.info(f"Splitting rows from file '{args.input}' groups of data: "
             f"{list(zip(data_filenames, label_filenames, group_proportions))}.")

bound_pairs_df = con_dat.read_bound_pairs(args.input)
total_bound_pairs = len(bound_pairs_df)
logging.info(f"Number of bound pairs in complete table: {total_bound_pairs}")

grouped_dfs = con_dat.split_dataset_random(bound_pairs_df, group_proportions, args.seed)

logging.info(f"Split data into {len(grouped_dfs)} sections.")

for df, label_file, data_file, intended_proportion in zip(grouped_dfs,
                                                          label_filenames,
                                                          data_filenames,
                                                          group_proportions):
    logging.info(f"Saving data frame of size {len(df)} to file '{data_file}', "
                 f"saving labels to file '{label_file}'.")
    logging.info(f"Intended proportion is {intended_proportion}, actual proportion "
                 f"is {len(df)/total_bound_pairs}.")
    utils.save_df_csv_quoted(df, data_file)
    labels = np.array(df['binding_observed'])
    np.save(label_file, labels)

logging.info("Done")
