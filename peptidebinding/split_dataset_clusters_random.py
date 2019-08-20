"""Given a list of CDR-like fragments and the target fragments they interact with,
split the list into train, test and validate."""
import argparse
import logging

import numpy as np
import pandas as pd

import peptidebinding.helper.construct_database as con_dat
import peptidebinding.helper.log_utils as log_utils
import peptidebinding.helper.utils as utils
import peptidebinding.helper.cluster_sequences as cluster

parser = argparse.ArgumentParser(description="Given a list of CDR-like fragments and the "
                                             "target fragments they interact with, split the "
                                             "list into training, validation and test sets.")
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
                            nargs=6,
                            required=True,
                            help="list of .csv filenames for data to be split randomly e.g. "
                                 "'--data_filenames training_rand.csv validation_rand.csv "
                                 "test_rand.csv training_clustered.csv "
                                 "validation_clustered.csv test_clustered.csv'")
required_named.add_argument("--label_filenames",
                            nargs=6,
                            required=True,
                            help="list of .npy filenames for labels to be split randomly e.g. "
                                 "'--label_filenames training_rand_y.csv validation_rand_y.csv "
                                 "test_rand_y.csv training_clustered_y.csv "
                                 "validation_clustered_y.csv test_clustered_y.csv'")
required_named.add_argument("--group_proportions",
                            type=int,
                            nargs=4,
                            required=True,
                            help="list of proportions each group should be assigned e.g. "
                                 "'--group_proportions 60 20 10 10'. Must add up to 100. "
                                 "The order should be "
                                 "'training validation test_clustered test_random'.")

args = parser.parse_args()

log_utils.setup_logging(args.verbosity)

label_filenames = args.label_filenames
data_filenames = args.data_filenames

group_proportions = args.group_proportions

print(args)

assert sum(group_proportions) == 100, \
    "--group_proportions must be a comma-separated list of percentages that add up to 100."
logging.info(f"Splitting rows from file '{args.input}' into groups of data.")

bound_pairs_df = con_dat.read_bound_pairs(args.input)
total_bound_pairs = len(bound_pairs_df)
logging.info(f"Number of bound pairs in complete table: {total_bound_pairs}")

logging.info(f"Performing initial allocation of random test data")
size_test_random = group_proportions[-1]
# pylint: disable=unbalanced-tuple-unpacking
remaining_df, test_random_df = con_dat.split_dataset_random(bound_pairs_df,
                                                            [100 - size_test_random,
                                                             size_test_random],
                                                            args.seed)

logging.info(f"Performing split of remaining data ({len(remaining_df)} rows) by clustering.")
grouped_dfs_clusters = cluster.split_dataset_clustered(remaining_df,
                                                       group_proportions[:-1],
                                                       seed=args.seed)

logging.info(f"Combining cluster partitions of training and validation data "
             f"to get whole 'non-test' set.")
non_test_data = pd.concat(grouped_dfs_clusters[:-1])
logging.info(f"Performing split of remaining data ({len(non_test_data)} rows) randomly.")
grouped_dfs_random = con_dat.split_dataset_random(non_test_data,
                                                  group_proportions[:-2],
                                                  seed=args.seed)

logging.info(f"Creating lists with all the partitions and their corresponding filenames")
grouped_dfs = grouped_dfs_random + [test_random_df] + grouped_dfs_clusters
intended_group_proportions = [group_proportions[0],
                              group_proportions[1],
                              group_proportions[3],
                              group_proportions[0],
                              group_proportions[1],
                              group_proportions[2]]

logging.info(f"Split data into {len(grouped_dfs)} sections.")

for df, label_file, data_file, intended_proportion in zip(grouped_dfs,
                                                          label_filenames,
                                                          data_filenames,
                                                          intended_group_proportions):
    logging.info(f"Saving data frame of size {len(df)} to file '{data_file}', "
                 f"saving labels to file '{label_file}'.")
    logging.info(f"Intended proportion is {intended_proportion}, actual proportion "
                 f"is {len(df)/total_bound_pairs}.")
    utils.save_df_csv_quoted(df, data_file)
    labels = np.array(df['binding_observed'])
    np.save(label_file, labels)

logging.info("Done")
