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
                                             "list into training, validation and test sets "
                                             "for each threshold value. The first k "
                                             "negatives in the data frame below the threshold "
                                             "will be chosen, along with the first k "
                                             "positives.")
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
required_named.add_argument("--num_negatives",
                            help="number of negative samples to include. The same "
                                 "number of positive samples will be included.",
                            required=True,
                            type=int)
required_named.add_argument("--input",
                            required=True,
                            type=argparse.FileType('r'),
                            help="csv file where each row is a bound pair")
required_named.add_argument("--data_filenames",
                            nargs='+',
                            required=True,
                            help="list of .csv filenames for data to be split randomly e.g. "
                                 "'--data_filenames training_0.csv validation_0.csv "
                                 "training_-2.csv validation_-2.csv'. "
                                 "Assumed to be grouped by threshold, in the same order as "
                                 "the thresholds parameter.")
required_named.add_argument("--label_filenames",
                            nargs='+',
                            required=True,
                            help="list of .npy filenames for labels to be split randomly e.g. "
                                 "'--label_filenames training_y_0.csv validation_y_0.csv "
                                 "training_y_-2.csv validation_y_-2.csv'. "
                                 "Assumed to be grouped by threshold, in the same order as "
                                 "the thresholds parameter.")
required_named.add_argument("--thresholds",
                            type=int,
                            nargs='+',
                            required=True,
                            help="list of thresholds to use to reduce dataset by similarity "
                                 "score e.g. --thresholds 0 -2")

args = parser.parse_args()

log_utils.setup_logging(args.verbosity)

label_filenames = args.label_filenames
data_filenames = args.data_filenames

group_proportions = [80, 20]
thresholds = args.thresholds

logging.info(f"Reading in data frame {args.input}")
data_frame = utils.read_bound_pairs(args.input)
logging.info(f"Read {len(data_frame)} rows")
positives_df = data_frame[data_frame['binding_observed'] == 1].iloc[:args.num_negatives]
logging.info(f"Chosen {len(positives_df)} random positive samples")

label_filenames_grouped = [label_filenames[2*i:2*i+2] for i in range(len(thresholds))]
logging.info(f"{len(label_filenames)} label files grouped into "
             f"{len(label_filenames_grouped)} groups: {label_filenames_grouped}")
data_filenames_grouped = [data_filenames[2*i:2*i+2] for i in range(len(thresholds))]
logging.info(f"{len(data_filenames)} label files grouped into "
             f"{len(data_filenames_grouped)} groups: {data_filenames_grouped}")

for threshold, label_files, data_files in zip(thresholds,
                                              label_filenames_grouped,
                                              data_filenames_grouped):
    logging.info(f"Constructing dataset for similarity threshold {threshold}")
    negatives_df = data_frame[data_frame['similarity_score'] < threshold].iloc[:args.num_negatives]

    logging.info(f"Chosen {len(negatives_df)} random negative samples")

    logging.info(f"Combining positives and negatives into one dataframe")
    combined_df = pd.concat([positives_df, negatives_df])

    logging.info(f"Splitting randomly into different data groups "
                 f"in proportions {group_proportions}")

    grouped_dfs_clusters = cluster.split_dataset_clustered(combined_df,
                                                           group_proportions,
                                                           seed=args.seed)

    for df, label_file, data_file, intended_proportion in zip(grouped_dfs_clusters,
                                                              label_files,
                                                              data_files,
                                                              group_proportions):
        logging.info(f"Saving data frame of size {len(df)} to file '{data_file}', "
                     f"saving labels to file '{label_file}'.")
        logging.info(f"Intended proportion is {intended_proportion}, actual proportion "
                     f"is {len(df) / len(combined_df)}.")
        utils.save_df_csv_quoted(df, data_file)
        labels = np.array(df['binding_observed'])
        np.save(label_file, labels)

    logging.info(f"Done")
