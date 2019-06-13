"""Given a list of CDR-like fragments and the target fragments they interact with,
split the list into train, test and validate."""
import argparse
import logging
import json

import numpy as np

import scripts.helper.construct_database as con_dat
import scripts.helper.log_utils as log_utils
import scripts.helper.representations as reps

parser = argparse.ArgumentParser(description="Generate feature matrices for the given bound pairs",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--verbosity",
                    help="verbosity level for logging",
                    default=2,
                    type=int,
                    choices=[0, 1, 2, 3, 4])

required_named = parser.add_argument_group('required named arguments')
required_named.add_argument("--input",
                            required=True,
                            type=argparse.FileType('r'),
                            help="csv file where each row is a bound pair which may or may "
                                 "not have been observed to bind")
required_named.add_argument("--output_file",
                            required=True,
                            help="file to store representation e.g. training/data_meiler.npy")
required_named.add_argument("--fragment_lengths_file",
                            required=True,
                            type=argparse.FileType('r'),
                            help="file containing information about lengths of fragments in data")
required_named.add_argument("--representation",
                            required=True,
                            help="""
representation type to use:
bag_of_words:         'bag of words' vector giving number of times each amino acid 
                      appears in the CDR concatenated with the 'bag of words' vector 
                      for the target
product_bag_of_words: vector formed by flattening matrix where each entry (i,j) of
                      the matrix is the number of times amino acid A_i
                      appears in the CDR `and` amino acid A_j appears in the target
padded_meiler_onehot: flat vector where each set of (21 + 7) entries describes one
                      amino acid in the cdr or target, padded so that all
                      representations have the same length""",
                            choices=['padded_meiler_onehot',
                                     'bag_of_words',
                                     'product_bag_of_words'])

args = parser.parse_args()

log_utils.setup_logging(args.verbosity)

logging.info(f"Generating representations of type {args.representation} for input "
             f"from file '{args.input}'.")

bound_pairs_df = con_dat.read_bound_pairs(args.input)
total_bound_pairs = len(bound_pairs_df)
logging.info(f"Number of bound pairs in complete table: {total_bound_pairs}")

if args.representation == 'bag_of_words':
    representation_matrix = reps.generate_representation_all(bound_pairs_df,
                                                             reps.generate_bagofwords)
elif args.representation == 'product_bag_of_words':
    representation_matrix = reps.generate_representation_all(bound_pairs_df,
                                                             reps.generate_crossed_bagofwords)
elif args.representation == 'padded_meiler_onehot':
    fragment_lengths = json.load(args.fragment_lengths_file)
    max_cdr_len = fragment_lengths['max_cdr_length']
    max_target_len = fragment_lengths['max_target_length']

    representation_matrix = reps.generate_representation_all(
        bound_pairs_df,
        lambda r: reps.generate_padded_onehot_meiler(r, max_cdr_len, max_target_len))
else:
    logging.error(f"Representation type {args.representation} not recognised. Aborting.")
    raise ValueError(f"Representation type {args.representation} not recognised.")

logging.info(f"Representations generated of shape {representation_matrix.shape}:\n"
             f"{representation_matrix}.")

np.save(args.output_file, representation_matrix)

logging.info("Done")
