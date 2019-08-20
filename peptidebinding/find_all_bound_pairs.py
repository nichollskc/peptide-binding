"""Finds all CDR-like fragments and the target fragments they interact with."""
import argparse
import logging

import peptidebinding.helper.construct_database as con_dat
import peptidebinding.helper.query_biopython as query_bp
import peptidebinding.helper.log_utils as log_utils

parser = argparse.ArgumentParser(description="Process the interaction matrix and .pdb file of "
                                             "a given PDB object and output the CDR-like fragments "
                                             "and their targets in a table. Must specify one of "
                                             "--fragmented_outfile and --complete_outfile.")
parser.add_argument("--verbosity",
                    help="verbosity level for logging",
                    default=2,
                    type=int,
                    choices=[0, 1, 2, 3, 4])

required_named = parser.add_argument_group('required named arguments')
required_named.add_argument("--pdb_id",
                            required=True,
                            help="id of PDB object to search within e.g. '2zxx'")
required_named.add_argument("--cdr_fragment_length",
                            help="length of CDR-like fragment",
                            default=4,
                            type=int,
                            choices=[4, 5, 6, 7, 8])

parser.add_argument("--fragmented_outfile",
                    help="destination for bound pairs where target is fragmented")
parser.add_argument("--complete_outfile",
                    help="destination for bound pairs where target is left complete")

args = parser.parse_args()
if not (args.fragmented_outfile or args.complete_outfile):
    parser.error("No output requested, add --fragmented_outfile or --complete_outfile")

log_utils.setup_logging(args.verbosity)

# Read in parameters from argparse
pdb_id = args.pdb_id
fragment_length = args.cdr_fragment_length

logging.info(f"Finding fragments of length {fragment_length} in {pdb_id}")

# Read in the matrix and find the cdrs and binding pairs within this file
matrix = con_dat.read_matrix_from_file(pdb_id)

logging.info("Read in matrix")

try:
    bound_pairs, bound_pairs_fragmented = query_bp.find_all_binding_pairs(matrix,
                                                                          pdb_id,
                                                                          fragment_length)
except AssertionError as e:
    logging.error(f"Error while searching for bound pairs in file with PDB ID '{pdb_id}'. "
                  f"Will output empty file. "
                  f"Error message was: {e.args[0]}.")
    bound_pairs = []
    bound_pairs_fragmented = []

logging.info(f"Number of bound pairs: {len(bound_pairs)}")
logging.info(f"Number of fragmented bound pairs: {len(bound_pairs_fragmented)}")

# Output to file
if args.complete_outfile:
    con_dat.print_targets_to_file(bound_pairs,
                                  args.complete_outfile)
    logging.info(f"Written complete bound pairs to {args.complete_outfile}")
if args.fragmented_outfile:
    con_dat.print_targets_to_file(bound_pairs_fragmented,
                                  args.fragmented_outfile)
    logging.info(f"Written fragmented bound pairs to {args.fragmented_outfile}")

logging.info("Done")
