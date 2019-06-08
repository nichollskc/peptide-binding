"""Finds all CDR-like fragments and the target fragments they interact with."""
import argparse

import scripts.helper.construct_database as con_dat
import scripts.helper.query_biopython as query_bp

parser = argparse.ArgumentParser(description="Process the interaction matrix and .pdb file of "
                                             "a given PDB object and output the CDR-like fragments "
                                             "and their targets in a table. Must specify one of "
                                             "--fragmented_outfile and --complete_outfile.")
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

# Read in parameters from argparse
pdb_id = args.pdb_id
fragment_length = args.cdr_fragment_length

print("Finding fragments of length {} in {}".format(fragment_length, pdb_id))

# Read in the matrix and find the cdrs and binding pairs within this file
matrix = con_dat.read_matrix_from_file(pdb_id)

print("Read in matrix")

bound_pairs, bound_pairs_fragmented = query_bp.find_all_binding_pairs(matrix,
                                                                      pdb_id,
                                                                      fragment_length)

print("Number of bound pairs:", len(bound_pairs))
print("Number of fragmented bound pairs:", len(bound_pairs_fragmented))

# Output to file
if args.complete_outfile:
    con_dat.print_targets_to_file(bound_pairs,
                                  args.complete_outfile)
if args.fragmented_outfile:
    con_dat.print_targets_to_file(bound_pairs_fragmented,
                                  args.fragmented_outfile)
    print("Written fragmented bound pairs to {}".format(args.fragmented_outfile))

print("Done")
