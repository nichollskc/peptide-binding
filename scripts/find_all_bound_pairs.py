"""Only works with snakemake.
Finds all CDR-like fragments and the target fragments they interact with."""
# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.environ.get('current_dir'))

import scripts.construct_database as con_dat
import scripts.query_biopython as query_bp

# Read in parameters from snakemake
pdb_id = snakemake.params.pdb_id
fragment_length = snakemake.params.cdr_fragment_length

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
con_dat.print_targets_to_file(bound_pairs,
                              snakemake.output.complete)
con_dat.print_targets_to_file(bound_pairs_fragmented,
                              snakemake.output.fragmented)

print("Done")
