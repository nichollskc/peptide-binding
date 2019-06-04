"""Only works with snakemake.
Finds all CDR-like fragments and the target fragments they interact with."""
import scripts.construct_database as con_dat
import scripts.query_pymol as query_pymol

# Read in parameters from snakemake
pdb_id = snakemake.params.pdb_id
fragment_length = snakemake.params.cdr_fragment_length

# Read in the matrix and find the cdrs and binding pairs within this file
matrix = con_dat.read_matrix_from_file(pdb_id)
bound_pairs, bound_pairs_fragmented = query_pymol.find_all_binding_pairs(matrix,
                                                                         pdb_id,
                                                                         fragment_length)

# Output to file
con_dat.print_targets_to_file(bound_pairs,
                              snakemake.output.complete)
con_dat.print_targets_to_file(bound_pairs_fragmented,
                              snakemake.output.fragmented)
