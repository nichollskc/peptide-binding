"""Only works with snakemake.
Given a list of CDR-like fragments and the target fragments they interact with,
removes duplicated pairs from the list."""
import scripts.construct_database as con_dat

filename_list = list(snakemake.input)

all_bound_pairs = con_dat.combine_bound_pairs(filename_list)
no_duplicates = con_dat.remove_duplicate_rows(all_bound_pairs,
                                              ['cdr_residues', 'target_residues'])

no_duplicates.to_csv(snakemake.output[0], header=True, index=None)
