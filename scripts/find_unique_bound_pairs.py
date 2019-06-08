"""Given a list of CDR-like fragments and the target fragments they interact with,
removes duplicated pairs from the list."""
import argparse

import scripts.helper.construct_database as con_dat

parser = argparse.ArgumentParser(description="Given a list of tables with each row "
                                             "giving a CDR-like fragment and its target, combines "
                                             "the tables and returns the table where duplicated "
                                             "rows have been removed.")
parser.add_argument('output_file',
                    help="file to write table of unique bound pairs")
parser.add_argument('bound_pairs_tables',
                    help="list of files containing tables of bound pairs",
                    type=argparse.FileType('r'),
                    nargs='+')

args = parser.parse_args()

filename_list = args.bound_pairs_tables

all_bound_pairs = con_dat.combine_bound_pairs(filename_list)
no_duplicates = con_dat.remove_duplicate_rows(all_bound_pairs,
                                              ['cdr_resnames', 'target_resnames'])

no_duplicates.to_csv(args.output_file, header=True, index=None)
