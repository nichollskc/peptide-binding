"""Given a list of CDR-like fragments and the target fragments they interact with,
removes duplicated pairs from the list."""
# pylint: disable=wrong-import-position
import argparse
import logging
import json
import os
import sys
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

import scripts.helper.construct_database as con_dat
import scripts.helper.log_utils as log_utils
import scripts.helper.utils as utils


def main(bound_pairs_tables, fragment_lengths_out, output_file):
    """Given a list of CDR-like fragments and the target fragments they interact with,
    removes duplicated pairs from the list."""
    filename_list = bound_pairs_tables

    logging.info(f"Combining {len(filename_list)} bound pairs tables: {filename_list}")

    all_bound_pairs = con_dat.combine_bound_pairs(filename_list)
    logging.info(f"Number of bound pairs in combined table: {all_bound_pairs.shape[0]}")

    no_duplicates = con_dat.remove_duplicate_rows(all_bound_pairs,
                                                  ['cdr_resnames', 'target_resnames'])
    logging.info(f"Number of bound pairs after removing duplicates: {no_duplicates.shape[0]}")

    logging.info(f"Saving to file {output_file}")
    utils.save_df_csv_quoted(no_duplicates, output_file)

    if fragment_lengths_out:
        fragment_lengths = {}
        fragment_lengths['max_cdr_length'] = len(no_duplicates['cdr_resnames'].iloc[0])
        fragment_lengths['min_target_length'] = int(no_duplicates['target_length'].min())
        fragment_lengths['max_target_length'] = int(no_duplicates['target_length'].max())

        logging.info(f"Saving fragment lengths to file '{fragment_lengths_out}'\n"
                     f"{fragment_lengths}")
        with open(fragment_lengths_out, 'w') as f:
            json.dump(fragment_lengths, f)

    logging.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given a list of tables with each row "
                    "giving a CDR-like fragment and its target, combines "
                    "the tables and returns the table where duplicated "
                    "rows have been removed.")
    parser.add_argument("--verbosity",
                        help="verbosity level for logging",
                        default=2,
                        type=int,
                        choices=[0, 1, 2, 3, 4])

    parser.add_argument('output_file',
                        help="file to write table of unique bound pairs")
    parser.add_argument('bound_pairs_tables',
                        help="list of files containing tables of bound pairs",
                        type=argparse.FileType('r'),
                        nargs='+')
    parser.add_argument('--fragment_lengths_out',
                        help="file to store lengths of CDR and target fragments found")

    args = parser.parse_args()

    log_utils.setup_logging(args.verbosity)

    main(args.bound_pairs_tables, args.fragment_lengths_out, args.output_file)
