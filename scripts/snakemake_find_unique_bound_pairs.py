"""Wrapper for snakemake to call the find_unique_bound_pairs script"""
# pylint: disable=wrong-import-position
import logging
import os
import sys
import traceback
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

import scripts.find_unique_bound_pairs as find_unique_bound_pairs
import scripts.helper.log_utils as log_utils

log_utils.setup_logging(3, logfile=snakemake.log[0])
try:
    find_unique_bound_pairs.main(bound_pairs_tables=snakemake.input.bound_pairs,
                                 output_file=snakemake.output.bound_pairs,
                                 fragment_lengths_out=snakemake.output.fragment_lengths)
except:
    logging.error(f"Unexpected error:\n{traceback.format_exc()}")
    raise
