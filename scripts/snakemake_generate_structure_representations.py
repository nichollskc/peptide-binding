"""Wrapper for snakemake to call the find_unique_bound_pairs script"""
# pylint: disable=wrong-import-position
import logging
import os
import sys
import traceback
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

import scripts.generate_structure_representations as generate_structure_representations
import scripts.helper.log_utils as log_utils

log_utils.setup_logging(3, logfile=snakemake.log[0])
try:
    generate_structure_representations.main(input_files=snakemake.input.sdf_filenames,
                                            outfile=snakemake.output.dataset)
except:
    logging.error(f"Unexpected error:\n{traceback.format_exc()}")
    raise
