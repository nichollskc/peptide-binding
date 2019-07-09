import argparse
import logging
# pylint: disable=wrong-import-position
import os
import sys
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

from e3fp.fingerprint import fprint, fprinter, db
from python_utilities.parallel import Parallelizer
from rdkit import Chem
import scipy.sparse

import scripts.helper.log_utils as log_utils


def generate_e3fp_fingerprint(sdf_file):
    logging.debug(f"Generating fingerprint for file {sdf_file}")
    # Load molecule from file using rdkit 
    molecule = Chem.SDMolSupplier(sdf_file, sanitize=False)[0]
    # Calculate necessary properties
    molecule.UpdatePropertyCache(strict=False)

    # Set up fingerprinter and generate fingerprint for this molecule
    fingerprinter = fprinter.Fingerprinter(bits=1048576)
    fingerprinter.run(mol=molecule)
    result = fingerprinter.get_fingerprint_at_level()

    # Set name of the fingerprint to the sdf file name
    result.name = sdf_file

    return result


def generate_fingerprints_parallel(sdf_files, threads):
    """Generate fingerprints for each sdf_file and construct a database.
    If threads=None, use all available processors, else specify an integral number
    of threads to use in parallel."""
    empty_fp = fprint.Fingerprint([])
    parallelizer = Parallelizer(parallel_mode="processes",
                                num_proc=threads,
                                fail_value=empty_fp)

    wrapped_files = [[f] for f in sdf_files]
    results = parallelizer.run(generate_e3fp_fingerprint, wrapped_files)

    results_dict = {}
    for r in results:
        results_dict[r[1][0]] = r[0]
    fingerprints = [results_dict[name] for name in sdf_files]

    database = db.FingerprintDatabase()
    database.add_fingerprints(fingerprints)
    return database


def main(input_files, outfile):
    logging.info(f"Generating fingerprints for {len(input_files)} SD files.")
    database = generate_fingerprints_parallel(input_files, None)

    logging.info(f"Generated {len(database)} fingerprints.")

    logging.info(f"Saving database to file {outfile}")
    scipy.sparse.save_npz(outfile, database.array)

    names = [entry.name for entry in database]
    first_names = '\n'.join(names[:30])
    logging.info(f"Checking fingerprints in the same order as input.\n"
                 f"First few files are:\n{first_names}")
    assert [entry.name for entry in database] == input_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an e3fp fingerprints database for all the SD files given.")
    parser.add_argument("--verbosity",
                        help="verbosity level for logging",
                        default=2,
                        type=int,
                        choices=[0, 1, 2, 3, 4])

    parser.add_argument('input',
                        help="names of SD files to generate fingerprints for",
                        nargs='+')
    parser.add_argument('--outfile',
                        help="file to store fingerprint database")

    args = parser.parse_args()

    log_utils.setup_logging(args.verbosity)

    main(args.input, args.outfile)
