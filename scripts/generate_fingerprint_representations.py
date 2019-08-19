import argparse
import logging
# pylint: disable=wrong-import-position
import os
import subprocess
import sys
sys.path.append(os.environ.get('KCN_CURRENT_DIR'))

from e3fp.fingerprint import fprint, fprinter, db
import pandas as pd
from python_utilities.parallel import Parallelizer
from rdkit import Chem
import scipy.sparse

import scripts.helper.log_utils as log_utils
import scripts.helper.query_biopython as qbp


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


def get_sdf_filename_from_pdb_filename(pdb_filename, sdf_filename_root):
    pdb_filename_basename = os.path.basename(pdb_filename)
    bound_pair_id = os.path.splitext(pdb_filename_basename)[0]
    sdf_filename = os.path.join(sdf_filename_root, bound_pair_id) + '.sdf'
    return sdf_filename


def convert_pdb_to_sdf(pdb_filenames, sdf_filename_root):
    """Convert all the PDB files in the list to SDF files."""
    sdf_filenames = [get_sdf_filename_from_pdb_filename(pdb_file, "processed/sdfs")
                     for pdb_file in pdb_filenames]
    lines = [' '.join(pair) + '\n' for pair in zip(pdb_filenames, sdf_filenames)
             if not os.path.exists(pair[1])]
    with open(".tmp.pdb_sdf_filenames.txt", "w") as f:
        f.writelines(lines)
    full_cmd = "parallel -j64 -m -k scripts/helper/run_obabel_convert_batch.sh :::: .tmp.pdb_sdf_filenames.txt"
    logging.debug(f"Full command is {full_cmd}")
    command = subprocess.run(full_cmd.split(" "))
    if command.stderr:
        logging.debug(f"Command to convert PDB files to SD files produced error output:"
                        f"\n{command.stderr.decode('utf-8')}")
    logging.info(f"PDB files converted to SD files.")
    return sdf_filenames


def main(df_filename, outfile):
    logging.info(f"Generating fingerprint database for file {df_filename}.")

    logging.info(f"Reading in bound pairs data frame.")
    bound_pairs_df = pd.read_csv(df_filename)

    logging.info("Checking folder already created for PDB files and SDF files.")
    try:
        os.makedirs("processed/pdbs")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs("processed/sdfs")
    except FileExistsError:
        # directory already exists
        pass

    logging.info(f"Generating PDB files.")
    pdb_filenames = qbp.write_all_bound_pairs_pdb(bound_pairs_df)

    logging.info(f"Converting PDB files to SD files.")
    sdf_filenames = convert_pdb_to_sdf(pdb_filenames, "processed/sdfs")

    logging.info(f"Generating fingerprints for {len(sdf_filenames)} SD files.")
    database = generate_fingerprints_parallel(sdf_filenames, None)

    logging.info(f"Generated {len(database)} fingerprints.")

    logging.info(f"Saving database to file {outfile}")
    scipy.sparse.save_npz(outfile, database.array)

    names = [entry.name for entry in database]
    first_names = '\n'.join(names[:30])
    logging.info(f"Checking fingerprints in the same order as input.\n"
                 f"First few files are:\n{first_names}")
    assert [entry.name for entry in database] == sdf_filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an e3fp fingerprints database for the dataframe given.")
    parser.add_argument("--verbosity",
                        help="verbosity level for logging",
                        default=2,
                        type=int,
                        choices=[0, 1, 2, 3, 4])

    parser.add_argument("--input",
                        required=True,
                        type=argparse.FileType('r'),
                        help="csv file where each row is a bound pair which may or may "
                             "not have been observed to bind")
    parser.add_argument('--outfile',
                        help="file to store fingerprint database")

    args = parser.parse_args()

    log_utils.setup_logging(args.verbosity)

    main(args.input, args.outfile)
