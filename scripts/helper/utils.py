"""Utils functions for rational design"""
import csv
import os


def get_id_filename(pdb_id):
    """Given the pdb id, return the full filename for the IDs file."""
    return os.path.join("IDs/", pdb_id + "_ids.txt")


def get_matrix_filename(pdb_id):
    """Given the pdb id, return the full filename for the matrix file."""
    return os.path.join("icMatrix/", pdb_id + "_icMat.bmat")


def get_pdb_filename(pdb_id):
    """Given the pdb id, return the full filename for the PDB file."""
    return os.path.join("cleanPDBs2/", pdb_id + ".pdb")


def save_df_csv_quoted(data_frame, filename):
    """Saves a dataframe to a csv file, quoting everything to make it safer."""
    data_frame.to_csv(filename, header=True, index=False, quoting=csv.QUOTE_ALL)
