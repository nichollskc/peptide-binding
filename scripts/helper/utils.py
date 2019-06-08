"""Utils functions for rational design"""
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
