"""Constructs database of interacting fragments."""

import numpy as np
import pandas

def read_matrix_data_frame(pdb_id):
    """Read interaction matrix from file, label using IDs file and return as a
    data frame.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".

    Returns:
        pandas.DataFrame: data frame containing the matrix, with labels given by
            the rows of the IDs file
    """
    ids_filename = pdb_id + "_ids.txt"
    matrix_filename = pdb_id + "_icMat.bmat"

    # Read in residue IDs
    ids = pandas.read_csv(ids_filename, sep=" ", header=None)
    num_residues = ids.shape[0]

    # Combine the three columns into one label for each residue
    combined_labels = ids.apply(lambda x: '_'.join(x.map(str)), axis=1)

    # Read in binary matrix
    with open(matrix_filename, 'rb') as f:
        raw = np.fromfile(f, np.int32)

    # Found dimensions from corresponding ids.txt file
    matrix = raw.reshape((num_residues, num_residues))

    df = pandas.DataFrame(matrix, index=combined_labels, columns=combined_labels)
    return df

def findInteractions(matrix, length, threshold=-2):
    """
    Find all interactions of a given length, with interaction counts
    between every pair of residues below the threshold.

    Input:
    np.array matrix: Matrix where rows and columns both correspond to residues
            with rows for CDR-like and columns for interacting-like.
    int length: length of desired interacting pairs
    int threshold: negative number which specifies number of interactions
            required to make this an interaction. ??(-(threshold+1) gives
            number of interactions observed elsewhere)??

    Output:
    ??
    """
    pass

if __name__ == "__main__":
    df_2zxx = read_matrix_data_frame("2zxx")
