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

def find_interactor_indices(matrix, cdr_indices):
    """
    Find all indices that interact with the given CDR according to the matrix.

    Args:
        matrix (np.array): square matrix giving interactions between residues,
            described further in find_all_binding_pairs_indices.
        cdr_indices (array): array of indices for which we want to find
            interactions.

    Returns:
        array: array of indices that interact with any of the indices of CDR.
    """
    interactor_indices = []

    # For each index in the CDR-like fragment, find all the residues it
    #   interacts with. These will have negative value in the matrix.
    for index in cdr_indices:
        negative_entries = (matrix[index] < 0).nonzero()
        interactor_indices += negative_entries

    return interactor_indices

def find_all_binding_pairs_indices(matrix, fragment_length):
    """
    Find all CDR-like regions of given length in the matrix, and also find
    the residues the CDR-like regions interact with in this matrix.

    Args:
        matrix (np.array): Interaction matrix where rows and columns both
            correspond to residues. Full description below.
        fragment_length (int): length of desired interacting pairs

    Returns:
        array[array[array]]: Each array corresponds to a binding pair and
            contains two arrays: the first is an array of indices in the
            CDR-like fragment and the second is an array of indices that
            interact with those indices.
            The CDR-like fragment will contain have length precisely
            fragment_length, but the interacting indices may not be this length.

    The interaction matrix tells us which fragments are CDR-like and also which
    residues interact. If M_{x,y} is negative then the residue x interacts with
    the residue y. Additionally, if M_{x,y} < -1 or M_{x,y} > 0 then the
    fragment x, x+1, ..., y is a CDR-like region. The magnitude indicates how
    many similar fragments were found in CDRs.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix is assumed to be square"

    matrix_size = matrix.shape[0]
    binding_pairs = []

    for start_index in range(0, matrix_size - fragment_length):
        end_index = start_index + fragment_length - 1
        matrix_entry = matrix[start_index][end_index]

        # First check if this fragment is a CDR-like fragment i.e. has it been
        #    observed in a CDR.
        if  matrix_entry > 0 or matrix_entry < -1:
            # Get the indices belonging to this fragment - note range() excludes
            #   second number given
            cdr_indices = list(range(start_index, end_index + 1))
            interactor_indices = find_interactor_indices(matrix, cdr_indices)

            binding_pair = [cdr_indices, interactor_indices]
            binding_pairs.append(binding_pair)

    return binding_pairs

if __name__ == "__main__":
    df_2zxx = read_matrix_data_frame("2zxx")
