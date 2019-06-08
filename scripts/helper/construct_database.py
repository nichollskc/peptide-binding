"""Constructs database of interacting fragments."""

import csv

import numpy as np
import pandas as pd

import scripts.helper.query_biopython as query_bp
import scripts.helper.utils as utils


def read_matrix_from_file(pdb_id):
    """Reads interaction matrix from file and return as np.array.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".

    Returns:
        np.array: interaction matrix as an np.array
    """
    ids_filename = utils.get_id_filename(pdb_id)
    matrix_filename = utils.get_matrix_filename(pdb_id)

    # Read in residue IDs
    ids = pd.read_csv(ids_filename, sep=" ", header=None)
    num_residues = ids.shape[0]

    # Read in binary matrix
    with open(matrix_filename, 'rb') as f:
        raw = np.fromfile(f, np.int32)

    # Found dimensions from corresponding ids.txt file
    matrix = raw.reshape((num_residues, num_residues))

    return matrix


def read_matrix_from_file_df(pdb_id):
    """Read interaction matrix from file, label using IDs file and return as a
    data frame.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".

    Returns:
        pd.DataFrame: data frame containing the matrix, with labels given by
            the rows of the IDs file
    """
    matrix = read_matrix_from_file(pdb_id)
    ids_filename = utils.get_id_filename(pdb_id)

    # Combine the three columns into one label for each residue
    ids = pd.read_csv(ids_filename, sep=" ", header=None)
    combined_labels = ids.apply(lambda x: '_'.join(x.map(str)), axis=1)

    df = pd.DataFrame(matrix, index=combined_labels, columns=combined_labels)
    return df


def find_target_indices_from_matrix(matrix, cdr_indices):
    """
    Finds all indices that interact with the given CDR according to the matrix.

    Args:
        matrix (np.array): square matrix giving interactions between residues,
            described further in find_all_binding_pairs_indices.
        cdr_indices (array): array of indices for which we want to find
            interactions.

    Returns:
        array: array of indices that interact with any of the indices of CDR.
    """
    target_indices_np = ((matrix.iloc[cdr_indices, :] < 0).sum(axis=0) > 0).to_numpy().nonzero()
    target_indices = list(target_indices_np[0])

    for index in cdr_indices:
        if index in target_indices:
            target_indices.remove(index)

    return target_indices


def print_targets_to_file(bound_pairs, filename):
    """Prints bound pairs to csv file."""
    df = pd.DataFrame(bound_pairs)
    df.to_csv(filename, header=True, index=False, quoting=csv.QUOTE_ALL)


def find_bound_pairs(pdb_id, fragment_length):
    """Finds all CDR-like fragments of the given length in the interaction
    matrix for the given pdb_id. Additionally the residues these fragments
    interact with.

    Finds contiguous fragments in the target residues for each CDR and for
    each contiguous fragment, writes the following out to file:
    cdr indices, chain, indices in PDB file, residue names
    same for fragment of target residues
    """
    matrix = read_matrix_from_file(pdb_id)
    bound_pairs, bound_pairs_fragmented = query_bp.find_all_binding_pairs(matrix,
                                                                          pdb_id,
                                                                          fragment_length)

    return bound_pairs, bound_pairs_fragmented


def read_bound_pairs(filename):
    """Read a csv file containing bound pairs and return the csv"""
    try:
        bound_pairs_df = pd.read_csv(filename, header=0, index_col=None)
        return bound_pairs_df
    except pd.errors.EmptyDataError:
        print("File '{}' didn't have any columns. Ignoring file.".format(filename))


def combine_bound_pairs(filename_list):
    """Read in all the bound pairs from the csv files in `filename_list` and
    combine them into a single dataframe"""
    data_frames = [read_bound_pairs(filename) for filename in filename_list]
    combined_data_frame = pd.concat(data_frames)
    return combined_data_frame


def remove_duplicate_rows(data_frame, columns):
    """
    Removes rows from the data frame if the values in all columns specified are the same.
    The first duplicate of each set will be removed.

    Args:
        data_frame (pandas.DataFrame): data frame
        columns (array): array of column names e.g. ['cdr_residues', 'target_residues']
            rows must match on all the given columns to count as a duplicate

    Returns:
        pandas.DataFrame: data frame which is a copy of the original but with
            duplicated rows removed.
    """
    row_is_duplicate = data_frame.duplicated(columns, keep='first')
    no_duplicates = data_frame[~ row_is_duplicate]

    return no_duplicates


def find_all_bound_pairs(ids_list, fragment_length):
    """Finds all CDR-like fragments of the given length in the files listed in
    the array ids_list. Additionally finds the residues these fragments interact
    with. Outputs all finds to files, with separate files for each pdb_id.
    """
    for pdb_id in ids_list:
        find_bound_pairs(pdb_id, fragment_length)


def find_unique_bound_pairs(filename_list):
    """
    Reads in all bound_pairs files from the filename_list, combines into a csv file
    after removing duplicated rows (based on sequence identify of both cdr and target).
    Args:
        filename_list (array): list of files containing bound_pairs

    Returns:
        pandas.DataFrame: data frame where each row is a bound pair and duplicates
            have been removed
    """
    all_bound_pairs = combine_bound_pairs(filename_list)
    bound_pairs_no_duplicates = remove_duplicate_rows(all_bound_pairs,
                                                      ['cdr_residues', 'target_residues'])

    return bound_pairs_no_duplicates
