"""Constructs database of interacting fragments."""

import csv
import glob
import os

import numpy as np
import pandas as pd

import query_pymol

def get_id_filename(pdb_id, workspace_root):
    """Given the pdb id, return the full filename for the IDs file."""
    return os.path.join(workspace_root, "IDs/", pdb_id + "_ids.txt")


def get_matrix_filename(pdb_id, workspace_root):
    """Given the pdb id, return the full filename for the matrix file."""
    return os.path.join(workspace_root, "icMatrix/", pdb_id + "_icMat.bmat")


def get_pdb_filename(pdb_id, workspace_root):
    """Given the pdb id, return the full filename for the PDB file."""
    return os.path.join(workspace_root, "cleanPDBs2/", pdb_id + ".pdb")


def read_matrix_from_file(pdb_id, workspace_root):
    """Reads interaction matrix from file and return as np.array.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".
        workspace_root (string): location of workspace for reading and
            storing files

    Returns:
        np.array: interaction matrix as an np.array
    """
    ids_filename = get_id_filename(pdb_id, workspace_root)
    matrix_filename = get_matrix_filename(pdb_id, workspace_root)

    # Read in residue IDs
    ids = pd.read_csv(ids_filename, sep=" ", header=None)
    num_residues = ids.shape[0]

    # Read in binary matrix
    with open(matrix_filename, 'rb') as f:
        raw = np.fromfile(f, np.int32)

    # Found dimensions from corresponding ids.txt file
    matrix = raw.reshape((num_residues, num_residues))

    return matrix


def read_matrix_from_file_df(pdb_id, workspace_root):
    """Read interaction matrix from file, label using IDs file and return as a
    data frame.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".
        workspace_root (string): location of workspace for reading and
            storing files

    Returns:
        pd.DataFrame: data frame containing the matrix, with labels given by
            the rows of the IDs file
    """
    matrix = read_matrix_from_file(pdb_id, workspace_root)
    ids_filename = get_id_filename(pdb_id, workspace_root)

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
    target_indices_np = ((matrix.iloc[cdr_indices, :] < 0).sum(axis=0) > 0).nonzero()
    target_indices = list(target_indices_np[0])

    for index in cdr_indices:
        if index in target_indices:
            target_indices.remove(index)

    return target_indices


def print_targets_to_file(bound_pairs, filename):
    """Prints bound pairs to csv file."""
    rows = [["cdr_residues",
             "cdr_chain",
             "cdr_pdb_indices",
             "target_residues",
             "target_length",
             "target_chain",
             "target_pdb_indices"]]

    for target in bound_pairs:
        rows.append([target['cdr_residues'],
                     target['cdr_chain'],
                     target['cdr_pdb_indices'],
                     target['target_residues'],
                     target['target_length'],
                     target['target_chain'],
                     target['target_pdb_indices']])

    with open(filename, 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def process_database_single(pdb_id, workspace_root, fragment_length):
    """Finds all CDR-like fragments of the given length in the interaction
    matrix for the given pdb_id. Additionally the residues these fragments
    interact with.

    Finds contiguous fragments in the target residues for each CDR and for
    each contiguous fragment, writes the following out to file:
    cdr indices, chain, indices in PDB file, residue names
    same for fragment of target residues

    workspace_root (string): location of workspace for reading and
            storing files

    """
    matrix = read_matrix_from_file(pdb_id, workspace_root)
    bound_pairs, bound_pairs_fragmented = find_all_binding_pairs(matrix,
                                                                 pdb_id,
                                                                 workspace_root,
                                                                 fragment_length)
    directory = os.path.join(workspace_root, "fragment_database/bound_pairs/individual")
    print_targets_to_file(bound_pairs,
                          os.path.join(directory, pdb_id + "_bound_pairs_all.csv"))
    print_targets_to_file(bound_pairs_fragmented,
                          os.path.join(directory, pdb_id + "_bound_pairs_fragmented.csv"))


def read_bound_pairs(filename):
    """Read a csv file containing bound pairs and return the csv"""
    bound_pairs_df = pd.read_csv(filename, header=0, index_col=None)

    return bound_pairs_df


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


def remove_duplicated_bound_pairs(workspace_root, filename=None):
    """
    Reads in all bound_pairs files from the workspace, combines into a csv file
    after removing duplicated rows (based on sequence identify of both cdr and target).
    Args:
        workspace_root (string): directory where database is found
        filename (string): file to write the table to, using None will cause it to
            default to a location in the workspace.

    Writes full table to file.
    """
    all_bound_pairs = combine_all_bound_pairs_fragmented(workspace_root)
    bound_pairs_no_duplicates = remove_duplicate_rows(all_bound_pairs,
                                                      ['cdr_residues', 'target_residues'])

    if not filename:
        filename = os.path.join(workspace_root,
                                "fragment_database/bound_pairs/unique_bound_pairs_fragmented.csv")
    bound_pairs_no_duplicates.to_csv(filename, header=True, index=None)


def process_database(ids_list, workspace_root, fragment_length):
    """Finds all CDR-like fragments of the given length in the files listed in
    the array ids_list. Additionally finds the residues these fragments interact
    with. Outputs all finds to files, with separate files for each pdb_id.
    """
    for pdb_id in ids_list:
        process_database_single(pdb_id, workspace_root, fragment_length)


if __name__ == "__main__":
    # Generate random order using `ls /sharedscratch/kcn25/icMatrix/ |sort -R > random_order.txt`
    with open("random_order.txt") as filelist:
        random_matrix_files = filelist.readlines()

    random_ids = [filename.split("_")[0] for filename in random_matrix_files[:1000]]

    process_database(random_ids, "/sharedscratch/kcn25/", fragment_length=4)
