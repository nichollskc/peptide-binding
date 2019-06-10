"""Constructs database of interacting fragments."""

import csv
import json
import logging
import random

import numpy as np
import pandas as pd

import scripts.helper.distances as distances
import scripts.helper.query_biopython as query_bp
import scripts.helper.utils as utils


def construct_cdr_id(row):
    """Generates an ID for the given bound pair row e.g. '2xzz_0_3_AAFF'"""
    indices = json.loads(row.cdr_bp_id_str)
    start_ind = indices[0]
    end_ind = indices[-1]
    name = "_".join(map(str, [row.pdb_id, start_ind, end_ind, row.cdr_resnames]))
    return name


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
        logging.warning(f"File '{filename}' didn't have any columns. Ignoring file.")


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
                                                      ['cdr_resnames', 'target_resnames'])

    return bound_pairs_no_duplicates


def sample_index_pairs(data_frame, k):
    """Samples k pairs of rows from the data frame. Returns as a zipped list of
    the random pairs."""
    random_indices = random.choices(list(range(len(data_frame.index))), k=2 * k)

    random_donors = random_indices[:k]
    random_acceptors = random_indices[k:]
    return zip(random_acceptors, random_donors)


def generate_negatives_alignment_threshold(bound_pairs_df, k=None):
    """Given a data frame consisting of bound pairs (i.e. positive examples of
    binding), return a copy with extra rows corresponding to negative examples.
    Negatives are formed by exchanging the CDR of one row for the CDR of another,
    and are marked by the `binding_observed` column being zero. The details of
    the original CDR that has been replaced will be included in the column.

    If (C,T) and (c,t) are two bound pairs, then we can generate a negative
    (C,t) as long as distance(C,c) and distance(T,t) are both sufficiently large.
    In this case distance is measured by sequence alignment.

    This negative will have:
    binding_observed = 0
    cdr = C
    target = t
    original_cdr = c

    Positives will have:
    binding_observed = 1
    cdr = C
    target = T
    original_cdr = NaN

    For each of cdr, target and original_cdr there are fields for PDB id, biopython
    ID string and resnames.
    """
    combined_df = bound_pairs_df.copy()
    combined_df['cdr_pdb_id'] = combined_df['pdb_id']
    combined_df['target_pdb_id'] = combined_df['pdb_id']
    combined_df = combined_df.drop(columns='pdb_id')
    combined_df['binding_observed'] = 1

    if k is None:
        k = len(bound_pairs_df.index)

    logging.info(f"Generating {k} negative examples for dataset "
                 f"containing {len(bound_pairs_df.index)} positive examples.")

    random_index_pairs = sample_index_pairs(bound_pairs_df, k)

    num_negatives_produced = 0
    num_tried = 0
    while num_negatives_produced < k:
        try:
            d_ind, a_ind = next(random_index_pairs)
        except StopIteration:
            # The list is empty, so let's repopulate it and then pop a sample
            #   from the updated list
            logging.info(f"Sampling {k} new pairs of rows.")
            random_index_pairs = sample_index_pairs(bound_pairs_df, k)
            d_ind, a_ind = next(random_index_pairs)

        donor_row = combined_df.iloc[d_ind, :]
        acceptor_row = combined_df.iloc[a_ind, :]
        similarity = distances.calculate_similarity_score_alignment(donor_row, acceptor_row)

        # We assume that exchanging the CDR of the acceptor row for the CDR of the donor row
        #   will produce a negative sample when similarity is less than 0
        if similarity < 0:
            negative = acceptor_row.copy()
            negative.loc['original_cdr_bp_id_str'] = acceptor_row['cdr_bp_id_str']
            negative.loc['original_cdr_resnames'] = acceptor_row['cdr_resnames']
            negative.loc['original_cdr_pdb_id'] = acceptor_row['cdr_pdb_id']

            negative.loc['cdr_bp_id_str'] = donor_row['cdr_bp_id_str']
            negative.loc['cdr_resnames'] = donor_row['cdr_resnames']
            negative.loc['cdr_pdb_id'] = donor_row['cdr_pdb_id']

            negative.loc['binding_observed'] = 0

            combined_df = combined_df.append(negative)

            num_negatives_produced += 1

        num_tried += 1

        if num_negatives_produced % 100 == 0:
            logging.info(f"Produced {num_negatives_produced} negatives "
                         f"from {num_tried} attempts. Latest was \n{negative}")

    return combined_df
