"""Constructs database of interacting fragments."""

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
    utils.save_df_csv_quoted(df, filename)


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


def generate_proposal_negatives(data_frame, k):
    """Given a data frame, shuffle and pair the rows to produce a set of k proposal
    negative examples. Return these in a data frame."""
    logging.info(f"Generating {k} negatives for data frame of length {len(data_frame)}")

    proposals_df = data_frame.sample(n=k, replace=True).reset_index(drop=True).join(
        data_frame.sample(n=k, replace=True).reset_index(drop=True),
        rsuffix="_donor")

    logging.info(f"Updating column values for these proposals")
    proposals_df['original_cdr_bp_id_str'] = proposals_df['cdr_bp_id_str']
    proposals_df['original_cdr_resnames'] = proposals_df['cdr_resnames']
    proposals_df['original_cdr_pdb_id'] = proposals_df['cdr_pdb_id']

    proposals_df['cdr_bp_id_str'] = proposals_df['cdr_bp_id_str_donor']
    proposals_df['cdr_resnames'] = proposals_df['cdr_resnames_donor']
    proposals_df['cdr_pdb_id'] = proposals_df['cdr_pdb_id_donor']

    proposals_df['binding_observed'] = 0

    logging.info(f"Updated column values for these proposals")

    return proposals_df


def remove_invalid_negatives(combined_df):
    """Finds similarity between the rows that have been combined to form negatives
    and removes any that are formed by rows that are too similar. Judged by
    sequence alignment between CDRs and targets."""
    new_negatives_rows = (combined_df['binding_observed'] == 0) & \
                         (np.isnan(combined_df['similarity_score']))
    logging.info(f"Verifying {(new_negatives_rows).sum()} new negatives")

    cdr_scores = distances.calculate_alignment_scores(combined_df.loc[new_negatives_rows,
                                                                      'cdr_resnames'],
                                                      combined_df.loc[new_negatives_rows,
                                                                      'original_cdr_resnames'])
    target_scores = distances.calculate_alignment_scores(combined_df.loc[new_negatives_rows,
                                                                         'target_resnames'],
                                                         combined_df.loc[new_negatives_rows,
                                                                         'target_resnames_donor'])

    logging.info(f"Computed scores")
    total_scores = [sum(scores) for scores in zip(cdr_scores, target_scores)]
    combined_df.loc[new_negatives_rows, 'similarity_score'] = total_scores

    too_similar_indices = combined_df.index[(combined_df['similarity_score'] >= 0)]
    logging.info(f"Rejected {len(too_similar_indices)} rows for being too similar")
    combined_df = combined_df.drop(too_similar_indices, axis=0)
    return combined_df


def generate_negatives_alignment_threshold(bound_pairs_df, k=None, seed=42):
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
    np.random.seed(seed)

    positives_df = bound_pairs_df.copy()
    positives_df['cdr_pdb_id'] = positives_df['pdb_id']
    positives_df['target_pdb_id'] = positives_df['pdb_id']
    positives_df = positives_df.drop(columns='pdb_id')
    positives_df['binding_observed'] = 1
    positives_df['similarity_score'] = np.nan
    combined_df = positives_df.copy()

    if k is None:
        k = len(bound_pairs_df.index)

    logging.info(f"Generating {k} negative examples for dataset "
                 f"containing {len(bound_pairs_df.index)} positive examples.")

    while len(combined_df) < k + len(positives_df):
        # Generate proposals which might be negative - by shuffling two versions of
        #   the positives data frame
        proposals_df = generate_proposal_negatives(positives_df, 2 * k)

        # Combine the positives and proposed negatives, and remove duplicate values
        combined_df = pd.concat([combined_df, proposals_df], sort=False).reset_index(drop=True)
        combined_df = remove_duplicate_rows(combined_df,
                                            ['cdr_resnames', 'target_resnames'])

        combined_df = remove_invalid_negatives(combined_df)

    combined_df = combined_df.iloc[:len(positives_df) + k, :]

    good_cols = [col for col in combined_df.columns if not col.endswith('donor')]
    combined_df = combined_df[good_cols]

    return combined_df


def split_dataset_random(data_frame, group_proportions, seed=42):
    """Splits the rows of a data frame randomly into groups according to the group
    proportions. Group proportions should be a list e.g. [60, 20, 20]."""
    # np.split requires a list of cummulative fractions for the groups
    #   e.g. [60, 20, 20] -> [0.6, 0.2, 0.2] -> [0.6, 0.6 + 0.2] = [0.6, 0.8]
    fractions = np.cumsum([group_proportions])[:-1] / 100

    logging.info(f"Intended fractions are {fractions}")
    counts = list(map(int, (fractions * len(data_frame))))
    logging.info(f"Intended counts per group are {counts}")
    grouped_dfs = np.split(data_frame.sample(frac=1, random_state=seed), counts)

    return grouped_dfs
