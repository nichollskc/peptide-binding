"""Calculates distances between bound pairs"""
import re
import subprocess

import numpy as np


def calculate_alignment_score(seq1, seq2):
    """Calculates the alignment score between two protein sequences."""
    full_cmd = "/Users/kath/tools/seq-align/bin/needleman_wunsch " \
               "--scoring BLOSUM62 --printscores --gapopen 0 " \
               "{} {}".format(seq1, seq2)
    alignment = subprocess.run(full_cmd.split(" "), capture_output=True)
    score = re.match(r".*\n.*\nscore: (-?\d+)\n\n", alignment.stdout.decode("utf-8"))[1]

    return int(score)


def calculate_distance_matrix(bound_pairs_df):
    """Given a data frame containing columns 'cdr_residues' and 'target_residues',
    constructs a distance matrix between each pair of rows."""
    assert 'cdr_residues' in bound_pairs_df.columns
    assert 'target_residues' in bound_pairs_df.columns

    # Initialise empty distance matrix
    num_rows = len(bound_pairs_df)
    distance_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i):
            cdr_distance = calculate_alignment_score(bound_pairs_df.loc[i, 'cdr_residues'],
                                                     bound_pairs_df.loc[j, 'cdr_residues'])
            target_distance = calculate_alignment_score(bound_pairs_df.loc[i, 'target_residues'],
                                                        bound_pairs_df.loc[j, 'target_residues'])
            distance = cdr_distance + target_distance
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix
