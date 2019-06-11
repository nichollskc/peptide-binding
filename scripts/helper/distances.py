"""Calculates distances between bound pairs"""
import re
import subprocess

import numpy as np


def calculate_alignment_score(seq1, seq2):
    """Calculates the alignment score between two protein sequences."""
    full_cmd = "seq-align/bin/needleman_wunsch " \
               "--scoring BLOSUM62 --printscores --gapopen 0 " \
               "{} {}".format(seq1, seq2)
    alignment = subprocess.run(full_cmd.split(" "), capture_output=True)
    score = re.match(r".*\n.*\nscore: (-?\d+)\n\n", alignment.stdout.decode("utf-8"))[1]

    return int(score)


def calculate_alignment_scores(column_1, column_2):
    """Calculates the alignment score for each row, where the score is the
    alignment between the element in the row in column_1 and the element in the
    row at column_2."""
    pairs_strings = [" ".join(pair) + "\n" for pair in zip(column_1, column_2)]
    full_input = bytes("".join(pairs_strings), 'utf-8')
    alignments = subprocess.run("scripts/helper/run_seq_align.sh",
                                input=full_input,
                                capture_output=True)
    scores = list(map(int, alignments.stdout.decode("utf-8").strip().split("\n")))
    return scores


def calculate_similarity_score_alignment(row1, row2):
    """Calculates the similarity score between two bound pairs, where they are given
    as rows of a pandas.DataFrame with columns 'cdr_resnames' and 'target_resnames'.
    The measure of similarity is the sum of the alignment score of the CDR fragments
    and of the target fragments."""
    cdr_distance = calculate_alignment_score(row1['cdr_resnames'],
                                             row2['cdr_resnames'])
    target_distance = calculate_alignment_score(row1['target_resnames'],
                                                row2['target_resnames'])

    similarity = cdr_distance + target_distance
    return similarity


def calculate_distance_matrix(bound_pairs_df):
    """Given a data frame containing columns 'cdr_resnames' and 'target_resnames',
    constructs a distance matrix between each pair of rows."""
    assert 'cdr_resnames' in bound_pairs_df.columns
    assert 'target_resnames' in bound_pairs_df.columns

    # Initialise empty distance matrix
    num_rows = len(bound_pairs_df)
    distance_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i + 1):
            distance = calculate_similarity_score_alignment(bound_pairs_df.iloc[i, :],
                                                            bound_pairs_df.iloc[j, :])

            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix
