"""Calculates distances between bound pairs"""
import logging
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
    logging.info(f"Computing aligments between two columns of length "
                 f"{len(column_1)} and {len(column_2)}")
    lines = ['_'.join(pair) + '\n' for pair in zip(column_1, column_2)]
    with open(".tmp.sequences.txt", "w") as f:
        f.writelines(lines)
    full_cmd = "parallel -j64 -m -k scripts/helper/run_seq_align_batch.sh :::: .tmp.sequences.txt"
    logging.debug(f"Full command is {full_cmd}")
    alignments = subprocess.run(full_cmd.split(" "),
                                capture_output=True)
    if alignments.stderr:
        logging.warning(f"Command to calculate alignment scores produced error output:"
                        f"\n{alignments.stderr.decode('utf-8')}")
    output = alignments.stdout.decode('utf-8')
    logging.info(f"Alignments computed. Size of output is {len(output)}. Head of output is:\n"
                 f"{output:.100s}")
    logging.debug(f"Alignments computed. Output is:\n{alignments.stdout.decode('utf-8')}")
    scores = list(map(int, alignments.stdout.decode("utf-8").strip().split("\n")))
    logging.info(f"Alignments decoded")
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


def calculate_distance_matrix(data_frame, columns):
    """Given a data frame, constructs a distance matrix between each pair of rows
    where the distance is the sum of the alignment scores between rows for each
    column in the list columns.
    E.g. if columns = ['cdr_resnames', 'target_resnames'] then the distance
    will be alignment(row_1_cdr, row_2_cdr) + alignment(row_1_target, row_2_target)."""

    # Initialise empty distance matrix
    num_rows = len(data_frame)
    distance_matrix = np.zeros((num_rows, num_rows))

    x_inds, y_inds = np.triu_indices(len(data_frame))

    for c in columns:
        distance_matrix[x_inds, y_inds] += calculate_alignment_scores(data_frame[c].iloc[x_inds],
                                                                      data_frame[c].iloc[y_inds])
        distance_matrix[y_inds, x_inds] += calculate_alignment_scores(data_frame[c].iloc[x_inds],
                                                                      data_frame[c].iloc[y_inds])

    return distance_matrix
