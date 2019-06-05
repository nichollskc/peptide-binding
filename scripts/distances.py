"""Calculates distances between bound pairs"""
import re
import subprocess


def calculate_alignment_score(seq1, seq2):
    """Calculates the alignment score between two protein sequences."""
    full_cmd = "/Users/kath/tools/seq-align/bin/needleman_wunsch " \
               "--scoring BLOSUM62 --printscores --gapopen 0 " \
               "{} {}".format(seq1, seq2)
    alignment = subprocess.run(full_cmd.split(" "), capture_output=True)
    score = re.match(r".*\n.*\nscore: (-?\d+)\n\n", alignment.stdout.decode("utf-8"))[1]

    return score
