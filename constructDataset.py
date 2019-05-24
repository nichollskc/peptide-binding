import numpy as np
import pandas

# Read in residue IDs
ids = pandas.read_csv('2zxx_ids.txt', sep=" ", header=None)
num_residues = ids.shape[0]

# Combine the three columns into one label for each residue
combined_labels = ids.apply(lambda x: '_'.join(x.map(str)), axis=1)

# Read in binary matrix
with open('2zxx_icMat.bmat', 'rb') as f:
    raw = np.fromfile(f, np.int32)

# Found dimensions from corresponding ids.txt file
matrix = raw.reshape((num_residues,num_residues))

df = pandas.DataFrame(matrix, index=combined_labels, columns=combined_labels)

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
