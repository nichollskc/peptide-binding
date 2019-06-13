"""Only works with snakemake.
Calculates distances between each pair of rows of the table."""
import numpy as np
import pandas as pd

import scripts.helper.distances as distances

bound_pairs_df = pd.read_csv(snakemake.input.bound_pairs,
                             index_col=None,
                             header=0)

distance_matrix = distances.calculate_distance_matrix(bound_pairs_df)

# Output to file
np.save(snakemake.output.distance_matrix, distance_matrix)
