"""Tests the file construct_database"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import os
import unittest

import numpy as np
import pandas as pd

import construct_database as con_dat


class Test(unittest.TestCase):

    def test_read_matrix(self):
        matrix = con_dat.read_matrix_from_file("3cuq", workspace_root)
        self.assertTrue(isinstance(matrix, np.ndarray), "Should return numpy array")

    def test_read_matrix_df(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq", workspace_root)
        self.assertTrue(isinstance(matrix, pd.DataFrame), "Should return data frame")

    def test_process_database(self):
        con_dat.process_database(["3cuq", "1mhp"], workspace_root, fragment_length=4)

    def test_interactions_matrix(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq", workspace_root)
        con_dat.find_target_indices_from_matrix(matrix, [0, 1, 2, 3])


if __name__ == '__main__':
    marcopolo_workspace = "/sharedscratch/kcn25/"
    mac_workspace = "/Users/kath/Documents/rational_design_AB/workspace/"

    if os.path.exists(marcopolo_workspace):
        workspace_root = marcopolo_workspace
    else:
        workspace_root = mac_workspace

    unittest.main()
