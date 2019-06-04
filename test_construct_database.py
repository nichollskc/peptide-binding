"""Tests the file construct_database"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import os
import unittest

import numpy as np
import pandas as pd
from pymol import cmd

import construct_database as con_dat


class Test(unittest.TestCase):

    def test_read_matrix(self):
        matrix = con_dat.read_matrix_from_file("3cuq", workspace_root)
        self.assertTrue(isinstance(matrix, np.ndarray), "Should return numpy array")

    def test_read_matrix_df(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq", workspace_root)
        self.assertTrue(isinstance(matrix, pd.DataFrame), "Should return data frame")

    def test_process_database(self):
        con_dat.process_database(["3cuq"], workspace_root, fragment_length=4)

    def test_interactions_matrix(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq", workspace_root)
        con_dat.find_target_indices_from_matrix(matrix, [0, 1, 2, 3])

    def test_contiguous_fragments(self):
        cmd.load(con_dat.get_pdb_filename("3cuq", workspace_root))
        residues = [('119', 'I', 'A'),
                    ('121', 'T', 'A'),
                    ('127', 'Q', 'A'),
                    ('128', 'V', 'A'),
                    ('129', 'L', 'A'),
                    ('139', 'V', 'A')]

        expected_03 = [[('127', 'Q', 'A'),
                        ('128', 'V', 'A'),
                        ('129', 'L', 'A')]]
        fragments_03 = con_dat.find_contiguous_fragments(residues,
                                                         "3cuq",
                                                         max_gap=0,
                                                         min_fragment_length=3)
        self.assertEqual(fragments_03, expected_03)

        expected_13 = [[('119', 'I', 'A'),
                        ('120', 'T', 'A'),
                        ('121', 'T', 'A')],
                       [('127', 'Q', 'A'),
                        ('128', 'V', 'A'),
                        ('129', 'L', 'A')]]
        fragments_13 = con_dat.find_contiguous_fragments(residues,
                                                         "3cuq",
                                                         max_gap=1,
                                                         min_fragment_length=3)
        self.assertEqual(fragments_13, expected_13)

        expected_11 = [[('119', 'I', 'A'),
                        ('120', 'T', 'A'),
                        ('121', 'T', 'A')],
                       [('127', 'Q', 'A'),
                        ('128', 'V', 'A'),
                        ('129', 'L', 'A')],
                       [('139', 'V', 'A')]]
        fragments_11 = con_dat.find_contiguous_fragments(residues,
                                                         "3cuq",
                                                         max_gap=1,
                                                         min_fragment_length=1)
        self.assertEqual(fragments_11, expected_11)


if __name__ == '__main__':
    marcopolo_workspace = "/sharedscratch/kcn25/"
    mac_workspace = "/Users/kath/Documents/rational_design_AB/workspace/"

    if os.path.exists(marcopolo_workspace):
        workspace_root = marcopolo_workspace
    else:
        workspace_root = mac_workspace

    unittest.main()
