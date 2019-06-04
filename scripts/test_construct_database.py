"""Tests the file construct_database"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import os
import unittest

import numpy as np
import pandas as pd

import construct_database as con_dat

class LongTest(unittest.TestCase):

    def test_process_database(self):
        con_dat.process_database(["3cuq", "1mhp"], workspace_root, fragment_length=4)

    def test_remove_duplicates(self):
        con_dat.remove_duplicated_bound_pairs(workspace_root)

class ShortTest(unittest.TestCase):

    def test_read_matrix(self):
        matrix = con_dat.read_matrix_from_file("3cuq", workspace_root)
        self.assertTrue(isinstance(matrix, np.ndarray), "Should return numpy array")

    def test_read_matrix_df(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq", workspace_root)
        self.assertTrue(isinstance(matrix, pd.DataFrame), "Should return data frame")

    def test_interactions_matrix(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq", workspace_root)
        con_dat.find_target_indices_from_matrix(matrix, [0, 1, 2, 3])

    def test_remove_duplicates(self):
        df = pd.DataFrame(data=[[1, 10, 100],
                                [2, 20, 100],
                                [3, 20, 100],
                                [4, 30, 200],
                                [5, 30, 200],
                                [6, 10, 300]],
                          columns=["ID", "a", "b"])

        # Rows are duplicates if they have matching value in column a
        # Keeps the *first* duplicate
        trimmed_a = con_dat.remove_duplicate_rows(df, ['a'])
        expected_a = pd.DataFrame(data=[[1, 10, 100],
                                        [2, 20, 100],
                                        [4, 30, 200]],
                                  columns=["ID", "a", "b"])
        pd.testing.assert_frame_equal(trimmed_a.reset_index(drop=True),
                                      expected_a.reset_index(drop=True))

        # Rows are duplicates if they have matching value in column b
        # Keeps the *first* duplicate
        trimmed_b = con_dat.remove_duplicate_rows(df, ['b'])
        expected_b = pd.DataFrame(data=[[1, 10, 100],
                                        [4, 30, 200],
                                        [6, 10, 300]],
                                  columns=["ID", "a", "b"])
        pd.testing.assert_frame_equal(trimmed_b.reset_index(drop=True),
                                      expected_b.reset_index(drop=True))

        # Rows are duplicates if they have matching value in both columns a and b
        # Keeps the *first* duplicate
        trimmed_ab = con_dat.remove_duplicate_rows(df, ['a', 'b'])
        expected_ab = pd.DataFrame(data=[[1, 10, 100],
                                         [2, 20, 100],
                                         [4, 30, 200],
                                         [6, 10, 300]],
                                   columns=["ID", "a", "b"])
        pd.testing.assert_frame_equal(trimmed_ab.reset_index(drop=True),
                                      expected_ab.reset_index(drop=True))


if __name__ == '__main__':
    workspace_root = ""

    unittest.main()
