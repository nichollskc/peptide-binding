"""Tests the file construct_database"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import unittest

import numpy as np
import pandas as pd

import peptidebinding.helper.construct_database as con_dat

class Test(unittest.TestCase):
    df_dict = {
        'cdr_bp_id_str': {1: '[1, 2, 3, 4]',
                          10: '[8, 9, 10, 11]',
                          50: '[42, 43, 44, 45]',
                          200: '[164, 165, 166, 167]',
                          1000: '[186, 187, 188, 189]',
                          1500: '[547, 548, 549, 550]',
                          2000: '[127, 128, 129, 130]',
                          2001: '[127, 128, 129, 130]',
                          2002: '[128, 129, 130, 131]'},
        'cdr_resnames': {1: 'QMSK',
                         10: 'MFKT',
                         50: 'GVDP',
                         200: 'IKAS',
                         1000: 'QLVE',
                         1500: 'QWKV',
                         2000: 'VRGL',
                         2001: 'VRGL',
                         2002: 'RGLT'},
        'pdb_id': {1: '3cuq',
                   10: '3cuq',
                   50: '3cuq',
                   200: '3cuq',
                   1000: '1mhp',
                   1500: '1mhp',
                   2000: '2h5c',
                   2001: '2h5c',
                   2002: '2h5c'},
        'target_bp_id_str': {1: '[6, 7, 8]',
                             10: '[4, 5, 6]',
                             50: '[38, 39, 40]',
                             200: '[169, 170, 171, 172]',
                             1000: '[276, 277, 278, 279]',
                             1500: '[552, 553, 554, 555]',
                             2000: '[116, 117, 118, 119, 120, 121, 122]',
                             2001: '[179, 180, 181, 182]',
                             2002: '[60, 61, 62]'},
        'target_length': {1: 3,
                          10: 3,
                          50: 3,
                          200: 4,
                          1000: 4,
                          1500: 4,
                          2000: 7,
                          2001: 4,
                          2002: 3},
        'target_resnames': {1: 'LDM',
                            10: 'KQL',
                            50: 'CAT',
                            200: 'KWET',
                            1000: 'YYCT',
                            1500: 'NALQ',
                            2000: 'KNVTANY',
                            2001: 'LFER',
                            2002: 'GND'},
        'cdr_pdb_id': {1: '3cuq',
                       10: '3cuq',
                       50: '3cuq',
                       200: '3cuq',
                       1000: '1mhp',
                       1500: '1mhp',
                       2000: '2h5c',
                       2001: '2h5c',
                       2002: '2h5c'}
    }

    def test_find_all_bound_pairs(self):
        con_dat.find_all_bound_pairs(["3cuq", "1mhp"], fragment_length=4)

    def test_short_generate_negatives_alignment_threshold(self):
        # There are precisely 9 rows and precisely 9 pairs that have <0 similarity
        #   so this should only just work
        num_positives = len(self.df_dict['cdr_resnames'])
        combined = con_dat.generate_negatives_alignment_threshold(pd.DataFrame(self.df_dict))

        self.assertEqual(len(combined.index), 2 * num_positives)

        # Should already be free of duplicates
        no_duplicates = con_dat.remove_duplicate_rows(combined, ['cdr_resnames', 'target_resnames'])
        pd.testing.assert_frame_equal(no_duplicates, combined)

        combined = con_dat.generate_negatives_alignment_threshold(pd.DataFrame(self.df_dict), k=2)
        self.assertEqual(len(combined.index), 2 + num_positives)

        # Should already be free of duplicates
        no_duplicates = con_dat.remove_duplicate_rows(combined, ['cdr_resnames', 'target_resnames'])
        pd.testing.assert_frame_equal(no_duplicates, combined)

    def test_short_read_matrix(self):
        matrix = con_dat.read_matrix_from_file("3cuq")
        self.assertTrue(isinstance(matrix, np.ndarray), "Should return numpy array")

    def test_short_read_matrix_df(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq")
        self.assertTrue(isinstance(matrix, pd.DataFrame), "Should return data frame")

    def test_short_interactions_matrix(self):
        matrix = con_dat.read_matrix_from_file_df("3cuq")
        con_dat.find_target_indices_from_matrix(matrix, [0, 1, 2, 3])

    def test_short_remove_duplicates(self):
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
    unittest.main()
