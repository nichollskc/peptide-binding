"""Tests the file construct_database"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import logging
import os
import unittest

import numpy as np
import pandas as pd

import peptidebinding.helper.construct_database as con_dat
import peptidebinding.helper.utils as utils


class Test(unittest.TestCase):
    df_dict = {'cdr_bp_id_str': {0: '[201, 202, 203, 204]', 1: '[186, 187, 188, 189]'},
               'cdr_resnames': {0: 'LRLS', 1: 'QLAA'},
               'target_bp_id_str': {0: '[44, 45, 46]', 1: '[191, 192, 193]'},
               'target_length': {0: 3, 1: 3},
               'target_resnames': {0: 'SQL', 1: 'TRR'},
               'cdr_pdb_id': {0: '1mhp', 1: '5waq'},
               'target_pdb_id': {0: '5waq', 1: '5waq'},
               'binding_observed': {0: 0, 1: 1},
               'similarity_score': {0: -2.0, 1: np.nan},
               'original_cdr_bp_id_str': {0: '[48, 49, 50, 51]', 1: np.nan},
               'original_cdr_resnames': {0: 'TAYA', 1: np.nan},
               'original_cdr_pdb_id': {0: '5waq', 1: np.nan},
               'paired': {0: "('LRLS', 'SQL')", 1: np.nan},
               'cdr_cluster_id': {0: 23, 1: 17},
               'target_cluster_id': {0: np.nan, 1: np.nan}}
    bound_pairs_df = pd.DataFrame(df_dict)

    bound_pairs_list1 = [{'pdb_id': '3cuq',
                          'cdr_resnames': 'AQMS',
                          'cdr_bp_id_str': '[0, 1, 2, 3]',
                          'target_length': 5,
                          'target_resnames': 'QLDLY',
                          'target_bp_id_str': '[5, 6, 7, 241, 257]'},
                         {'pdb_id': '3cuq',
                          'cdr_resnames': 'MSKQ',
                          'cdr_bp_id_str': '[2, 3, 4, 5]',
                          'target_length': 7,
                          'target_resnames': 'ADMFMLY',
                          'target_bp_id_str': '[0, 7, 8, 9, 238, 241, 257]'}]
    bound_pairs_list2 = [{'pdb_id': '3cuq',
                          'cdr_resnames': 'SKQL',
                          'cdr_bp_id_str': '[3, 4, 5, 6]',
                          'target_length': 5,
                          'target_resnames': 'AMFKM',
                          'target_bp_id_str': '[0, 8, 9, 10, 238]'},
                         {'pdb_id': '3cuq',
                          'cdr_resnames': 'MSKQ',
                          'cdr_bp_id_str': '[2, 3, 4, 5]',
                          'target_length': 7,
                          'target_resnames': 'ADMFMLY',
                          'target_bp_id_str': '[0, 7, 8, 9, 238, 241, 257]'},]

    def test_short_sanitise_pdb_id(self):
        # Disable logging for this test to ensure output is clean
        logging.disable(logging.CRITICAL)

        test_pdb_ids = ['3e18', '5e01', '3e02', '1e00', '23e1'] + utils.all_pdb_ids
        utils.map_float_to_str_pdb_ids = utils.get_map_float_to_str_pdb_ids(test_pdb_ids)
        pairs = [('3cuq', '3cuq'),
                 ('3e+18', '3e18'),
                 ('50.0', '5e01'),
                 ('300.0', '3e02'),
                 ('1.0', '1e00'),
                 ('e344', 'e344'),
                 ('230.0', '23e1'),
                 # This isn't in the list, but the function can figure it out
                 ('7e+17', '7e17'),
                 # Expect this to fail
                 ('7e117', '7e117'), ]

        for pair in pairs:
            sanitised = utils.sanitise_pdb_id(pair[0])
            self.assertEqual(sanitised, pair[1])

        logging.disable(logging.NOTSET)

    def test_short_construct_filenames(self):
        pdb_id = '3cuq'

        id_file = utils.get_id_filename(pdb_id)
        self.assertTrue(os.path.exists(id_file))
        self.assertTrue(id_file.endswith('.txt'))

        pdb_file = utils.get_pdb_filename(pdb_id)
        self.assertTrue(os.path.exists(pdb_file))
        self.assertTrue(pdb_file.endswith('.pdb'))

        mat_file = utils.get_matrix_filename(pdb_id)
        self.assertTrue(os.path.exists(mat_file))
        self.assertTrue(mat_file.endswith('.bmat'))

    def test_short_save_df(self):
        df_dict = {
            'cdr_bp_id_str': {1: [1, 2, 3, 4],
                              10: [8, 9, 10, 11],
                              50: [42, 43, 44, 45],
                              200: [164, 165, 166, 167],
                              1000: [186, 187, 188, 189],
                              1500: [547, 548, 549, 550],
                              2000: [127, 128, 129, 130],
                              2001: [127, 128, 129, 130],
                              2002: [128, 129, 130, 131]},
            'cdr_resnames': {1: 'QMSK',
                             10: 'MFKT',
                             50: 'GVDP',
                             200: 'IKAS',
                             1000: 'QLVE',
                             1500: 'QWKV',
                             2000: 'VRGL',
                             2001: 'VRGL',
                             2002: 'RGLT'},
        }
        utils.save_df_csv_quoted(pd.DataFrame(df_dict), '.test_short_save_df')
        utils.save_df_csv_quoted(pd.DataFrame(self.bound_pairs_df), '.test_short_save_df')

    def test_short_bound_pair_id(self):
        self.assertEqual(utils.get_bound_pair_id_from_row(self.bound_pairs_df.iloc[0, :]),
                         '1mhp-201-4__5waq-44-3__5waq-48-4')
        self.assertEqual(utils.get_bound_pair_id_from_row(self.bound_pairs_df.iloc[1, :]),
                         '5waq-186-4__5waq-191-3')

    def test_short_interpret_bound_pair_id_through_df(self):
        for row in self.bound_pairs_df.iterrows():
            bound_pair_id = utils.get_bound_pair_id_from_row(row[1])
            intepreted = utils.get_row_from_bound_pair_id(self.bound_pairs_df, bound_pair_id)
            pd.testing.assert_series_equal(intepreted.iloc[0, :], row[1])

    def test_short_intepret_bound_pair_dict(self):
        for row in self.bound_pairs_df.iterrows():
            bound_pair_id = utils.get_bound_pair_id_from_row(row[1])
            intepreted = utils.get_dict_from_bound_pair_id(bound_pair_id)
            for key, value in intepreted.items():
                self.assertEqual(row[1][key], value)

    def test_short_read_write_dfs(self):
        utils.print_targets_to_file(self.bound_pairs_list1, "tests/.tmp.bound_pairs1.csv")
        utils.print_targets_to_file(self.bound_pairs_list2, "tests/.tmp.bound_pairs2.csv")
        all_bound_pairs = con_dat.combine_bound_pairs(["tests/.tmp.bound_pairs1.csv",
                                                       "tests/.tmp.bound_pairs2.csv"])
        self.assertEqual(len(all_bound_pairs), 4)


if __name__ == '__main__':
    unittest.main()
