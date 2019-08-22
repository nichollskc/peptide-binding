"""Tests the file construct_database"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import unittest

import pandas as pd

import peptidebinding.helper.cluster_sequences as cluster

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
    df = pd.DataFrame(df_dict)

    def test_short_clustered_split(self):
        grouped_dfs = cluster.split_dataset_clustered(self.df, [60, 20, 20], seed=42)
        self.assertEqual(len(grouped_dfs), 3)
        self.assertEqual(len(grouped_dfs[0]), 5)
        self.assertEqual(len(grouped_dfs[1]), 2)
        self.assertEqual(len(grouped_dfs[2]), 2)


if __name__ == '__main__':
    unittest.main()
