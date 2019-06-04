"""Tests the file query_pymol.py"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import os
import unittest

from pymol import cmd

import query_pymol


class Test(unittest.TestCase):

    def test_contiguous_fragments(self):
        cmd.reinitialize()
        cmd.load(query_pymol.get_pdb_filename("3cuq", workspace_root))
        residues = [('119', 'I', 'A'),
                    ('121', 'T', 'A'),
                    ('127', 'Q', 'A'),
                    ('128', 'V', 'A'),
                    ('129', 'L', 'A'),
                    ('139', 'V', 'A')]

        expected_03 = [[('127', 'Q', 'A'),
                        ('128', 'V', 'A'),
                        ('129', 'L', 'A')]]
        fragments_03 = query_pymol.find_contiguous_fragments(residues,
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
        fragments_13 = query_pymol.find_contiguous_fragments(residues,
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
        fragments_11 = query_pymol.find_contiguous_fragments(residues,
                                                             "3cuq",
                                                             max_gap=1,
                                                             min_fragment_length=1)
        self.assertEqual(fragments_11, expected_11)


if __name__ == '__main__':
    workspace_root = ""

    unittest.main()
