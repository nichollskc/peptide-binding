"""Tests the file query_pymol.py"""
# pylint: disable=missing-docstring
# pylint: disable=no-self-use

import unittest

import Bio.PDB

import scripts.query_biopython as query_bp


class Test(unittest.TestCase):

    def test_short_round_test_compact_bp_id(self):
        parser = Bio.PDB.PDBParser()
        structure = parser.get_structure("3cuq", "cleanPDBs2/3cuq.pdb")

        all_residues = list(structure[0].get_residues())
        block = all_residues[2:5]

        bp_id_string = query_bp.get_compact_bp_id_string(block)

        result = query_bp.select_residues_from_compact_bp_id_string(bp_id_string, structure)

        self.assertEqual(result, block)

    def test_short_round_test_bp_id(self):
        parser = Bio.PDB.PDBParser()
        structure = parser.get_structure("3cuq", "cleanPDBs2/3cuq.pdb")

        all_residues = list(structure[0].get_residues())
        block = all_residues[2:5]

        bp_id_string = query_bp.get_full_bp_id_string(block)

        result = query_bp.select_residues_from_bp_id_string(bp_id_string, structure)

        self.assertEqual(result, block)

    # pylint: disable=too-many-locals
    def test_short_contiguous_fragments(self):
        parser = Bio.PDB.PDBParser()
        structure = parser.get_structure("3cuq", "cleanPDBs2/3cuq.pdb")

        all_residues = list(structure[0].get_residues())
        block_1 = all_residues[2:5]
        block_2 = all_residues[6:8]
        filled_block = all_residues[2:8]

        block_3 = all_residues[216:219]
        block_4 = all_residues[219:223]
        block_5 = [all_residues[460]]
        block_6 = [all_residues[633]]
        raw_residues = block_1 + block_2 + block_3 + block_4 + block_5 + block_6

        unused_sorted_res, sorted_res_z = query_bp.sort_bp_residues(raw_residues,
                                                                    all_residues)

        expected_03 = [block_1, block_3, block_4]
        fragments_03 = query_bp.find_contiguous_fragments(sorted_res_z,
                                                          max_gap=0,
                                                          min_fragment_length=3)
        self.assertEqual(fragments_03, expected_03)

        expected_13 = [filled_block, block_3, block_4]
        fragments_13 = query_bp.find_contiguous_fragments(sorted_res_z,
                                                          max_gap=1,
                                                          min_fragment_length=3)
        self.assertEqual(fragments_13, expected_13)

        expected_11 = [filled_block, block_3, block_4, block_5, block_6]
        fragments_11 = query_bp.find_contiguous_fragments(sorted_res_z,
                                                          max_gap=1,
                                                          min_fragment_length=1)
        self.assertEqual(fragments_11, expected_11)


if __name__ == '__main__':

    unittest.main()
