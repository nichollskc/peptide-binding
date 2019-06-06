"""Interacts with biopython to find interactions within PDB files."""
import json

import pandas as pd
import Bio.PDB

import scripts.utils as utils


def atom_is_ordered(atom):
    """Helper function for finding disordered atoms in a PDB structure."""
    return not atom.is_disordered() or atom.get_altloc() == 'A'


def get_full_bp_id_string(residues):
    """Take a list of Bio.PDB residue objects and return a string that contains
    all the information needed to retrieve these objects from the structure
    using select_residues_from_bp_id_string."""
    res_dicts = [{'chain': res.get_parent().get_id(), 'id': res.get_id()}
                 for res in residues]

    return json.dumps(res_dicts)


def select_residues_from_bp_id_string(bp_id_string, structure):
    """Given a loaded Bio.PDB structure, select the residues specified by the
    string, which should have been produced by get_full_bp_id_string."""
    res_dicts = json.loads(bp_id_string)

    residues = [structure[0][res_dict['chain']].child_dict[tuple(res_dict['id'])]
                for res_dict in res_dicts]

    return residues


def get_compact_bp_id_string(residues):
    """Take a list of Bio.PDB residue objects and return a string that contains
    all the information needed to retrieve these objects from the structure
    using select_residues_from_compact_bp_id_string.

    Not as robust as get_full_bp_id_string as it assumes that the residues
    will be listed in the same order when we read in the PDB file again. This
    should be true, but no checks are made."""
    if residues:
        all_residues = list(residues[0].get_parent().get_parent().get_residues())
        indices = [all_residues.index(res) for res in residues]
    else:
        indices = []

    return json.dumps(indices)


def select_residues_from_compact_bp_id_string(bp_id_string, structure):
    """Given a loaded Bio.PDB structure, select the residues specified by the
    string, which should have been produced by get_compact_bp_id_string."""
    indices = json.loads(bp_id_string)

    all_residues = list(structure[0].get_residues())
    residues = [all_residues[ind] for ind in indices]

    return residues


def sort_bp_residues(bp_residues, all_residues):
    """Zips together the residues list with their indices in the full list of
    residues and returns this list, sorted by the indices."""
    bp_residue_indices = [all_residues.index(res) for res in bp_residues]

    zipped = zip(bp_residue_indices, bp_residues)
    sorted_residues_zipped = sorted(zipped, key=lambda pair: pair[0])
    sorted_residues = [pair[1] for pair in sorted_residues_zipped]

    return sorted_residues, sorted_residues_zipped


def get_bp_nbrs(residues, distance=1):
    """Finds all residues that are within `distance` along the respective chains
    of the given residues.
    For example, given a list of one residue structure[0]["A"].child_list[i]
    and using distance=1 it will return the list
        structure[0]["A"].child_list[i-1]
        structure[0]["A"].child_list[i]
        structure[0]["A"].child_list[i+1]"""
    res_nbrs = set()
    for residue in residues:
        chain_list = residue.get_parent().child_list
        index = chain_list.index(residue)
        min_index = max(0, index - distance)
        max_index = min(len(chain_list), index + distance + 1)

        nbrs = chain_list[min_index:max_index]
        res_nbrs.update(nbrs)

    return list(res_nbrs)


def load_structure_for_search(pdb_id):
    """Loads structure using Biopython, extracts the list of residues and sets
    up the tree used to search for neighbouring atoms."""
    # Load the structure and label it with the pdb_id
    structure = Bio.PDB.PDBParser().get_structure(pdb_id,
                                                  utils.get_pdb_filename(pdb_id))
    all_residues = list(structure.get_residues())

    atom_list = [atom for atom in structure.get_atoms() if atom_is_ordered(atom)]
    neighbor_search = Bio.PDB.NeighborSearch(atom_list)

    return all_residues, neighbor_search


def construct_bound_pair_dict(cdr_residues, target_residues):
    """Constructs a dictionary of the bound pair consisting of a CDR fragment
    and its target. Many can be saved together in a csv file e.g. using
    construct_database.print_targets_to_file."""
    cdr_resnames = [Bio.PDB.protein_letters_3to1[res.get_resname()]
                    for res in cdr_residues]
    cdr_bp_ids_str = get_compact_bp_id_string(cdr_residues)

    target_resnames = [Bio.PDB.protein_letters_3to1[res.get_resname()]
                       for res in target_residues]
    target_bp_ids_str = get_compact_bp_id_string(target_residues)

    bound_pairs = [{'cdr_resnames': "".join(cdr_resnames),
                    'cdr_bp_id_str': cdr_bp_ids_str,
                    'target_length': len(target_residues),
                    'target_resnames': "".join(target_resnames),
                    'target_bp_id_str': target_bp_ids_str}]
    return bound_pairs


def find_missing_residues(residue_1, residue_2):
    """Returns a list of any residues that appear in the chain between these
    two residues."""
    chain_obj_1 = residue_1.get_parent()
    chain_obj_2 = residue_2.get_parent()
    assert chain_obj_1 == chain_obj_2,\
        "Residues must be on the same chain"

    min_index = chain_obj_1.child_list.index(residue_1) + 1
    max_index = chain_obj_1.child_list.index(residue_2)
    assert min_index < max_index,\
        "residue_1 must come (strictly) earlier in the chain than residue_2"

    missing_residues = [chain_obj_1.child_list[ind]
                        for ind in range(min_index, max_index)]
    return missing_residues


def find_all_binding_pairs(matrix, pdb_id, fragment_length):
    """
    Finds all CDR-like regions of given length in the matrix, and also finds
    the residues the CDR-like regions interact by looking at PDB file.

    Args:
        matrix (np.array): Interaction matrix where rows and columns both
            correspond to residues. Full description below.
        pdb_id (string): ID of protein in PDB e.g. '2xzz'
        fragment_length (int): length of desired interacting pairs

    Returns:
        array[array[array]]: Each array corresponds to a binding pair and
            contains two arrays: the first is an array of indices in the
            CDR-like fragment and the second is an array of indices that
            interact with those indices.
            The CDR-like fragment will have length precisely
            fragment_length, but the target indices may not be this length.

    The interaction matrix tells us which fragments are CDR-like and also which
    residues interact. If M_{x,y} is negative then the residue x interacts with
    the residue y. Additionally, if M_{x,y} < -1 or M_{x,y} > 0 then the
    fragment x, x+1, ..., y is a CDR-like region. The magnitude indicates how
    many similar fragments were found in CDRs.

    We ignore the interactions described in the matrix, and instead use the PDB
    file to find interactions.
    """
    assert matrix.shape[0] == matrix.shape[1],\
        "Matrix is assumed to be square"

    matrix_size = matrix.shape[0]

    # Read in IDs file to get pdb indices of these indices
    ids_df = pd.read_csv(utils.get_id_filename(pdb_id),
                         sep=" ",
                         header=None)

    all_residues, neighbor_search = load_structure_for_search(pdb_id)

    assert matrix_size == len(all_residues),\
        "Biopython should return the same number of residues as are listed by the matrix"

    all_bound_pairs = []
    all_bound_pairs_fragmented = []

    for start_index in range(0, matrix_size - fragment_length):
        end_index = start_index + fragment_length - 1
        matrix_entry = matrix[start_index][end_index]

        # First check if this fragment is a CDR-like fragment i.e. has it been
        #    observed in a CDR.
        if matrix_entry > 0 or matrix_entry < -1:
            # Get the indices belonging to this fragment - note range() excludes
            #   second number given
            cdr_indices = list(range(start_index, end_index + 1))
            bound_pair, bound_pairs_fragmented = find_targets_from_pdb(cdr_indices,
                                                                       ids_df,
                                                                       neighbor_search,
                                                                       all_residues)

            all_bound_pairs.extend(bound_pair)
            all_bound_pairs_fragmented.extend(bound_pairs_fragmented)

    return all_bound_pairs, all_bound_pairs_fragmented


def find_targets_from_pdb(cdr_indices, ids_df, neighbor_search, all_residues):
    """
    Finds target fragments of a given CDR.

    Args:
        cdr_indices (array): array of integer indices to the
            interaction matrix
        ids_df (pd.DataFrame): data frame indexed by the indices
            for the interaction matrix, with columns
                chain, pdb_index, one-letter amino acid code
        neighbor_search (Bio.PDB.NeighborSearch): object that allows fast
            neighbor searching - can be set up using load_structure_for_search
        all_residues (array): array of all the Bio.PDB.Residue objects that are
            in the structure for this PDB file

    Returns:
        array: (array of dicts, usually 1), each containing information about
            the whole CDR fragment and the whole interacting region
        array: (array of dicts, usually many), each containing information about
            the whole CDR fragment and an interacting fragment
    """
    cdr_resnames_from_ids = [ids_df.loc[index, 2] for index in cdr_indices]

    cdr_residues_from_bp = [all_residues[index] for index in cdr_indices]
    cdr_resnames_from_bp = [Bio.PDB.protein_letters_3to1[res.get_resname()]
                            for res in cdr_residues_from_bp]

    assert cdr_resnames_from_bp == list(cdr_resnames_from_ids),\
        "Residue names from ids list and Biopython parser should match"

    bound_pairs = []
    bound_pairs_fragmented = []

    nearby_residues = find_contacting_residues_pdb(cdr_residues_from_bp,
                                                   neighbor_search)

    if nearby_residues:
        sorted_nearby_residues, sorted_nearby_residues_z = sort_bp_residues(nearby_residues,
                                                                            all_residues)
        bound_pairs.append(construct_bound_pair_dict(cdr_residues_from_bp,
                                                     sorted_nearby_residues))

        targets_fragmented = find_contiguous_fragments(sorted_nearby_residues_z)

        for fragment in targets_fragmented:
            bound_pairs_fragmented.append(construct_bound_pair_dict(cdr_residues_from_bp,
                                                                    fragment))

    return bound_pairs, bound_pairs_fragmented


def find_contacting_residues_pdb(cdr_residues, neighbor_search):
    """
    Finds the residues in contact with the fragment on the given chain
    between the given indices.

    Args:
        cdr_residues (array): array of Bio.PDB.Residue objects
        neighbor_search (Bio.PDB.NeighborSearch): object that allows fast
            neighbor searching - can be set up using load_structure_for_search

    Returns:
        array: array containing Bio.PDB.Residue objects corresponding to the residues
            in contact with the CDR residues
    """
    # Some PDB files contain multiple locations for each atom in certain residues.
    #   We will ignore all locations except the first (usually labelled A).
    #   Biopython represents such residues by putting the first location as one normal
    #   residue object, then adding extra 'disordered' atoms to this object. We can
    #   therefore just remove the disordered atoms to be sure we are left with a
    #   single location for each residue.

    # Get only ordered atoms of CDR
    cdr_atoms = [atom
                 for atom in Bio.PDB.Selection.unfold_entities(cdr_residues, 'A')
                 if atom_is_ordered(atom)]

    # Find ordered atoms which are neighbours of these atoms (within 3.5 Angstroms)
    radius = 3.5
    nearby_atoms = {atom for cdr_atom in cdr_atoms
                    for atom in neighbor_search.search(cdr_atom.coord, radius, 'A')
                    if atom_is_ordered(atom)}

    # Find residues these atoms belong to
    nearby_residues = {atom.get_parent() for atom in nearby_atoms}

    extended_cdr = get_bp_nbrs(cdr_residues)
    cleaned_residues = [res
                        for res in nearby_residues
                        if res not in extended_cdr]

    return cleaned_residues


def find_contiguous_fragments(residues_z, max_gap=1, min_fragment_length=3):
    """
    Splits a list of residues into contiguous fragments.

    Args:
        residues_z (array): Array of tuples, where the first in each tuple is the
            index of the residue in the list of all residues in this structure and
            the second entry in each tuple is the Bio.PDB.Residue object
        max_gap (int): Maximum number of residues allowed to be missing
            in a fragment and it still be called contiguous. E.g. if max_gap=0
            then no gaps are allowed.
            max_gap=1 would allow [8,10,11] to be contiguous
        min_fragment_length (int): Minimum number of residues in a fragment
            before it is counted as a fragment. E.g. with min_fragment_length=3
            any fragments shorter than 3 would be discarded.

    Returns:
        array: array of arrays. Each array corresponds to a contiguous fragment, and
            contains Bio.PDB.Residue objects

    using max_gap=1, min_fragment_length=3
    [1,3,4] -> [1,2,3,4]
    [1,3] -> [1,2,3]
    [1] -> (too short)
    [1,5] -> (both too short)
    """
    # We will assume that there are no missing residues in the PDB file, so that
    #   we can rely on the indices of the residues in the list to determine
    #   whether two residues are consecutive.

    fragments = []

    if residues_z:
        # Build up each fragment element by element, starting a new fragment
        #   when the next element isn't compatible with the current fragment
        #   either because there is too big a gap between residue numbers or
        #   because they are on separate chains
        # Recall that the list residues_z contains pairs (index, residue_obj)
        current_index = residues_z[0][0]
        current_residue = residues_z[0][1]
        current_chain_obj = current_residue.get_parent()

        working_fragment = [residues_z[0][1]]
        for target in residues_z[1:]:
            new_index = target[0]
            new_residue = target[1]
            new_chain_obj = new_residue.get_parent()

            if new_chain_obj == current_chain_obj:
                assert new_index > current_index, \
                    "List of indices must be sorted {} {}".format(new_index, current_index)

            gap = (new_index - current_index) - 1
            # If the gap is bigger than allowed or the chain has changed
            #   then we must start a new fragment
            if new_chain_obj != current_chain_obj or gap > max_gap:
                # Add the completed fragment to the list of fragments if it is long enough
                if len(working_fragment) >= min_fragment_length:
                    fragments.append(working_fragment)
                # Start a new fragment
                working_fragment = [new_residue]
            else:
                if gap:
                    # Select the residues strictly between these two indices
                    working_fragment.extend(find_missing_residues(current_residue,
                                                                  new_residue))

                working_fragment.append(new_residue)

            current_chain_obj = new_chain_obj
            current_index = new_index
            current_residue = new_residue

        if len(working_fragment) >= min_fragment_length:
            fragments.append(working_fragment)

    return fragments
