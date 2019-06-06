"""Interacts with biopython to find interactions within PDB files."""
import json

import pandas as pd
import Bio.PDB

import scripts.utils as utils


def accept_atom(atom):
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
    file to calculate these.
    """
    assert matrix.shape[0] == matrix.shape[1],\
        "Matrix is assumed to be square"

    matrix_size = matrix.shape[0]

    parser = Bio.PDB.PDBParser()
    pdb_filename = utils.get_pdb_filename(pdb_id)

    # Read in IDs file to get pdb indices of these indices
    ids_filename = utils.get_id_filename(pdb_id)
    ids_df = pd.read_csv(ids_filename, sep=" ", header=None)

    # Load the structure and label it with the pdb_id
    structure = parser.get_structure(pdb_id, pdb_filename)
    all_residues = list(structure.get_residues())

    atom_list = [atom for atom in structure.get_atoms() if accept_atom(atom)]
    neighbor_search = Bio.PDB.NeighborSearch(atom_list)

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
                                                                       structure,
                                                                       neighbor_search,
                                                                       all_residues)

            all_bound_pairs.extend(bound_pair)
            all_bound_pairs_fragmented.extend(bound_pairs_fragmented)

    return all_bound_pairs, all_bound_pairs_fragmented


def find_targets_from_pdb(cdr_indices, ids_df, bp_structure, neighbor_search, all_residues):
    """
    Finds target fragments of a given CDR.

    Args:
        cdr_indices (array): array of integer indices to the
            interaction matrix
        pdb_id (string): string of pdb_id, used to check if pymol
            has loaded object
        ids_df (pd.DataFrame): data frame indexed by the indices
            for the interaction matrix, with columns
                chain, pdb_index, one-letter amino acid code

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
    cdr_bp_ids_str = get_full_bp_id_string(cdr_residues_from_bp)

    assert cdr_resnames_from_bp == list(cdr_resnames_from_ids),\
        "Residue names from ids list and Biopython parser should match"

    bound_pairs = []
    bound_pairs_fragmented = []

    nearby_residues = find_contacting_residues_pdb(cdr_residues_from_bp,
                                                   neighbor_search)

    if nearby_residues:
        sorted_nearby_residues, sorted_nearby_residues_z = sort_bp_residues(nearby_residues,
                                                                            all_residues)
        nearby_resnames = [Bio.PDB.protein_letters_3to1[res.get_resname()]
                           for res in sorted_nearby_residues]
        target_bp_ids_str = get_full_bp_id_string(sorted_nearby_residues)

        bound_pairs = [{'cdr_resnames': "".join(cdr_resnames_from_bp),
                        'cdr_bp_id_str': cdr_bp_ids_str,
                        'target_length': len(nearby_residues),
                        'target_resnames': "".join(nearby_resnames),
                        'target_bp_id_str': target_bp_ids_str}]

        # targets_fragmented = find_contiguous_fragments(nearby_residues,
        #                                                bp_structure)
        #
        # for fragment in targets_fragmented:
        #
        #     bound_pair_fragment = {'cdr_residues': "".join(cdr_resnames_from_bp),
        #                            'cdr_chain': cdr_chain,
        #                            'cdr_pdb_indices': ",".join(map(str, cdr_resnames_from_bp)),
        #                            'target_residues': "".join([base[1] for base in fragment]),
        #                            'target_length': len(fragment),
        #                            'target_chain': fragment[0][2],
        #                            'target_pdb_indices': ",".join([base[0] for base in fragment])}
        #     bound_pairs_fragmented.append(bound_pair_fragment)

    return bound_pairs, bound_pairs_fragmented


def find_contacting_residues_pdb(cdr_residues, neighbor_search):
    """
    Finds the residues in contact with the fragment on the given chain
    between the given indices.

    Args:
        chain (string): chain where the fragment lies
        start_ind (int): residue index where fragment begins
        end_ind (int): residue index where fragment ends

    Returns:
        dict: dict containing (1) the pdb_indices as strings, (2) the one-letter
            codes for the amino acids, (3) list of the chains for each residue,
            (4) a list where each entry contains [pdb_index, residue_one-letter, chain]
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
                 if accept_atom(atom)]

    # Find ordered atoms which are neighbours of these atoms (within 3.5 Angstroms)
    radius = 3.5
    nearby_atoms = {atom for cdr_atom in cdr_atoms
                    for atom in neighbor_search.search(cdr_atom.coord, radius, 'A')
                    if accept_atom(atom)}

    # Find residues these atoms belong to
    nearby_residues = {atom.get_parent() for atom in nearby_atoms}

    extended_cdr = get_bp_nbrs(cdr_residues)
    cleaned_residues = [res
                        for res in nearby_residues
                        if res not in extended_cdr]
    print("cdr", cdr_residues)
    print("nearby", nearby_residues)
    print("cleaned", cleaned_residues)
    print("======")

    return cleaned_residues


def find_contiguous_fragments(residues, pdb_id, max_gap=1, min_fragment_length=3):
    """
    Splits a list of residues into contiguous fragments. The list should
    contain the one-letter codes for amino acids, PDB indices of residues
    and the chain IDs for each residue.

    Args:
        residues (array): Each entry should be an array of the form
            [pdb_index, residue_one-letter, chain]
        pdb_id (string): string of pdb_id, used to check if pymol
            has loaded object
        max_gap (int): Maximum number of residues allowed to be missing
            in a fragment and it still be called contiguous. E.g. if max_gap=0
            then no gaps are allowed.
            max_gap=1 would allow [8,10,11] to be contiguous
        min_fragment_length (int): Minimum number of residues in a fragment
            before it is counted as a fragment. E.g. with min_fragment_length=3
            any fragments shorter than 3 would be discarded.

    Returns:
        array: array of arrays. Each array corresponds to a contiguous fragment, and
            contains the original entries of residues i.e. each contiguous
            fragment array will contain elements of the form
            [pdb_index, residue_one-letter, chain]

    using max_gap=1, min_fragment_length=3
    [1,3,4] -> [1,2,3,4]
    [1,3] -> [1,2,3]
    [1] -> (too short)
    [1,5] -> (both too short)
    """
    # We will assume that there are no missing residues in the PDB file, so that
    #   we can rely on the indices of the residues in the list to determine
    #   whether two residues are consecutive.
    #   Print out a statement if this assumption appears to be false

    fragments = []

    assert cmd.get_object_list(selection='(all)') == [pdb_id], \
        "No PDB file loaded prior to calling find_contiguous_fragments"

    if residues:
        # Build up each fragment element by element, starting a new fragment
        #   when the next element isn't compatible with the current fragment
        #   either because there is too big a gap between residue numbers or
        #   because they are on separate chains
        current_index, _, current_chain = residues[0]
        current_index = int(current_index)

        working_fragment = [residues[0]]
        for target in residues[1:]:
            new_index, _, new_chain = target
            new_index = int(new_index)

            if new_chain == current_chain:
                assert new_index > current_index, \
                    "List of indices must be sorted {} {}".format(new_index, current_index)

            gap = (new_index - current_index) - 1
            # If the gap is bigger than allowed or the chain has changed
            #   then we must start a new fragment
            if new_chain != current_chain or gap > max_gap:
                # Add the completed fragment to the list of fragments if it is long enough
                if len(working_fragment) >= min_fragment_length:
                    fragments.append(working_fragment)
                # Start a new fragment
                working_fragment = [target]
            else:
                if gap:
                    stored.list = []
                    missing_residues_select_str = "(chain {} and resi {}-{} & n. ca)"\
                                                  .format(new_chain,
                                                          current_index + 1,
                                                          new_index - 1)
                    cmd.iterate(missing_residues_select_str,
                                "stored.list.append((resi, oneletter, chain))")
                    working_fragment.extend(stored.list)

                working_fragment.append(target)

            current_chain = new_chain
            current_index = new_index

        if len(working_fragment) >= min_fragment_length:
            fragments.append(working_fragment)

    return fragments
