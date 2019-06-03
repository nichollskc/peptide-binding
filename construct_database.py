"""Constructs database of interacting fragments."""

import csv
import glob
import os

import numpy as np
import pandas as pd
from pymol import cmd, stored       # pylint: disable-msg=no-name-in-module


def get_id_filename(pdb_id, workspace_root):
    """Given the pdb id, return the full filename for the IDs file."""
    return os.path.join(workspace_root, "IDs/", pdb_id + "_ids.txt")


def get_matrix_filename(pdb_id, workspace_root):
    """Given the pdb id, return the full filename for the matrix file."""
    return os.path.join(workspace_root, "icMatrix/", pdb_id + "_icMat.bmat")


def get_pdb_filename(pdb_id, workspace_root):
    """Given the pdb id, return the full filename for the PDB file."""
    return os.path.join(workspace_root, "cleanPDBs2/", pdb_id + ".pdb")


def read_matrix_from_file(pdb_id, workspace_root):
    """Reads interaction matrix from file and return as np.array.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".
        workspace_root (string): location of workspace for reading and
            storing files

    Returns:
        np.array: interaction matrix as an np.array
    """
    ids_filename = get_id_filename(pdb_id, workspace_root)
    matrix_filename = get_matrix_filename(pdb_id, workspace_root)

    # Read in residue IDs
    ids = pd.read_csv(ids_filename, sep=" ", header=None)
    num_residues = ids.shape[0]

    # Read in binary matrix
    with open(matrix_filename, 'rb') as f:
        raw = np.fromfile(f, np.int32)

    # Found dimensions from corresponding ids.txt file
    matrix = raw.reshape((num_residues, num_residues))

    return matrix


def read_matrix_from_file_df(pdb_id, workspace_root):
    """Read interaction matrix from file, label using IDs file and return as a
    data frame.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".
        workspace_root (string): location of workspace for reading and
            storing files

    Returns:
        pd.DataFrame: data frame containing the matrix, with labels given by
            the rows of the IDs file
    """
    matrix = read_matrix_from_file(pdb_id, workspace_root)
    ids_filename = get_id_filename(pdb_id, workspace_root)

    # Combine the three columns into one label for each residue
    ids = pd.read_csv(ids_filename, sep=" ", header=None)
    combined_labels = ids.apply(lambda x: '_'.join(x.map(str)), axis=1)

    df = pd.DataFrame(matrix, index=combined_labels, columns=combined_labels)
    return df


def find_contiguous_fragments(residues, max_gap=1, min_fragment_length=3):
    """
    Splits a list of residues into contiguous fragments. The list should
    contain the one-letter codes for amino acids, PDB indices of residues
    and the chain IDs for each residue.

    Args:
        residues (array): Each entry should be an array of the form
            [pdb_index, residue_one-letter, chain]
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
    fragments = []

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
            if new_chain != current_chain or gap >= max_gap:
                # Add the completed fragment to the list of fragments if it is long enough
                if len(working_fragment) >= min_fragment_length:
                    fragments.append(working_fragment)
                # Start a new fragment
                working_fragment = [target]
            else:
                # TODO: If there is a gap, find the residues that should go in the gap
                working_fragment.append(target)

            current_chain = new_chain
            current_index = new_index

        if len(working_fragment) >= min_fragment_length:
            fragments.append(working_fragment)

    return fragments


def find_contacting_residues_pdb(chain, start_ind, end_ind):
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
    # Select the atoms in the CDR-like fragment
    cdr_select_string = "chain {} and resi {}-{}".format(chain, start_ind, end_ind)
    cmd.select("cdr", cdr_select_string)

    # Select all atoms within 3.5 Angstroms of the CDR-like atoms, excluding
    #   residues on either side of the fragment
    contacts_select_string = "(cdr around 3.5)"
    contacts_select_string += "and (not resi {}) and (not resi {})".format(start_ind - 1,
                                                                           end_ind + 1)
    cmd.select("contact_atoms", contacts_select_string)

    # Expand the selection to whole residues (byres)
    cmd.select("contact_residues", "byres contact_atoms")

    # Select only the alpha carbons of these residues and store in a list
    stored.list = []
    cmd.iterate("(contact_residues & n. ca)", "stored.list.append((resi, oneletter, chain))")

    # Split results into indices and residue codes, converting the three letter
    #   codes to one letter codes
    pdb_indices = [target[0] for target in stored.list]
    residues = [target[1] for target in stored.list]
    chains = [target[2] for target in stored.list]

    targets = {'pdb_indices': pdb_indices,
               'residues': residues,
               'chains': chains,
               'combined_list': stored.list}

    return targets


def find_targets_from_pdb(cdr_indices, pdb_id, ids_df):
    """
    Finds target fragments of a given CDR. The pdb file must have
    been loaded in pymol before calling this function.

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
    cdr_residues = [ids_df.loc[index, 2] for index in cdr_indices]
    cdr_pdb_indices = [ids_df.loc[index, 1] for index in cdr_indices]
    cdr_chain = ids_df.loc[cdr_indices[0], 0]

    start_ind = cdr_pdb_indices[0]
    end_ind = cdr_pdb_indices[-1]

    # Check that the object is already loaded, and nothing else is loaded
    assert cmd.get_object_list(selection='(all)') == [pdb_id], \
        "No PDB file loaded prior to calling find_targets_from_pdb"

    targets = find_contacting_residues_pdb(cdr_chain, start_ind, end_ind)

    if targets['combined_list']:
        bound_pairs = [{'cdr_residues': "".join(cdr_residues),
                        'cdr_chain': cdr_chain,
                        'cdr_pdb_indices': ",".join(map(str, cdr_pdb_indices)),
                        'target_residues': "".join(targets['residues']),
                        'target_length': len(targets['residues']),
                        'target_chain': targets['chains'][0],
                        'target_pdb_indices': ",".join(targets['pdb_indices'])}]
    else:
        bound_pairs = []

    targets_fragmented = find_contiguous_fragments(targets['combined_list'])

    bound_pairs_fragmented = []
    for fragment in targets_fragmented:
        bound_pair_fragment = {'cdr_residues': "".join(cdr_residues),
                               'cdr_chain': cdr_chain,
                               'cdr_pdb_indices': ",".join(map(str, cdr_pdb_indices)),
                               'target_residues': "".join([base[1] for base in fragment]),
                               'target_length': len(fragment),
                               'target_chain': fragment[0][2],
                               'target_pdb_indices': ",".join([base[0] for base in fragment])}
        bound_pairs_fragmented.append(bound_pair_fragment)

    return bound_pairs, bound_pairs_fragmented


def find_target_indices_from_matrix(matrix, cdr_indices):
    """
    Finds all indices that interact with the given CDR according to the matrix.

    Args:
        matrix (np.array): square matrix giving interactions between residues,
            described further in find_all_binding_pairs_indices.
        cdr_indices (array): array of indices for which we want to find
            interactions.

    Returns:
        array: array of indices that interact with any of the indices of CDR.
    """
    target_indices_np = ((matrix.iloc[cdr_indices, :] < 0).sum(axis=0) > 0).nonzero()
    target_indices = list(target_indices_np[0])

    for index in cdr_indices:
        if index in target_indices:
            target_indices.remove(index)

    return target_indices


def find_all_binding_pairs(matrix, pdb_id, workspace_root, fragment_length):
    """
    Finds all CDR-like regions of given length in the matrix, and also finds
    the residues the CDR-like regions interact by looking at PDB file.

    Args:
        matrix (np.array): Interaction matrix where rows and columns both
            correspond to residues. Full description below.
        pdb_id (string): ID of protein in PDB e.g. '2xzz'
        workspace_root (string): location of workspace for reading and storing files
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
    assert matrix.shape[0] == matrix.shape[1], "Matrix is assumed to be square"

    matrix_size = matrix.shape[0]

    # Read in IDs file to get pdb indices of these indices
    ids_df = pd.read_csv(get_id_filename(pdb_id, workspace_root), sep=" ", header=None)

    pdb_filename = get_pdb_filename(pdb_id, workspace_root)
    cmd.reinitialize()
    cmd.load(pdb_filename)

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
                                                                       pdb_id,
                                                                       ids_df)

            all_bound_pairs.extend(bound_pair)
            all_bound_pairs_fragmented.extend(bound_pairs_fragmented)

    return all_bound_pairs, all_bound_pairs_fragmented


def print_targets_to_file(bound_pairs, filename):
    """Prints bound pairs to csv file."""
    rows = [["cdr_residues",
             "cdr_chain",
             "cdr_pdb_indices",
             "target_residues",
             "target_length",
             "target_chain",
             "target_pdb_indices"]]

    for target in bound_pairs:
        rows.append([target['cdr_residues'],
                     target['cdr_chain'],
                     target['cdr_pdb_indices'],
                     target['target_residues'],
                     target['target_length'],
                     target['target_chain'],
                     target['target_pdb_indices']])

    with open(filename, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)


def process_database_single(pdb_id, workspace_root, fragment_length):
    """Finds all CDR-like fragments of the given length in the interaction
    matrix for the given pdb_id. Additionally the residues these fragments
    interact with.

    Finds contiguous fragments in the target residues for each CDR and for
    each contiguous fragment, writes the following out to file:
    cdr indices, chain, indices in PDB file, residue names
    same for fragment of target residues

    workspace_root (string): location of workspace for reading and
            storing files

    """
    matrix = read_matrix_from_file(pdb_id, workspace_root)
    bound_pairs, bound_pairs_fragmented = find_all_binding_pairs(matrix,
                                                                 pdb_id,
                                                                 workspace_root,
                                                                 fragment_length)
    directory = os.path.join(workspace_root, "fragment_database/bound_pairs")
    print_targets_to_file(bound_pairs,
                          os.path.join(directory, pdb_id + "_bound_pairs_all.csv"))
    print_targets_to_file(bound_pairs_fragmented,
                          os.path.join(directory, pdb_id + "_bound_pairs_fragmented.csv"))


def read_bound_pairs(filename):
    """Read a csv file containing bound pairs and return the csv"""
    bound_pairs_df = pd.read_csv(filename, header=0, index_col=None)

    return bound_pairs_df


def combine_bound_pairs(filename_list):
    """Read in all the bound pairs from the csv files in `filename_list` and
    combine them into a single dataframe"""
    data_frames = [read_bound_pairs(filename) for filename in filename_list]

    combined_data_frame = pd.concat(data_frames)

    return combined_data_frame


def combine_all_bound_pairs_fragmented():
    """Read in all files containing bound pairs where the target has been
    fragmented into contiguous fragments and combine into a single dataframe"""
    fragmented_files = glob.glob("/sharedscratch/kcn25/fragment_database/*frag*")
    bound_pairs_fragmented = combine_bound_pairs(fragmented_files)
    return bound_pairs_fragmented


def combine_all_bound_pairs_complete():
    """Read in all files containing bound pairs where the target is intact
    and combine all into a single dataframe"""
    complete_files = glob.glob("/sharedscratch/kcn25/fragment_database/*all*")
    bound_pairs_complete = combine_bound_pairs(complete_files)
    return bound_pairs_complete


def process_database(ids_list, workspace_root, fragment_length):
    """Finds all CDR-like fragments of the given length in the files listed in
    the array ids_list. Additionally finds the residues these fragments interact
    with. Outputs all finds to files, with separate files for each pdb_id.
    """
    for pdb_id in ids_list:
        process_database_single(pdb_id, workspace_root, fragment_length)


if __name__ == "__main__":
    # Generate random order using `ls /sharedscratch/kcn25/icMatrix/ |sort -R > random_order.txt`
    with open("random_order.txt") as filelist:
        random_matrix_files = filelist.readlines()

    random_ids = [filename.split("_")[0] for filename in random_matrix_files[:1000]]

    process_database(random_ids, "/sharedscratch/kcn25/", fragment_length=4)
