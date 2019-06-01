"""Constructs database of interacting fragments."""

import csv

import numpy as np
import pandas

MATRIX_DIR = "/sharedscratch/kcn25/icMatrix/"
IDS_DIR = "/sharedscratch/kcn25/IDs/"
PDB_DIR = "/sharedscratch/kcn25/cleanPDBs2/"

def get_id_filename(pdb_id):
    """Given the pdb id, return the full filename for the IDs file."""
    return IDS_DIR + pdb_id + "_ids.txt"

def get_matrix_filename(pdb_id):
    """Given the pdb id, return the full filename for the matrix file."""
    return MATRIX_DIR + pdb_id + "_icMat.bmat"

def get_pdb_filename(pdb_id):
    """Given the pdb id, return the full filename for the PDB file."""
    return PDB_DIR + pdb_id + ".pdb"

def read_matrix_from_file(pdb_id):
    """Reads interaction matrix from file and return as np.array.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".

    Returns:
        np.array: interaction matrix as an np.array
    """
    ids_filename = get_id_filename(pdb_id)
    matrix_filename = get_matrix_filename(pdb_id)

    # Read in residue IDs
    ids = pandas.read_csv(ids_filename, sep=" ", header=None)
    num_residues = ids.shape[0]

    # Read in binary matrix
    with open(matrix_filename, 'rb') as f:
        raw = np.fromfile(f, np.int32)

    # Found dimensions from corresponding ids.txt file
    matrix = raw.reshape((num_residues, num_residues))

    return matrix

def read_matrix_from_file_df(pdb_id):
    """Read interaction matrix from file, label using IDs file and return as a
    data frame.

    Args:
        pdb_id (string): string of PDB ID e.g. "2zxx".

    Returns:
        pandas.DataFrame: data frame containing the matrix, with labels given by
            the rows of the IDs file
    """
    matrix = read_matrix_from_file(pdb_id)
    ids_filename = get_matrix_filename(pdb_id)

    # Combine the three columns into one label for each residue
    ids = pandas.read_csv(ids_filename, sep=" ", header=None)
    combined_labels = ids.apply(lambda x: '_'.join(x.map(str)), axis=1)

    df = pandas.DataFrame(matrix, index=combined_labels, columns=combined_labels)
    return df

def find_contiguous_fragments(indices, ids_filename, max_gap=3):
    """Decomposes a sorted set of indices into smaller sets which are
    contiguous, based on the residue numbers given in the ids_filename.
    E.g. [1,2,5,100,101,102,200] becomes [[1,2,5], [100,101,102], [200]].
    Allow gaps of up to max_gap for a fragment to count as contiguous.

    Args:
        indices (array): array of indices to split into fragments
        ids_filename (string): name of file listing residues
        max_gap (int): maximum number of missing residues in a contiguous
            fragment

    Returns:
        array[array]: array of fragments, where each fragment is given by
            an array of indices
    """
    fragments = []

    ids = pandas.read_csv(ids_filename, sep=" ", header=None)

    if indices:
        # Build up each fragment element by element, starting a new fragment
        #   when the next element isn't compatible with the current fragment
        #   either because there is too big a gap between residue numbers or
        #   because they are on separate chains
        current_index = indices[0]
        current_chain = ids.loc[current_index, 0]
        current_residue = ids.loc[current_index, 1]

        working_fragment = [current_index]
        for new_index in indices[1:]:
            new_chain = ids.loc[new_index, 0]
            new_residue = ids.loc[new_index, 1]

            assert new_index > current_index, "List of indices must be sorted"

            # If the gap is bigger than allowed or the chain has changed
            #   then we must start a new fragment
            if new_chain != current_chain or (new_residue - current_residue) > max_gap:
                # Add the completed fragment to the list of fragments
                fragments.append(working_fragment)
                # Start a new fragment
                working_fragment = [new_index]
            else:
                working_fragment.append(new_index)

            current_residue = new_residue
            current_chain = new_chain
            current_index = new_index
        fragments.append(working_fragment)

    return fragments

def find_interactor_indices(matrix, cdr_indices):
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
    interactor_indices_np = ((matrix[cdr_indices, :] < 0).sum(axis=0) > 0).nonzero()
    interactor_indices = list(interactor_indices_np[0])

    for index in cdr_indices:
        if index in interactor_indices:
            interactor_indices.remove(index)

    return interactor_indices

def find_all_binding_pairs_indices(matrix, fragment_length):
    """
    Finds all CDR-like regions of given length in the matrix, and also finds
    the residues the CDR-like regions interact with in this matrix.

    Args:
        matrix (np.array): Interaction matrix where rows and columns both
            correspond to residues. Full description below.
        fragment_length (int): length of desired interacting pairs

    Returns:
        array[array[array]]: Each array corresponds to a binding pair and
            contains two arrays: the first is an array of indices in the
            CDR-like fragment and the second is an array of indices that
            interact with those indices.
            The CDR-like fragment will contain have length precisely
            fragment_length, but the interacting indices may not be this length.

    The interaction matrix tells us which fragments are CDR-like and also which
    residues interact. If M_{x,y} is negative then the residue x interacts with
    the residue y. Additionally, if M_{x,y} < -1 or M_{x,y} > 0 then the
    fragment x, x+1, ..., y is a CDR-like region. The magnitude indicates how
    many similar fragments were found in CDRs.
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix is assumed to be square"

    matrix_size = matrix.shape[0]
    binding_pairs = []

    for start_index in range(0, matrix_size - fragment_length):
        end_index = start_index + fragment_length - 1
        matrix_entry = matrix[start_index][end_index]

        # First check if this fragment is a CDR-like fragment i.e. has it been
        #    observed in a CDR.
        if  matrix_entry > 0 or matrix_entry < -1:
            # Get the indices belonging to this fragment - note range() excludes
            #   second number given
            cdr_indices = list(range(start_index, end_index + 1))
            interactor_indices = find_interactor_indices(matrix, cdr_indices)

            binding_pair = [cdr_indices, interactor_indices]
            binding_pairs.append(binding_pair)

    return binding_pairs

# Disable pylint warning about too many local variables for this function
#pylint: disable-msg=too-many-locals
def process_database_single(pdb_id, fragment_length, do_fragment_target):
    """Finds all CDR-like fragments of the given length in the interaction
    matrix for the given pdb_id. Additionally the residues these fragments
    interact with.

    Writes the following out to file for each fragment found:
    cdr indices, chain, indices in PDB file, residue names
    same for the whole region of interacting residues

    If do_fragment_target is True, then also finds contiguous fragments in the
    interacting residues and for each contiguous fragment, writes the following
    out to file:
    cdr indices, chain, indices in PDB file, residue names
    same for fragment of interacting residues
    """
    matrix = read_matrix_from_file(pdb_id)
    bind_pairs = find_all_binding_pairs_indices(matrix, fragment_length)

    bound_pairs_all = [["cdr_indices",
                        "cdr_chain",
                        "cdr_pdb_indices",
                        "cdr_residues",
                        "interacting_indices",
                        "interacting_chain",
                        "interacting_pdb_indices"
                        "interacting_residues"]]
    bound_pairs_fragmented = [["cdr_indices",
                               "cdr_chain",
                               "cdr_pdb_indices",
                               "cdr_residues",
                               "interacting_indices",
                               "interacting_chain",
                               "interacting_pdb_indices"
                               "interacting_residues"]]

    ids_filename = get_id_filename(pdb_id)
    ids = pandas.read_csv(ids_filename, sep=" ", header=None)

    for bp in bind_pairs:
        # Write out cdr fragment, interacting indices, pdb_id
        cdr_indices = bp[0]
        cdr_indices_str = ",".join(map(str, cdr_indices))
        cdr_residues = [ids.loc[index, 2] for index in cdr_indices]
        cdr_residues_str = "".join(cdr_residues)
        cdr_pdb_indices = [ids.loc[index, 1] for index in cdr_indices]
        cdr_pdb_indices_str = ",".join(map(str, cdr_pdb_indices))
        cdr_chain = ids.loc[cdr_indices[0], 0]

        interacting_indices = bp[1]
        interacting_indices_str = ",".join(map(str, interacting_indices))
        interacting_residues = [ids.loc[index, 2] for index in interacting_indices]
        interacting_residues_str = "".join(interacting_residues)
        interacting_pdb_indices = [ids.loc[index, 1] for index in interacting_indices]
        interacting_pdb_indices_str = ",".join(map(str, interacting_pdb_indices))
        interacting_chain = ids.loc[interacting_indices[0], 0]

        bound_pairs_all.append([cdr_indices_str,
                                cdr_chain,
                                cdr_pdb_indices_str,
                                cdr_residues_str,
                                interacting_indices_str,
                                interacting_chain,
                                interacting_pdb_indices_str,
                                interacting_residues_str])

        if do_fragment_target:
            interacting_fragments = find_contiguous_fragments(interacting_indices,
                                                              ids_filename,
                                                              max_gap=2)

            for interacting_fragment in interacting_fragments:
                interacting_fragment_str = ",".join(map(str, interacting_fragment))
                interacting_fragment_residues = [ids.loc[index, 2]
                                                 for index in interacting_fragment]
                interacting_fragment_residues_str = "".join(interacting_fragment_residues)
                interacting_fragment_pdb_indices = [ids.loc[index, 1]
                                                    for index in interacting_fragment]
                interacting_fragment_pdb_indices_str = ",".join(map(str,
                                                                    interacting_fragment_pdb_indices))
                interacting_fragment_chain = ids.loc[interacting_fragment[0], 0]

                bound_pairs_fragmented.append([cdr_indices_str,
                                               cdr_chain,
                                               cdr_pdb_indices_str,
                                               cdr_residues_str,
                                               interacting_fragment_str,
                                               interacting_fragment_chain,
                                               interacting_fragment_pdb_indices_str,
                                               interacting_fragment_residues_str])

    all_residues_filename = ("/sharedscratch/kcn25/fragment_database/" +
                             pdb_id +
                             "bound_pairs_all.csv")
    with open(all_residues_filename, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(bound_pairs_all)

    if do_fragment_target:
        fragmented_residues_filename = ("/sharedscratch/kcn25/fragment_database/" +
                                        pdb_id +
                                        "bound_pairs_fragmented.csv")
        with open(fragmented_residues_filename, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerows(bound_pairs_fragmented)

def process_database(ids_list, fragment_length, do_fragment_target):
    """Finds all CDR-like fragments of the given length in the files listed in
    the array ids_list. Additionally finds the residues these fragments interact
    with. Outputs all finds to files, with separate files for each pdb_id.
    """
    for pdb_id in ids_list:
        process_database_single(pdb_id, fragment_length, do_fragment_target)

if __name__ == "__main__":
    # Generate random order using `ls /sharedscratch/kcn25/icMatrix/ |sort -R > random_order.txt`
    with open("random_order.txt") as filelist:
        random_matrix_files = filelist.readlines()

    random_ids = [filename.split("_")[0] for filename in random_matrix_files[:1000]]

    process_database(random_ids, fragment_length=4, do_fragment_target=False)
