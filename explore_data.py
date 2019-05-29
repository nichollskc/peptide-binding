"""Performs data exploration of database."""

import seaborn as sns
import matplotlib.pyplot as plt

import construct_database as con_dat

def save_plot(filename, folder="../plots/"):
    """Saves plot to a file in the given folder."""
    full_filename = folder + filename
    plt.savefig(full_filename, bbox_inches='tight', dpi=300)

def investigate_interaction_distributions_single(pdb_id, fragment_length):
    """Finds all CDR-like fragments of the given length in the PDB entry and
    investigates distribution of various quantities related to the
    interactions with these fragments.

    Args:
        pdb_id (string): root of filename e.g. ../example_files/3cuq
        fragment_length (int): length of CDR-like fragment to search for

    Returns:
        int: proportion of possible kmers (length fragment_length) that are
            labelled as CDR-like
        array[int]: for each CDR-like fragment, number of residues labelled as
            interacting with residues in the fragment
        array[int]: for each CDR-like fragment, if we split the interacting
            residues into contiguous fragments, how many fragments do we have?
        array[int]: lengths of each contiguous fragment as described above
    """
    matrix = con_dat.read_matrix_from_file(pdb_id)
    binding_pairs = con_dat.find_all_binding_pairs_indices(matrix, fragment_length)

    # Calculate the proportion of kmers that are cdr-like
    num_cdrs = len(binding_pairs)
    num_total_kmers = matrix.shape[0] - fragment_length + 1
    proportion_cdrs = num_cdrs / num_total_kmers

    interactor_lengths = [len(bp[1][0]) for bp in binding_pairs]

    interactor_fragments = [con_dat.find_contiguous_fragments(bp[1][0],
                                                              pdb_id + "_ids.txt")
                            for bp in binding_pairs]
    num_interactor_fragments = [len(fragments) for fragments in interactor_fragments]
    sizes_interactor_fragments = [len(fragment)
                                  for fragments in interactor_fragments
                                  for fragment in fragments]

    results = {}
    results['proportion_cdrs'] = proportion_cdrs
    results['interactor_lengths'] = interactor_lengths
    results['num_interactor_fragments'] = num_interactor_fragments
    results['sizes_interactor_fragments'] = sizes_interactor_fragments
    return results

def plot_interaction_distributions_single(pdb_id, fragment_length):
    """Finds all CDR-like fragments of the given length in the PDB entry,
    investigates distribution of various quantities related to the
    interactions with these fragments and plots these distributions"""
    results = investigate_interaction_distributions_single(pdb_id, fragment_length)

    print(results['proportion_cdrs'])

    sns.distplot(results['interactor_lengths'], kde=False, norm_hist=True)
    plt.title("Size of interacting region")
    plt.xlabel("Number of residues interacting with CDR-like fragment")
    plt.ylabel("Density")
    save_plot("3cuq_interactor_lengths.png")
    plt.clf()

    sns.distplot(results['num_interactor_fragments'], kde=False, norm_hist=True)
    plt.title("Number of interacting fragments")
    plt.xlabel("Number of contiguous fragments interacting with CDR-like fragment")
    plt.ylabel("Density")
    save_plot("3cuq_num_interactor_fragments.png")
    plt.clf()

    sns.distplot(results['sizes_interactor_fragments'], kde=False, norm_hist=True)
    plt.title("Sizes of interacting fragments")
    plt.xlabel("Size of contiguous fragments interacting with CDR-like fragment")
    plt.ylabel("Density")
    save_plot("3cuq_sizes_interactor_fragments.png")
    plt.clf()

if __name__ == "__main__":
    plot_interaction_distributions_single("../example_files/3cuq", 4)
