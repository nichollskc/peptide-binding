"""Performs data exploration of database."""

import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    interactor_lengths = [len(bp[1]) for bp in binding_pairs]

    interactor_fragments = [con_dat.find_contiguous_fragments(bp[1],
                                                              con_dat.get_id_filename(pdb_id))
                            for bp in binding_pairs]
    num_interactor_fragments = [len(fragments) for fragments in interactor_fragments]
    sizes_interactor_fragments = [len(fragment)
                                  for fragments in interactor_fragments
                                  for fragment in fragments]

    results = {}
    results['proportion_cdrs'] = proportion_cdrs
    results['num_cdrs'] = num_cdrs
    results['interactor_lengths'] = interactor_lengths
    results['num_interactor_fragments'] = num_interactor_fragments
    results['sizes_interactor_fragments'] = sizes_interactor_fragments
    results['cdr_observation_counts'] = list(np.diagonal(matrix, offset=3))
    return results

def plot_interaction_distributions_single(pdb_id, fragment_length):
    """Finds all CDR-like fragments of the given length in the PDB entry,
    investigates distribution of various quantities related to the
    interactions with these fragments and plots these distributions"""
    results = investigate_interaction_distributions_single(pdb_id, fragment_length)

    print(results['num_cdrs'])
    print(results['proportion_cdrs'])

    sns.distplot(results['interactor_lengths'], norm_hist=True)
    plt.title("Size of interacting region")
    plt.xlabel("Number of residues interacting with CDR-like fragment")
    plt.ylabel("Density")
    save_plot("3cuq_interactor_lengths.png")
    plt.clf()

    sns.distplot(results['num_interactor_fragments'], norm_hist=True)
    plt.title("Number of interacting fragments")
    plt.xlabel("Number of contiguous fragments interacting with CDR-like fragment")
    plt.ylabel("Density")
    save_plot("3cuq_num_interactor_fragments.png")
    plt.clf()

    sns.distplot(results['sizes_interactor_fragments'], norm_hist=True)
    plt.title("Sizes of interacting fragments")
    plt.xlabel("Size of contiguous fragments interacting with CDR-like fragment")
    plt.ylabel("Density")
    save_plot("3cuq_sizes_interactor_fragments.png")
    plt.clf()

def plot_combined_interaction_distributions(combined_results):
    """Plot the combined results from analysing distribution of length and
    fragment size of interacting residues in multiple interaction matrices."""

    plt.clf()
    dummy_fig, ax = plt.subplots()
    sns.distplot(combined_results['proportions_cdrs'], ax=ax)
    ax.set_title("Proportion of fragments of length " +
                 str(combined_results['fragment_length']) +
                 " that are CDR-like")
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Density")
    save_plot("../plots/proportion_cdrs.png")

    plt.clf()
    dummy_fig, ax = plt.subplots()
    sns.distplot(combined_results['interactor_lengths'], ax=ax)
    ax.set_title("Size of interacting region")
    ax.set_xlabel("Number of residues interacting with CDR-like fragment")
    ax.set_ylabel("Density")
    save_plot("../plots/interactor_lengths.png")

    plt.clf()
    dummy_fig, ax = plt.subplots()
    sns.distplot(combined_results['num_interactor_fragments'], ax=ax)
    ax.set_title("Number of contiguous interacting fragments for each CDR-like fragment")
    ax.set_xlabel("Number of contiguous interacting fragments")
    ax.set_ylabel("Density")
    save_plot("../plots/num_interactor_fragments.png")

    plt.clf()
    dummy_fig, ax = plt.subplots()
    sns.distplot(combined_results['sizes_interactor_fragments'], ax=ax)
    ax.set_title("Lengths of contiguous interacting fragments")
    ax.set_xlabel("Length of interacting fragment")
    ax.set_ylabel("Density")
    save_plot("../plots/sizes_interactor_fragments.png")

    plt.clf()
    dummy_fig, ax = plt.subplots()
    sns.distplot(combined_results['cdr_observation_counts'], ax=ax)
    ax.set_title("Number of CDR fragments each fragment is similar to")
    ax.set_xlabel("Number of CDR fragments")
    ax.set_ylabel("Density")
    save_plot("../plots/cdr_observation_counts.png")

def plot_interaction_distributions_many(num_to_plot, fragment_length):
    """Investigates distribution of interacting fragments of many protein files.
    Chooses num_to_plot random files from con_dat.MATRIX_DIR and runs
    investigate_interaction_distributions_single on each, collates and plots the
    results of these analysis methods."""
    random.seed(42)

    # Choose random pdb_ids to work with
    random_matrix_files = random.sample(os.listdir(con_dat.MATRIX_DIR), k=num_to_plot)
    random_ids = [filename.split("_")[0] for filename in random_matrix_files]

    combined_results = {'fragment_length': fragment_length,
                        'num_to_plot': num_to_plot,
                        'proportions_cdrs': [],
                        'num_cdrs': [],
                        'interactor_lengths': [],
                        'num_interactor_fragments': [],
                        'sizes_interactor_fragments': [],
                        'cdr_observation_counts': []}

    fig_l, ax_l = plt.subplots()
    fig_n, ax_n = plt.subplots()
    fig_s, ax_s = plt.subplots()

    for pdb_id in random_ids:
        results = investigate_interaction_distributions_single(pdb_id,
                                                               fragment_length)

        sns.distplot(results['interactor_lengths'], norm_hist=True, ax=ax_l)
        sns.distplot(results['num_interactor_fragments'], norm_hist=True, ax=ax_n)
        sns.distplot(results['sizes_interactor_fragments'], norm_hist=True, ax=ax_s)

        combined_results['interactor_lengths'].extend(results['interactor_lengths'])
        combined_results['num_interactor_fragments'].extend(results['num_interactor_fragments'])
        combined_results['sizes_interactor_fragments'].extend(results['sizes_interactor_fragments'])

        combined_results['proportions_cdrs'].append(results['proportion_cdrs'])
        combined_results['num_cdrs'].append(results['num_cdrs'])
        combined_results['cdr_observation_counts'].extend(results['cdr_observation_counts'])

    def default(obj):
        if isinstance(obj, np.int32):
            return int(obj)
        raise TypeError

    with open("/sharedscratch/kcn25/eda/interaction_distributions.json", "w") as f:
        json.dump(combined_results, f, default=default)

    print("Total number of CDRs found: ", sum(combined_results['num_cdrs']))

    ax_l.set_title("Size of interacting region")
    ax_l.set_xlabel("Number of residues interacting with CDR-like fragment")
    ax_l.set_ylabel("Density")
    fig_l.savefig("../plots/individual_interactor_lengths.png")

    ax_n.set_title("Number of contiguous interacting fragments for each CDR-like fragment")
    ax_n.set_xlabel("Number of contiguous interacting fragments")
    ax_n.set_ylabel("Density")
    fig_n.savefig("../plots/individual_num_interactor_fragments.png")

    ax_s.set_title("Lengths of contiguous interacting fragments")
    ax_s.set_xlabel("Length of interacting fragment")
    ax_s.set_ylabel("Density")
    fig_s.savefig("../plots/individual_sizes_interactor_fragments.png")

    plot_combined_interaction_distributions(combined_results)

if __name__ == "__main__":
    plot_interaction_distributions_many(1000, 4)
