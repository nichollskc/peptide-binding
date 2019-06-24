"""Performs data exploration of database."""

import json
# import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# import construct_database as con_dat
import scripts.helper.distances as distances


def save_plot(filename, folder="plots/"):
    """Saves plot to a file in the given folder."""
    full_filename = folder + filename
    plt.tight_layout()
    plt.savefig(full_filename, bbox_inches='tight', dpi=300)


#
# def investigate_interaction_distributions_single(pdb_id, fragment_length):
#     """Finds all CDR-like fragments of the given length in the PDB entry and
#     investigates distribution of various quantities related to the
#     interactions with these fragments.
#
#     Args:
#         pdb_id (string): root of filename e.g. ../example_files/3cuq
#         fragment_length (int): length of CDR-like fragment to search for
#
#     Returns:
#         int: proportion of possible kmers (length fragment_length) that are
#             labelled as CDR-like
#         array[int]: for each CDR-like fragment, number of residues labelled as
#             interacting with residues in the fragment
#         array[int]: for each CDR-like fragment, if we split the target
#             residues into contiguous fragments, how many fragments do we have?
#         array[int]: lengths of each contiguous fragment as described above
#     """
#     matrix = con_dat.read_matrix_from_file(pdb_id)
#     binding_pairs = con_dat.find_all_binding_pairs_indices(matrix, fragment_length)
#
#     # Calculate the proportion of kmers that are cdr-like
#     num_cdrs = len(binding_pairs)
#     num_total_kmers = matrix.shape[0] - fragment_length + 1
#     proportion_cdrs = num_cdrs / num_total_kmers
#
#     target_lengths = [len(bp[1]) for bp in binding_pairs]
#
#     target_fragments = [con_dat.find_contiguous_fragments(bp[1],
#                                                           con_dat.get_id_filename(pdb_id))
#                         for bp in binding_pairs]
#
#     num_target_fragments = [len(fragments) for fragments in target_fragments]
#     sizes_target_fragments = [len(fragment)
#                               for fragments in target_fragments
#                               for fragment in fragments]
#
#     results = {}
#     results['proportion_cdrs'] = proportion_cdrs
#     results['num_cdrs'] = num_cdrs
#     results['target_lengths'] = target_lengths
#     results['num_target_fragments'] = num_target_fragments
#     results['sizes_target_fragments'] = sizes_target_fragments
#     results['cdr_observation_counts'] = list(np.diagonal(matrix, offset=3))
#     return results
#
#
# def plot_interaction_distributions_single(pdb_id, fragment_length):
#     """Finds all CDR-like fragments of the given length in the PDB entry,
#     investigates distribution of various quantities related to the
#     interactions with these fragments and plots these distributions"""
#     results = investigate_interaction_distributions_single(pdb_id, fragment_length)
#
#     print(results['num_cdrs'])
#     print(results['proportion_cdrs'])
#
#     sns.distplot(results['target_lengths'], norm_hist=True)
#     plt.title("Size of target region")
#     plt.xlabel("Number of residues target with CDR-like fragment")
#     plt.ylabel("Density")
#     save_plot("3cuq_target_lengths.png")
#     plt.clf()
#
#     sns.distplot(results['num_target_fragments'], norm_hist=True)
#     plt.title("Number of target fragments")
#     plt.xlabel("Number of contiguous fragments interacting with CDR-like fragment")
#     plt.ylabel("Density")
#     save_plot("3cuq_num_target_fragments.png")
#     plt.clf()
#
#     sns.distplot(results['sizes_target_fragments'], norm_hist=True)
#     plt.title("Sizes of target fragments")
#     plt.xlabel("Size of contiguous fragments interacting with CDR-like fragment")
#     plt.ylabel("Density")
#     save_plot("3cuq_sizes_target_fragments.png")
#     plt.clf()
#
#
# def plot_combined_interaction_distributions(combined_results):
#     """Plot the combined results from analysing distribution of length and
#     fragment size of target residues in multiple interaction matrices."""
#
#     plt.clf()
#     dummy_fig, ax = plt.subplots()
#     sns.distplot(combined_results['proportions_cdrs'], ax=ax)
#     ax.set_title("Proportion of fragments of length " +
#                  str(combined_results['fragment_length']) +
#                  " that are CDR-like")
#     ax.set_xlabel("Proportion")
#     ax.set_ylabel("Density")
#     save_plot("../plots/proportion_cdrs.png")
#
#     plt.clf()
#     dummy_fig, ax = plt.subplots()
#     sns.distplot(combined_results['target_lengths'], ax=ax)
#     ax.set_title("Size of target region")
#     ax.set_xlabel("Number of residues interacting with CDR-like fragment")
#     ax.set_ylabel("Density")
#     save_plot("../plots/target_lengths.png")
#
#     plt.clf()
#     dummy_fig, ax = plt.subplots()
#     sns.distplot(combined_results['num_target_fragments'], ax=ax)
#     ax.set_title("Number of contiguous target fragments for each CDR-like fragment")
#     ax.set_xlabel("Number of contiguous target fragments")
#     ax.set_ylabel("Density")
#     save_plot("../plots/num_target_fragments.png")
#
#     plt.clf()
#     dummy_fig, ax = plt.subplots()
#     sns.distplot(combined_results['sizes_target_fragments'], ax=ax)
#     ax.set_title("Lengths of contiguous target fragments")
#     ax.set_xlabel("Length of target fragment")
#     ax.set_ylabel("Density")
#     save_plot("../plots/sizes_target_fragments.png")
#
#     plt.clf()
#     dummy_fig, ax = plt.subplots()
#     sns.distplot(combined_results['cdr_observation_counts'], ax=ax)
#     ax.set_title("Number of CDR fragments each fragment is similar to")
#     ax.set_xlabel("Number of CDR fragments")
#     ax.set_ylabel("Density")
#     save_plot("../plots/cdr_observation_counts.png")
#
#
# def plot_interaction_distributions_many(num_to_plot, fragment_length):
#     """Investigates distribution of target fragments of many protein files.
#     Chooses num_to_plot random files from con_dat.MATRIX_DIR and runs
#     investigate_interaction_distributions_single on each, collates and plots the
#     results of these analysis methods."""
#     random.seed(42)
#
#     # Choose random pdb_ids to work with
#     random_matrix_files = random.sample(os.listdir(con_dat.MATRIX_DIR), k=num_to_plot)
#     random_ids = [filename.split("_")[0] for filename in random_matrix_files]
#
#     combined_results = {'fragment_length': fragment_length,
#                         'num_to_plot': num_to_plot,
#                         'proportions_cdrs': [],
#                         'num_cdrs': [],
#                         'target_lengths': [],
#                         'num_target_fragments': [],
#                         'sizes_target_fragments': [],
#                         'cdr_observation_counts': []}
#
#     fig_l, ax_l = plt.subplots()
#     fig_n, ax_n = plt.subplots()
#     fig_s, ax_s = plt.subplots()
#
#     for pdb_id in random_ids:
#         results = investigate_interaction_distributions_single(pdb_id,
#                                                                fragment_length)
#
#         sns.distplot(results['target_lengths'], norm_hist=True, ax=ax_l)
#         sns.distplot(results['num_target_fragments'], norm_hist=True, ax=ax_n)
#         sns.distplot(results['sizes_target_fragments'], norm_hist=True, ax=ax_s)
#
#         combined_results['target_lengths'].extend(results['target_lengths'])
#         combined_results['num_target_fragments'].extend(results['num_target_fragments'])
#         combined_results['sizes_target_fragments'].extend(results['sizes_target_fragments'])
#
#         combined_results['proportions_cdrs'].append(results['proportion_cdrs'])
#         combined_results['num_cdrs'].append(results['num_cdrs'])
#         combined_results['cdr_observation_counts'].extend(results['cdr_observation_counts'])
#
#     def default(obj):
#         if isinstance(obj, np.int32):
#             return int(obj)
#         raise TypeError
#
#     with open("/sharedscratch/kcn25/eda/interaction_distributions.json", "w") as f:
#         json.dump(combined_results, f, default=default)
#
#     print("Total number of CDRs found: ", sum(combined_results['num_cdrs']))
#
#     ax_l.set_title("Size of target region")
#     ax_l.set_xlabel("Number of residues interacting with CDR-like fragment")
#     ax_l.set_ylabel("Density")
#     fig_l.savefig("../plots/individual_target_lengths.png")
#
#     ax_n.set_title("Number of contiguous target fragments for each CDR-like fragment")
#     ax_n.set_xlabel("Number of contiguous target fragments")
#     ax_n.set_ylabel("Density")
#     fig_n.savefig("../plots/individual_num_target_fragments.png")
#
#     ax_s.set_title("Lengths of contiguous target fragments")
#     ax_s.set_xlabel("Length of target fragment")
#     ax_s.set_ylabel("Density")
#     fig_s.savefig("../plots/individual_sizes_target_fragments.png")
#
#     plot_combined_interaction_distributions(combined_results)


def explore_observation_count_threshold(thresholds):
    """Investigate how many CDR-like fragments would be discarded if we insisted
    that it must have been found in `threshold` CDR domains for it to count
    as CDR-like."""
    with open("/sharedscratch/kcn25/eda/interaction_distributions.json", "r") as f:
        results = json.load(f)

    cdr_observation_counts = np.array(results['cdr_observation_counts'])

    print("Total fragments considered: " + str(len(cdr_observation_counts)))

    # First look at the number of fragments that were not observed as similar
    #   to *any* CDR domains
    not_cdrs = np.where((cdr_observation_counts < 1) &
                        (cdr_observation_counts > - 2))
    print("Number of fragments not similar to any CDR domains: " +
          str(len(not_cdrs[0])))

    for threshold in thresholds:
        below_threshold = np.where((cdr_observation_counts < threshold) &
                                   (cdr_observation_counts > - threshold))
        print("Number of fragments similar to " +
              str(threshold) +
              " or fewer CDR fragments: " +
              str(len(below_threshold[0])))


def calculate_alignment_scores(data_frame, index_1, index_2):
    """Find the alignment scores between CDRs and between targets in the rows
    given by index_1 and index_2. I.e. return alignment(row_1['cdr'], row_2['cdr'])
    and alignment(row_1['target'], row_2['target'])."""
    row1 = data_frame.loc[index_1, :]
    row2 = data_frame.loc[index_2, :]

    cdr_score = distances.calculate_alignment_score(row1['cdr_resnames'],
                                                    row2['cdr_resnames'])
    target_score = distances.calculate_alignment_score(row1['target_resnames'],
                                                       row2['target_resnames'])

    return cdr_score, target_score


def explore_alignment_scores(bound_pairs_df, k=10):
    """Explore distribution of alignment scores between rows of the data frame.
    Look at alignment scores just between CDRs, just between targets and combined."""
    donors = random.sample(list(bound_pairs_df.index), k)
    acceptors = random.sample(list(bound_pairs_df.index), k)

    similarities = []
    cdr_similarities = []
    target_similarities = []
    for donor_index, acceptor_index in zip(donors, acceptors):
        cdr_score, target_score = calculate_alignment_scores(bound_pairs_df,
                                                             donor_index,
                                                             acceptor_index)

        similarity = cdr_score + target_score
        similarities.append(similarity)
        cdr_similarities.append(cdr_score)
        target_similarities.append(target_score)

    plt.clf()
    unused_fig, ax = plt.subplots(2, 1)

    sns.distplot(similarities, label="sum", ax=ax[0])
    sns.distplot(cdr_similarities, label="cdr", ax=ax[1])
    sns.distplot(target_similarities, label="target", ax=ax[1])

    ax[0].set_title("Sum of CDR alignment and target alignment")
    ax[1].set_title("Individual sequence alignments")
    ax[1].legend()

    save_plot(f"explore_alignments_{k}.png")
    plt.show()


if __name__ == "__main__":
    # plot_interaction_distributions_many(1000, 4)
    explore_observation_count_threshold(thresholds=[2, 3, 5, 10, 20, 50, 100])
