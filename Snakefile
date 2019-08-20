import glob
import itertools
import os
import random
import re
import string

import peptidebinding.helper.construct_database as con_dat
import peptidebinding.helper.utils as utils

# The PEPBIND_PYTHON variable allows us to decide which command to use to run the
#   python scripts when calling snakemake. For example we can use
#   PEPBIND_PYTHON='coverage run -a -m' to collect code coverage information.
if 'PEPBIND_PYTHON' not in os.environ:
    os.environ['PEPBIND_PYTHON'] = 'python3 -m'


def get_all_pdb_ids():
    """Returns a list of all PDB IDs for which we have a .bmat input file, i.e.
    all PDB IDs that will be used in construction of datasets."""
    files = glob.glob("icMatrix/*_icMat.bmat")
    ids = [f.split("/")[1].split("_")[0] for f in files]
    return ids


def get_groups_of_ids(ids):
    """Takes a list of IDs and separates them into groups. Returns a list of lists,
    with each list being one group. Also returns a list of names of the groups.
    Every ID will be placed in precisely one group."""
    groups = {}
    group_names = []

    for char in string.ascii_lowercase + string.digits:
        pattern = re.compile('\w\w\w' + char)
        matches = list(filter(pattern.match, ids))

        if matches:
            groups[char] = matches
            group_names.append(char)

    return groups, group_names


def get_csv_filenames_from_groupname(wildcards):
    """Given the name of a group of PDB IDs (given as a wildcard), return the
    list of csv filenames corresponding to the members of this group."""
    # Obtain the list of PDB IDs using the group name
    id_group = GROUPED_IDS[wildcards.group_name]

    # Use snakemake's expand to convert this list into a list of the csv filenames
    #   containing fragmented bound pairs
    return expand('processed/bound_pairs/fragmented/individual/{pdb_id}.csv', pdb_id=id_group)


PDB_IDS = get_all_pdb_ids()
GROUPED_IDS, GROUP_NAMES = get_groups_of_ids(PDB_IDS)

LABELS = ['positive', 'negative']
DATA_TYPES = ['bag_of_words', 'padded_meiler_onehot', 'product_bag_of_words']
ALPHA_DATA_GROUPS = ['training', 'validation', 'test']
ALPHA_DATA_GROUP_PROPORTIONS = [60, 20, 20]

BETA_DATA_GROUPS = ['rand/training', 'rand/validation', 'rand/test',
                    'clust/training', 'clust/validation', 'clust/test']
BETA_DATA_GROUP_PROPORTIONS = [60, 20, 10, 10]

THRESHOLD_GROUPS = ['training', 'validation']
ALIGNMENT_THRESHOLDS = [0, -2, -4, -8]

rule all:
    input:
        # Full dataset
        expand('datasets/beta/{data_group}/data_{data_type}.npy',
               data_group=BETA_DATA_GROUPS,
               data_type=DATA_TYPES),
        expand('datasets/beta/{data_group}/labels.npy',
               data_group=BETA_DATA_GROUPS),
        # Smaller subsets of the dataset
        expand('datasets/beta/small/10000/{data_group}/data_{data_type}.npy',
               data_group=BETA_DATA_GROUPS,
               data_type=DATA_TYPES),
        expand('datasets/beta/small/10000/{data_group}/data_fingerprints.npz',
               data_group=BETA_DATA_GROUPS),
        expand('datasets/beta/small/10000/{data_group}/labels.npy',
               data_group=BETA_DATA_GROUPS),
        expand('datasets/beta/small/100000/{data_group}/data_{data_type}.npy',
               data_group=BETA_DATA_GROUPS,
               data_type=DATA_TYPES),
        expand('datasets/beta/small/100000/{data_group}/data_fingerprints.npz',
               data_group=BETA_DATA_GROUPS),
        expand('datasets/beta/small/100000/{data_group}/labels.npy',
               data_group=BETA_DATA_GROUPS),
        expand('datasets/beta/small/1000000/{data_group}/data_{data_type}.npy',
               data_group=BETA_DATA_GROUPS,
               data_type=DATA_TYPES),
        expand('datasets/beta/small/1000000/{data_group}/data_fingerprints.npz',
               data_group=BETA_DATA_GROUPS),
        expand('datasets/beta/small/1000000/{data_group}/labels.npy',
               data_group=BETA_DATA_GROUPS),
        # Different versions of the dataset at different threshold values
        expand('datasets/beta/thresholds/clust/{threshold}/{data_group}/data_{data_type}.npy',
               data_group=THRESHOLD_GROUPS,
               data_type=DATA_TYPES,
               threshold=ALIGNMENT_THRESHOLDS),
        expand('datasets/beta/thresholds/clust/{threshold}/{data_group}/labels.npy',
               data_group=THRESHOLD_GROUPS,
               threshold=ALIGNMENT_THRESHOLDS),

rule test:
    input:
        # Representations of one of the small subsets (test set only)
        expand('datasets/beta/small/100/clust/test/data_{data_type}.npy',
               data_type=DATA_TYPES),
        # Train, validation and test for fingerprints, since we will train on this
        #   representation
        expand('datasets/beta/small/100/{data_group}/data_fingerprints.npz',
               data_group=['clust/training', 'clust/validation']),
        expand('datasets/beta/small/100/{data_group}/labels.npy',
               data_group=['clust/training', 'clust/validation']),
        # Different version of the dataset at different threshold value
        expand('datasets/beta/thresholds/clust/-2/{data_group}/bound_pairs.csv',
               data_group=THRESHOLD_GROUPS),
        # Simple random splitting of dataset
        expand('datasets/alpha/{data_group}/bound_pairs.csv',
               data_group=ALPHA_DATA_GROUPS),

rule find_all_bound_pairs:
    input:
        ids=utils.get_id_filename('{pdb_id}'),
        pdb=utils.get_pdb_filename('{pdb_id}'),
        matrix=utils.get_matrix_filename('{pdb_id}')
    group:
        'bound_pairs'
    params:
        pdb_id='{pdb_id}',
        cdr_fragment_length=4,
    log:
        'logs/find_all_bound_pairs/{pdb_id}.log'
    output:
        complete='processed/bound_pairs/complete/individual/{pdb_id}.csv',
        fragmented='processed/bound_pairs/fragmented/individual/{pdb_id}.csv',
    shell:
        '$PEPBIND_PYTHON peptidebinding.find_all_bound_pairs --pdb_id {params.pdb_id} ' \
        '--cdr_fragment_length {params.cdr_fragment_length} ' \
        '--fragmented_outfile {output.fragmented} ' \
        '--complete_outfile {output.complete} --verbosity 3 2>&1 | tee {log}'

rule aggregate_bound_pairs:
# This rule just forces the find_all_bound_pairs rules to run in batches
    #   by placing them in the same rule group, and forcing aggregation in
    #   this rule
    input:
        get_csv_filenames_from_groupname
    output:
        touch('processed/checks/{group_name}')
    group:
        'bound_pairs'

rule find_unique_bound_pairs:
    input:
        bound_pairs=expand('processed/bound_pairs/fragmented/individual/{pdb_id}.csv',
               pdb_id=PDB_IDS),
        group_names=expand('processed/checks/{group_name}', group_name=GROUP_NAMES)
    log:
        'logs/find_unique_bound_pairs.log'
    output:
        bound_pairs='processed/bound_pairs/fragmented/unique_bound_pairs.csv',
        fragment_lengths='processed/bound_pairs/fragmented/fragment_lengths.txt'
    script:
        'peptidebinding/snakemake_find_unique_bound_pairs.py'

rule generate_simple_negatives:
    # Within each group, permute cdrs and targets to generate (assumed) negative
    #   examples
    input:
        positives='processed/bound_pairs/fragmented/unique_bound_pairs.csv',
    log:
        log='logs/generate_simple_negatives.log'
    output:
        combined='processed/bound_pairs_simple_negatives.csv'
    shell:
        '$PEPBIND_PYTHON peptidebinding.generate_simple_negatives {input.positives} '\
        '{output.combined} --verbosity 3 2>&1 | tee {log}'

rule split_dataset_random:
    input:
        combined=rules.generate_simple_negatives.output.combined,
    params:
        group_proportions=ALPHA_DATA_GROUP_PROPORTIONS
    log:
        'logs/split_dataset_random.log'
    output:
        data_filenames=expand('datasets/alpha/{data_group}/bound_pairs.csv',
                              data_group=ALPHA_DATA_GROUPS),
        label_filenames=expand('datasets/alpha/{data_group}/labels.npy',
                               data_group=ALPHA_DATA_GROUPS)
    shell:
        '$PEPBIND_PYTHON peptidebinding.split_dataset_random --input {input.combined} '\
        '--group_proportions {params.group_proportions} '\
        '--data_filenames {output.data_filenames} '\
        '--label_filenames {output.label_filenames} '\
        '--seed 13 --verbosity 3 2>&1 | tee {log}'

rule split_dataset_beta:
    input:
        combined=rules.generate_simple_negatives.output.combined,
    params:
        group_proportions=BETA_DATA_GROUP_PROPORTIONS
    log:
        'logs/split_dataset_beta.log'
    output:
        "processed/clusters/cdr_fragments.fasta",
        data_filenames=expand('datasets/beta/{data_group}/bound_pairs.csv',
                              data_group=BETA_DATA_GROUPS),
        label_filenames=expand('datasets/beta/{data_group}/labels.npy',
                               data_group=BETA_DATA_GROUPS)
    shell:
        '$PEPBIND_PYTHON peptidebinding.split_dataset_clusters_random --input {input.combined} '\
        '--group_proportions {params.group_proportions} '\
        '--data_filenames {output.data_filenames} '\
        '--label_filenames {output.label_filenames} '\
        '--seed 13 --verbosity 3 2>&1 | tee {log}'

rule reduce_dataset:
    input:
        combined='datasets/beta/rand/training/bound_pairs.csv',
    params:
        size='{size}'
    log:
        'logs/split_dataset_small_{size}.log'
    output:
        reduced='datasets/beta/small/{size}/full_bound_pairs.csv'
    shell:
        'head -n$(({params.size}+1)) {input.combined} > {output.reduced}'

rule split_dataset_beta_small:
    input:
        combined='datasets/beta/small/{size}/full_bound_pairs.csv'
    params:
        group_proportions=BETA_DATA_GROUP_PROPORTIONS,
        size='{size}'
    log:
        'logs/split_dataset_beta_small_{size}.log'
    output:
        data_filenames=expand('datasets/beta/small/{{size}}/{data_group}/bound_pairs.csv',
                              data_group=BETA_DATA_GROUPS),
        label_filenames=expand('datasets/beta/small/{{size}}/{data_group}/labels.npy',
                               data_group=BETA_DATA_GROUPS)
    shell:
        '$PEPBIND_PYTHON peptidebinding.split_dataset_clusters_random --input {input.combined} '\
        '--group_proportions {params.group_proportions} '\
        '--data_filenames {output.data_filenames} '\
        '--label_filenames {output.label_filenames} '\
        '--seed 13 --verbosity 3 2>&1 | tee {log}'

rule split_dataset_thresholds:
    input:
        combined='datasets/beta/small/1000000/full_bound_pairs.csv'
    log:
        'logs/split_dataset_alignment_thresholds.log'
    output:
        data_filenames=expand('datasets/beta/thresholds/clust/{threshold}/{data_group}/bound_pairs.csv',
                              threshold=ALIGNMENT_THRESHOLDS,
                              data_group=THRESHOLD_GROUPS),
        label_filenames=expand('datasets/beta/thresholds/clust/{threshold}/{data_group}/labels.npy',
                               threshold=ALIGNMENT_THRESHOLDS,
                               data_group=THRESHOLD_GROUPS)
    shell:
        '$PEPBIND_PYTHON peptidebinding.split_dataset_thresholds --input {input.combined} '\
        '--data_filenames {output.data_filenames} --label_filenames {output.label_filenames} '\
        '--thresholds 0 -2 -4 -8 --num_negatives 15000 '\
        '--seed 13 --verbosity 3 2>&1 | tee {log}'

rule generate_representations:
    # For each group, generate the representations for both positive and negative
    #   data, and also produce the labels file.
    input:
         dataset='datasets/{full_data_group}/bound_pairs.csv',
         fragment_lengths='processed/bound_pairs/fragmented/fragment_lengths.txt'
    params:
        representation='{representation}'
    log:
        'logs/generate_representations/{full_data_group}_{representation}.log'
    output:
         outfile='datasets/{full_data_group}/data_{representation}.npy'
    shell:
         '$PEPBIND_PYTHON peptidebinding.generate_representations --input {input.dataset} '\
         '--output_file {output.outfile} --representation {params.representation} '\
         '--fragment_lengths_file {input.fragment_lengths} --verbosity 3 2>&1 | tee {log}'

rule generate_e3fp_fingerprints:
    input:
         dataset='datasets/{full_dataset}/bound_pairs.csv'
    params:
         dataset='{full_dataset}'
    log:
         'logs/generate_representations/{full_dataset}_e3fp_fingerprints.log'
    output:
         dataset='datasets/{full_dataset}/data_fingerprints.npz'
    conda:
         'e3fp_env.yml'
    shell:
        '$PEPBIND_PYTHON peptidebinding.generate_fingerprint_representations --input {input.dataset} '\
        '--outfile {output.dataset} --verbosity 3 2>&1 | tee {log}'
