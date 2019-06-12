import glob
import random
import re
import string

import scripts.helper.utils as utils

def get_pdb_ids_file(filename):
    with open(filename, 'r') as f:
        files = f.readlines()
    ids = [f.split("_")[0] for f in files]
    return ids

def get_all_pdb_ids():
    files = glob.glob("icMatrix/*_icMat.bmat")
    ids = [f.split("/")[1].split("_")[0] for f in files]
    return ids

def get_random_pdb_ids(k=1000):
    ids = get_all_pdb_ids()

    random.seed(42)
    num_to_sample = min(k, len(ids))
    rand_ids = random.sample(ids, num_to_sample)
    return rand_ids

def group_ids(ids):
    groups = {}
    group_names = []

    for char in string.ascii_lowercase + string.digits:
        pattern = re.compile('\w\w\w' + char)
        matches = list(filter(pattern.match, ids))

        if matches:
            groups[char] = matches
            group_names.append(char)

    return groups, group_names

def group_name_to_csv_files(wildcards):
    id_group = GROUPED_IDS[wildcards.group_name]
    return expand('processed/bound_pairs/fragmented/individual/{pdb_id}.csv', pdb_id=id_group)

PDB_IDS = get_all_pdb_ids()

GROUPED_IDS, GROUP_NAMES = group_ids(PDB_IDS)

LABELS = ['positive', 'negative']
DATA_TYPES = ['bag_of_words', 'padded_meiler_onehot', 'product_bag_of_words']
DATA_GROUPS = ['training', 'validation', 'test']
DATA_GROUP_PROPORTIONS = [60, 20, 20]

rule all:
    input:
        expand('datasets/alpha/{data_group}/data_{data_type}.npy',
               data_group=DATA_GROUPS,
               data_type=DATA_TYPES),
        expand('datasets/alpha/{data_group}/labels.npy',
               data_group=DATA_GROUPS)

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
        'python3 scripts/find_all_bound_pairs.py --pdb_id {params.pdb_id} ' \
        '--cdr_fragment_length {params.cdr_fragment_length} ' \
        '--fragmented_outfile {output.fragmented} ' \
        '--complete_outfile {output.complete} --verbosity 3 2>&1 | tee {log}'

rule aggregate_bound_pairs:
# This rule just forces the find_all_bound_pairs rules to run in batches
    #   by placing them in the same rule group, and forcing aggregation in
    #   this rule
    input:
        group_name_to_csv_files
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
    shell:
        'python3 scripts/find_unique_bound_pairs.py {output.bound_pairs} {input.bound_pairs} '\
        '--fragment_lengths_out {output.fragment_lengths} --verbosity 3 2>&1 | tee {log}'

rule generate_simple_negatives:
    # Within each group, permute cdrs and targets to generate (assumed) negative
    #   examples
    input:
        positives='processed/bound_pairs/fragmented/unique_bound_pairs.csv',
    log:
        'logs/generate_simple_negatives.log'
    output:
        combined='processed/bound_pairs_simple_negatives.csv'
    shell:
        'python3 scripts/generate_simple_negatives.py {input.positives} '\
        '{output.combined} --verbosity 3 2>&1 | tee {log}'

rule split_dataset_random:
    input:
        combined=rules.generate_simple_negatives.output.combined,
    params:
        group_proportions=DATA_GROUP_PROPORTIONS
    log:
        'logs/split_dataset_random.log'
    output:
        data_filenames=expand('datasets/alpha/{data_group}/bound_pairs.csv',
                              data_group=DATA_GROUPS),
        label_filenames=expand('datasets/alpha/{data_group}/labels.npy',
                               data_group=DATA_GROUPS)
    shell:
        'python3 scripts/split_dataset_random.py --input {input.combined} '\
        '--group_proportions {params.group_proportions} '\
        '--data_filenames {output.data_filenames} '\
        '--label_filenames {output.label_filenames} '\
        '--seed 13 --verbosity 3 2>&1 | tee {log}'

rule generate_representations:
    # For each group, generate the representations for both positive and negative
    #   data, and also produce the labels file.
    input:
         dataset='datasets/alpha/{data_group}/bound_pairs.csv',
         fragment_lengths='processed/bound_pairs/fragmented/fragment_lengths.txt'
    params:
        representation='{representation}'
    log:
        'logs/generate_representations/{data_group}_{representation}.log'
    output:
         outfile='datasets/alpha/{data_group}/data_{representation}.npy'
    shell:
         'python3 scripts/generate_representations.py --input {input.dataset} '\
         '--output_file {output.outfile} --representation {params.representation} '\
         '--fragment_lengths_file {input.fragment_lengths} --verbosity 3 2>&1 | tee {log}'
