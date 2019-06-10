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
    groups = []
    group_names = []

    for char in string.ascii_lowercase + string.digits:
        pattern = re.compile('\w\w\w' + char)
        matches = list(filter(pattern.match, ids))

        if matches:
            groups.append(matches)
            group_names.append(char)

    return groups, group_names

PDB_IDS = get_all_pdb_ids()

GROUPED_IDS, GROUP_NAMES = group_ids(PDB_IDS)

LABELS = ['positive', 'negative']
DATA_TYPES = ['cdrs', 'targets', 'combined', 'labels']
DATA_GROUPS = ['training', 'validation', 'test']
DATA_GROUP_PROPORTIONS = [60, 20, 20]

rule all:
    input:
        expand('datasets/dataset_random/{data_group}_{data_type}.npy',
               data_group=DATA_GROUPS,
               data_type=DATA_TYPES)

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

for id_group, group_name in zip(GROUPED_IDS, GROUP_NAMES):
    # This rule just forces the find_all_bound_pairs rules to run in batches
        #   by placing them in the same rule group, and forcing aggregation in
        #   this rule
    rule:
        input:
            expand('processed/bound_pairs/fragmented/individual/{pdb_id}.csv', pdb_id=id_group)
        output:
            touch('processed/checks/' + group_name)
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
        'processed/bound_pairs/fragmented/unique_bound_pairs.csv',
    shell:
        'python3 scripts/find_unique_bound_pairs.py {output} {input.bound_pairs} '\
        '--verbosity 3 2>&1 | tee {log}'

rule generate_simple_negatives:
    # Within each group, permute cdrs and targets to generate (assumed) negative
    #   examples
    input:
        positives='processed/bound_pairs/fragmented/unique_bound_pairs.csv',
    log:
        'logs/generate_simple_negatives.log'
    output:
        combined='processed/simple_negatives.csv'
    shell:
        'python3 scripts/generate_simple_negatives.py {input.positives} '\
        '{output.combined} --verbosity 3 2>&1 | tee {log}'

rule generate_simple_representations:
    # For each group, generate the representations for both positive and negative
    #   data, and also produce the labels file.
    input:
         positives='processed/bound_pairs/fragmented/unique_bound_pairs.csv',
         negatives='processed/simple_negatives.csv'
    output:
         features='processed/N1_R1/features.npy',
         labels='processed/N1_R1/labels.npy'
    script:
         'scripts/generate_simple_representations.py'

rule split_dataset_random:
    input:
        features='processed/N1_R1/features.npy',
        labels='processed/N1_R1/labels.npy'
    params:
        group_proportions=DATA_GROUP_PROPORTIONS
    output:
        expand('datasets/N1_R1_S1/{data_group}/features.npy',
               data_group=DATA_GROUPS),
        expand('datasets/N1_R1_S1/{data_group}/labels.npy',
               data_group=DATA_GROUPS)
    script:
        'scripts/split_dataset_random.py'
