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
DATA_GROUPS = ['training', 'test', 'validation']

rule all:
    input:
        expand('processed/dataset/{data_group}_{data_type}.npy',
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
        '--complete_outfile {output.complete} > {log} 2>&1'

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
        'logs/find_unique_bound_pairs/log.log'
    output:
        'processed/bound_pairs/fragmented/unique_bound_pairs.csv',
    shell:
        'python3 scripts/find_unique_bound_pairs.py {output} {input.bound_pairs} > {log} 2>&1'

rule distance_matrix:
    input:
        bound_pairs='processed/bound_pairs/fragmented/unique_bound_pairs.csv',
    output:
        distance_matrix='processed/bound_pairs/fragmented/distance_matrix.npy',
    script:
        'scripts/generate_distance_matrix.py'

rule split_dataset:
    # Use this distance matrix to split the samples into different groups.
    #   Each group should contain similar samples, so that we can e.g.
    #   learn with one group and use another to assess generalisation ability.
    input:
        'processed/bound_pairs/fragmented/distance_matrix.npy',
    output:
        expand('processed/dataset_raw/{data_group}/positive.csv',
               data_group=DATA_GROUPS)
    script:
        'scripts/split_dataset.py'

rule generate_negatives:
    # Within each group, permute cdrs and targets to generate (assumed) negative
    #   examples
    input:
        'processed/dataset_raw/{data_group}/positive.csv'
    output:
        'processed/dataset_raw/{data_group}/negative.csv'
    script:
        'scripts/generate_negatives.py'

rule generate_representations:
    # For each group, generate the representations for both positive and negative
    #   data, and also produce the labels file.
    input:
         expand('processed/dataset_raw/{{data_group}}/{label}.csv',
                label=LABELS)
    output:
         expand('processed/dataset/{{data_group}}/{data_type}.npy',
                data_type=DATA_TYPES)
    script:
         'scripts/generate_representations.py'
