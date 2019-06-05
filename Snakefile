import scripts.utils as utils

PDB_IDS = ['3cuq', '1mhp']
LABELS = ['positive', 'negative']
DATA_TYPES = ['cdrs', 'targets', 'combined', 'labels']
DATA_GROUPS = ['training', 'test', 'validation']

rule all:
    input:
        expand('dataset/{data_group}_{data_type}.npy',
               data_group=DATA_GROUPS,
               data_type=DATA_TYPES)

rule find_all_bound_pairs:
    input:
        ids=utils.get_id_filename('{pdb_id}'),
        pdb=utils.get_pdb_filename('{pdb_id}'),
        matrix=utils.get_matrix_filename('{pdb_id}')
    params:
        pdb_id='{pdb_id}',
        cdr_fragment_length=4,
    output:
        complete='bound_pairs/complete/individual/{pdb_id}.csv',
        fragmented='bound_pairs/fragmented/individual/{pdb_id}.csv',
    script:
        'scripts/find_all_bound_pairs.py'

rule find_unique_bound_pairs:
    input:
        expand('bound_pairs/fragmented/individual/{pdb_id}.csv',
               pdb_id=PDB_IDS)
    output:
        'bound_pairs/fragmented/unique_bound_pairs.csv',
    script:
        'scripts/find_unique_bound_pairs.py'

rule distance_matrix:
    input:
        bound_pairs='bound_pairs/fragmented/unique_bound_pairs.csv',
    output:
        distance_matrix='bound_pairs/fragmented/distance_matrix.npy',
    script:
        'scripts/generate_distance_matrix.py'

rule split_dataset:
    # Use this distance matrix to split the samples into different groups.
    #   Each group should contain similar samples, so that we can e.g.
    #   learn with one group and use another to assess generalisation ability.
    input:
        'bound_pairs/fragmented/distance_matrix.npy',
    output:
        expand('dataset_raw/{data_group}/positive.csv',
               data_group=DATA_GROUPS)
    script:
        'scripts/split_dataset.py'

rule generate_negatives:
    # Within each group, permute cdrs and targets to generate (assumed) negative
    #   examples
    input:
        'dataset_raw/{data_group}/positive.csv'
    output:
        'dataset_raw/{data_group}/negative.csv'
    script:
        'scripts/generate_negatives.py'

rule generate_representations:
    # For each group, generate the representations for both positive and negative
    #   data, and also produce the labels file.
    input:
         expand('dataset_raw/{{data_group}}/{label}.csv',
                label=LABELS)
    output:
         expand('dataset/{{data_group}}/{data_type}.npy',
                data_type=DATA_TYPES)
    script:
         'scripts/generate_representations.py'