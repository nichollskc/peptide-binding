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
        'icMatrix/{pdb_id}.bmat',
    output:
        'bound_pairs_complete/bound_pairs_complete_{pdb_id}.csv',
        'bound_pairs_fragmented/bound_pairs_fragmented_{pdb_id}.csv',
    script:
        'scripts/find_all_bound_pairs.py',

rule find_unique_bound_pairs:
    input:
        expand('bound_pairs_fragmented/bound_pairs_fragmented_{pdb_id}.csv',
               pdb_id=PDB_IDS)
    output:
        'positive_datapoints/unique_bound_pairs.csv',
        'positive_datapoints/fragments.csv',
    script:
        'scripts/find_unique_bound_pairs.py'

rule split_dataset:
    # Calculate the distances between each bound pair and use this distance
    #   matrix to split the samples into different groups. Each group should
    #   contain similar samples, so that we can e.g. learn with one group and
    #   use another to assess generalisation ability.
    input:
        'bound_pairs_fragmented/unique_bound_pairs.csv'
    output:
        'output/distance_matrix.npy',
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