import glob
import itertools
import os
import random
import re
import string

import scripts.helper.construct_database as con_dat
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

def group_name_to_sdf_files(wildcards):
    id_group = GROUPED_BOUND_PAIR_IDS[wildcards.group_name]
    return expand('processed/sdfs/{bound_pair_id}.sdf', bound_pair_id=id_group)

def get_bound_pair_sdf_filenames(wildcards):
    bound_pair_ids = get_all_bound_pair_ids(f"datasets/{wildcards.full_dataset}/bound_pairs.csv")
    return [f"processed/sdfs/{bound_pair_id}.sdf" for bound_pair_id in bound_pair_ids]

def get_all_bound_pair_ids(bound_pairs_df_filename):
    prefix, extension = os.path.splitext(bound_pairs_df_filename)
    ids_filename = prefix + '.ids.txt'
    try:
        with open(ids_filename, 'r') as f:
            raw_bound_pair_ids = f.readlines()
            bound_pair_ids = [line.strip() for line in raw_bound_pair_ids]
    except FileNotFoundError:
        df = con_dat.read_bound_pairs(bound_pairs_df_filename)
        bound_pair_ids = [utils.get_bound_pair_id_from_row(row) for ind, row in df.iterrows()]
        with open(ids_filename, 'w') as f:
            f.write('\n'.join(bound_pair_ids))

    return bound_pair_ids

def combine_all_bound_pair_ids():
    filenames = expand('datasets/beta/small/{size}/{data_group}/bound_pairs.csv',
                                   data_group=BETA_DATA_GROUPS,
                                   size=['10000']) +\
                expand('datasets/beta/thresholds/{threshold}/{data_group}/bound_pairs.csv',
                       data_group=THRESHOLD_GROUPS,
                       threshold=ALIGNMENT_THRESHOLDS)
    all_lists = [get_all_bound_pair_ids(file) for file in filenames]
    combined = itertools.chain.from_iterable(all_lists)
    no_duplicates = list(set(combined))
    return no_duplicates

PDB_IDS = get_all_pdb_ids()
GROUPED_IDS, GROUP_NAMES = group_ids(PDB_IDS)

LABELS = ['positive', 'negative']
DATA_TYPES = ['bag_of_words', 'padded_meiler_onehot', 'product_bag_of_words']
ALPHA_DATA_GROUPS = ['training', 'validation', 'test']
ALPHA_DATA_GROUP_PROPORTIONS = [60, 20, 20]

BETA_DATA_GROUPS = ['rand/training', 'rand/validation', 'rand/test',
                    'clust/training', 'clust/validation', 'clust/test']
BETA_DATA_GROUP_PROPORTIONS = [60, 20, 10, 10]

THRESHOLD_GROUPS = ['training', 'validation']
ALIGNMENT_THRESHOLDS = [0, -2, -4, -8]

BOUND_PAIR_IDS = combine_all_bound_pair_ids()
GROUPED_BOUND_PAIR_IDS, GROUP_BOUND_PAIR_NAMES = group_ids(BOUND_PAIR_IDS)

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
    script:
        'scripts/snakemake_find_unique_bound_pairs.py'

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
        'python3 scripts/generate_simple_negatives.py {input.positives} '\
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
        'python3 scripts/split_dataset_random.py --input {input.combined} '\
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
        'python3 scripts/split_dataset_clusters_random.py --input {input.combined} '\
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
        'python3 scripts/split_dataset_clusters_random.py --input {input.combined} '\
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
        'python3 scripts/split_dataset_thresholds.py --input {input.combined} '\
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
         'python3 scripts/generate_representations.py --input {input.dataset} '\
         '--output_file {output.outfile} --representation {params.representation} '\
         '--fragment_lengths_file {input.fragment_lengths} --verbosity 3 2>&1 | tee {log}'

rule generate_pdb:
    params:
         bound_pair_id='{bound_pair_id}'
    group:
        'e3fp'
    output:
         'processed/pdbs/{bound_pair_id}.pdb'
    shell:
         'python3 -c "import scripts.helper.query_biopython as bio; '\
         'bio.write_bound_pair_to_pdb_wrapped(\'{params.bound_pair_id}\')" '\
         '>> logs/generate_pdb.log 2>&1'

rule convert_pdb_sdf:
    input:
         'processed/pdbs/{bound_pair_id}.pdb'
    group:
        'e3fp'
    output:
         'processed/sdfs/{bound_pair_id}.sdf'
    shell:
         'obabel -ipdb {input} -osdf -O {output} >> logs/convert_pdb_sdf.log 2>&1'

rule aggregate_pdb_generation:
# This rule just forces the convert_pdb_sdf rules to run in batches
    #   by placing them in the same rule group, and forcing aggregation in
    #   this rule
    input:
        group_name_to_sdf_files
    output:
        touch('processed/sdf_checks/{group_name}')
    group:
        'e3fp'

rule all_sdf_files:
    input:
         group_names=expand('processed/sdf_checks/{group_name}', group_name=GROUP_BOUND_PAIR_NAMES)

rule generate_e3fp_fingerprints:
    input:
         sdf_filenames=get_bound_pair_sdf_filenames
    params:
         dataset='{full_dataset}'
    log:
         'logs/generate_representations/{full_dataset}_e3fp_fingerprints.log'
    output:
         dataset='datasets/{full_dataset}/data_fingerprints.npz'
    conda:
         'e3fp_env.yml'
    script:
         'scripts/snakemake_generate_structure_representations.py'
