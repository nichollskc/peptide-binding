"""Clusters sequences and uses this clustering to partition dataset into
e.g. training and test sets."""
import json
import logging
import random
import re
import subprocess

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import numpy as np
import pandas as pd


def generate_cdr_sequence_record(row):
    """Generates a Bio.SeqRecord for the CDR sequence in a given row, with
    name containing information about the rest of the row."""
    indices = json.loads(row.cdr_bp_id_str)
    start_ind = indices[0]
    end_ind = indices[-1]
    name = "_".join(map(str, [row.cdr_pdb_id,
                              "@",
                              row.cdr_resnames,
                              "@",
                              start_ind,
                              end_ind]))
    record = SeqRecord(Seq(row.cdr_resnames, IUPAC.protein),
                       id=name,
                       name=name,
                       description=name)
    return record


def generate_target_sequence_record(row):
    """Generates a Bio.SeqRecord for the target sequence in a given row, with
    name containing information about the rest of the row."""
    indices = json.loads(row.target_bp_id_str)
    start_ind = indices[0]
    end_ind = indices[-1]
    name = "_".join(map(str, [row.target_pdb_id,
                              "@",
                              row.target_resnames,
                              "@",
                              start_ind,
                              end_ind]))
    record = SeqRecord(Seq(row.target_resnames, IUPAC.protein),
                       id=name,
                       name=name,
                       description=name)
    return record


def generate_sequence_records(data_frame):
    """Generate Bio.SeqRecords for every unique CDR sequence and every unique
    target sequence in the dataframe. Return each set of records in a list."""
    cdr_sequence_records = [generate_cdr_sequence_record(row)
                            for index, row
                            in data_frame.drop_duplicates(['cdr_resnames']).iterrows()]
    target_sequence_records = [generate_target_sequence_record(row)
                               for index, row
                               in data_frame.drop_duplicates(['target_resnames']).iterrows()]

    return cdr_sequence_records, target_sequence_records


def generate_fasta(data_frame, cdr_outfile, target_outfile):
    """Generate two fasta files: one containing all CDR sequences found in
    this data frame and one containing all target sequences found in this data frame.
    Write each to the specified fasta file."""
    cdr_sequence_records, target_sequence_records = generate_sequence_records(data_frame)

    SeqIO.write(cdr_sequence_records, cdr_outfile, "fasta")
    SeqIO.write(target_sequence_records, target_outfile, "fasta")


def generate_cdhit_clusters(fasta_file):
    """Cluster the sequences in a fasta file using cd-hit, returning the cluster
    that each sequence belongs to as a dictionary."""
    temp_cluster_output = "processed/clusters/cdhit_output"
    full_cmd = "cd-hit" \
               " -i {} -o {} -c 0.4 -n 2 -M 16000" \
               " -d 0 -T 32 -l 3".format(fasta_file,
                                         temp_cluster_output)
    subprocess.run(full_cmd.split(" "))
    clusters_dict = read_cdhit_clusters(temp_cluster_output + ".clstr")

    return clusters_dict


def read_cdhit_clusters(cdhit_cluster_file):
    """Read in the output of clustering with cd-hit and return as a dictionary."""
    with open(cdhit_cluster_file) as f:
        reader = f.read()
        sections = reader.split(">Cluster ")
    clusters_dict = {}
    for cluster_index, section in enumerate(sections):
        for line in section.split("\n"):
            id_match = re.match(r'\d+\s+\w+, >[\w_]*@_(\w+)_@[\w_]*\.\.\. ', line)
            if id_match:
                clusters_dict[id_match[1]] = cluster_index
    return clusters_dict


def cluster_sequences(bound_pairs_df):
    """Generate clusters using cd-hit based on CDR sequences and target sequences
    and return the data frame with columns for cluster IDs added."""
    cdr_fasta_file = "processed/clusters/cdr_fragments.fasta"
    target_fasta_file = "processed/clusters/target_fragments.fasta"
    generate_fasta(bound_pairs_df, cdr_fasta_file, target_fasta_file)

    cdr_clusters_dict = generate_cdhit_clusters(cdr_fasta_file)
    cdr_clusters_df = pd.DataFrame(cdr_clusters_dict.items(),
                                   columns=['cdr_resnames', 'cdr_cluster_id'])

    target_clusters_dict = generate_cdhit_clusters(target_fasta_file)
    target_clusters_df = pd.DataFrame(target_clusters_dict.items(),
                                      columns=['target_resnames', 'target_cluster_id'])

    with_cdr_clusters = bound_pairs_df.merge(cdr_clusters_df,
                                             on='cdr_resnames',
                                             how='left')
    with_both_clusters = with_cdr_clusters.merge(target_clusters_df,
                                                 on='target_resnames',
                                                 how='left')
    return with_both_clusters


def split_dataset_clustered(data_frame, group_proportions, seed=42):
    """Splits the rows of a data frame into groups according to the group
    proportions, keeping clusters (based on CDR sequence similarity) together.
    Group proportions should be a list e.g. [60, 20, 20].
    Within each group, clusters will be randomly ordered and randomly shuffled but
    will be blocks in the data. This means that splitting a group into k folds
    without shuffling result in clusters (mostly) being present in just a single fold."""
    #   e.g. [60, 20, 20] -> [0.6, 0.2, 0.2] -> [0.6, 0.6 + 0.2] = [0.6, 0.8]
    fractions = np.cumsum([group_proportions])[:-1] / sum(group_proportions)

    if 'cdr_cluster_id' in data_frame.columns:
        df_with_clusters = data_frame
    else:
        df_with_clusters = cluster_sequences(data_frame)
    clustered = [df.sample(frac=1, random_state=seed)
                 for _, df
                 in df_with_clusters.groupby('cdr_cluster_id')]
    random.shuffle(clustered)

    full_shuffled = pd.concat(clustered)

    logging.info(f"Intended fractions are {fractions}")
    counts = list(map(int, (fractions * len(full_shuffled))))
    logging.info(f"Intended (cumulative) cluster counts per group are {counts}")

    partitions = np.split(full_shuffled, counts)

    return partitions
