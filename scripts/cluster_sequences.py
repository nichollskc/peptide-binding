import json
import random
import re
import subprocess

import matplotlib.pylot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC


def generate_cdr_sequence_record(row):
    indices = json.loads(row.cdr_bp_id_str)
    start_ind = indices[0]
    end_ind = indices[-1]
    name = "_".join(map(str, [row.pdb_id, start_ind, end_ind]))
    record = SeqRecord(Seq(row.cdr_resnames, IUPAC.protein), id=name, name=name, description=name)
    return record


def generate_cdr_sequence_records(csv_filename):
    df = pd.read_csv(csv_filename, header=0, index_col=None)
    sequence_records = [generate_cdr_sequence_record(row)
                        for index, row in df.iterrows()]
    return sequence_records


def generate_fasta(csv_filename, outfile):
    sequence_records = generate_cdr_sequence_records(csv_filename)
    SeqIO.write(sequence_records, outfile, "fasta")


def read_cdhit_clusters(cdhit_cluster_file):
    with open(cdhit_cluster_file) as f:
        reader = f.read()
        sections = reader.split(">Cluster ")
    clusters_dict = {}
    for cluster_index, section in enumerate(sections):
        for line in section.split("\n"):
            id_match = re.match('\d+\s+\w+, >([\w_]+)\.\.\. ', line)
            if id_match:
                clusters_dict[id_match[1]] = cluster_index
    return clusters_dict


def cluster_multiple_cdhit(csv_filename, m=100):
    sequence_records = generate_cdr_sequence_records(csv_filename)
    clusters_dicts = []
    for i in range(m):
        random.shuffle(sequence_records)
        temp_fasta_file = "shuffled.fasta"
        SeqIO.write(sequence_records, temp_fasta_file, "fasta")
        temp_cluster_output = "cluster_output"
        full_cmd = "/home/kcn25/tools/cd-hit-v4.8.1-2019-0228/cd-hit" \
                   " -i {} -o {} -c 0.4 -n 2 -M 16000" \
                   " -d 0 -T 8 -l 3".format(temp_fasta_file,
                                            temp_cluster_output)
        subprocess.run(full_cmd.split(" "))
        clusters_dict = read_cdhit_clusters(temp_cluster_output + ".clstr")
        clusters_dicts.append(clusters_dict)

    cluster_df = pd.DataFrame(clusters_dicts, dtype=int)

    ar_scores = []
    for i in cluster_df.index:
        for j in range(i):
            ar_scores.append(adjusted_rand_score(cluster_df.loc[i, :],
                                                 cluster_df.loc[j, :]))

    return cluster_df, ar_scores


def find_consensus_matrix(cluster_df):
    consensus_df = pd.DataFrame(np.zeros([cluster_df.shape[1],
                                          cluster_df.shape[1]],
                                         np.int32),
                                index=cluster_df.columns,
                                columns=cluster_df.columns)
    for cluster_ind in cluster_df.index:
        for row in consensus_df.index:
            for col in consensus_df.columns:
                if cluster_df.loc[cluster_ind, row] == cluster_df.loc[cluster_ind, col]:
                    consensus_df.loc[row, col] += 1
    return consensus_df


def investigate_robustness_cdhit(csv_filename, m):
    cluster_df, ar_scores = cluster_multiple_cdhit(csv_filename, m)
    columns = random.sample(list(cluster_df.columns), k=100)
    consensus = find_consensus_matrix(cluster_df.loc[:,columns])

    sns.distplot(ar_scores)
    plt.savefig("cluster_ars.png")
    plt.show()

    sns.clustermap(consensus)
    plt.savefig("consensus.png")
    plt.show()
