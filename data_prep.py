
import os
import shutil
import numpy as np
import pandas as pd
from Bio import SeqIO
import treeswift
import subprocess

from depp import utils
import merge_replicants as merge


def group_data(data_dir, output_dir, replicates):
    seq_file = data_dir + '/seq.fa'
    shutil.copy(seq_file, output_dir+'/seq.fa')


    seq = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))
    if not replicates:
        seq_labels = np.array(list(seq.keys()), dtype=str)
    else:
        seq_labels = np.array(merge.merge_replicates_in_list(list(seq.keys())))
    np.savetxt(output_dir + '/seq_label.txt',seq_labels, delimiter='\n', fmt='%s')

    if 'query_label.txt' in os.listdir(data_dir):
        query_file = data_dir + '/query_label.txt'
    else:
        query_file = os.path.dirname(data_dir) + '/query_label.txt'
    queries = np.loadtxt(query_file, dtype=str)
    backbone_labels = [label for label in seq_labels if label not in queries]
    np.savetxt(output_dir + '/backbone_label.txt', backbone_labels, delimiter='\n', fmt='%s')
    shutil.copy(query_file,output_dir+'/query_label.txt')


def split_sequences(output_dir):
    seq_file = output_dir + '/seq.fa'
    seq = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))

    query_file = output_dir + '/query_label.txt'
    query_labels = np.loadtxt(query_file, dtype=str)

    backbone_seq = {}
    query_seq = {}
    # seq_labels = merge.merge_replicates_in_list(list(seq.keys()))
    for key in seq.keys():
        key_merged = key[0:key.find('_')]
        if key_merged in query_labels:
            query_seq[key] = seq[key]
        else:
            backbone_seq[key] = seq[key]

    with open(output_dir + '/backbone_seq.fa', 'w') as handle:
        SeqIO.write(backbone_seq.values(), handle, 'fasta')

    with open(output_dir + '/query_seq.fa', 'w') as handle:
        SeqIO.write(query_seq.values(), handle, 'fasta')


def create_dist_from_seq(data_dir):
    processed_dir = data_dir
    seq_file = processed_dir + '/seq.fa'
    seq_dict = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))

    seq_len = len(seq_dict[list(seq_dict.keys())[0]])
    print('\t sequence amount is: ', len(seq_dict))
    print('\t sequences length is: ', seq_len)

    names = list(seq_dict.keys())
    raw_seqs = [np.array(seq_dict[k].seq).reshape(1, -1) for k in seq_dict]
    raw_seqs = np.concatenate(raw_seqs, axis=0)

    dist_df_ham, dist_df_jc = utils.jc_dist(raw_seqs, raw_seqs, names, names)
    dist_df_ham.to_csv(processed_dir + '/hamming_full.csv', sep='\t')
    dist_df_jc.to_csv(processed_dir + '/jc_full.csv', sep='\t')


def copy_data_to_training(data_dir, output_dir):

    old_data_dir = os.path.join(data_dir, 'processed_data')

    shutil.copy(old_data_dir + '/backbone_label.txt', output_dir + '/backbone_label.txt')
    shutil.copy(old_data_dir + '/backbone_seq.fa', output_dir + '/backbone_seq.fa')

    shutil.copy(old_data_dir + '/backbone_label.txt', output_dir + '/query_label.txt')
    shutil.copy(old_data_dir + '/backbone_seq.fa', output_dir + '/query_seq.fa')

    shutil.copy(old_data_dir + '/seq_label.txt', output_dir + '/seq_label.txt')
    shutil.copy(old_data_dir + '/seq.fa', output_dir + '/seq.fa')

    shutil.copy(old_data_dir + '/true_tree.newick', output_dir + '/true_tree.newick')

    shutil.copy(old_data_dir + '/hamming_full.csv', output_dir + '/hamming_full.csv')
    shutil.copy(old_data_dir + '/jc_full.csv', output_dir + '/jc_full.csv')


    model_dir_old = os.path.join(data_dir, 'models')
    model_dir_new = os.path.join(data_dir, 'training', 'models')
    if os.path.exists(model_dir_new):
        shutil.rmtree(model_dir_new)
    shutil.copytree(model_dir_old,model_dir_new)




