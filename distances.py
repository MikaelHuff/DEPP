#!/home/user/packages/anaconda3/envs/depp_env/bin/python
import os

import numpy as np
import pandas as pd
import torch

import treeswift
from Bio import SeqIO
import subprocess

from depp import utils
import merge_replicants as merge


def create_baselines_from_dist(data_dir, output_dir):
    processed_dir = os.path.join(data_dir, 'processed_data')
    dist_df_ham = pd.read_csv(processed_dir + '/hamming_full.csv', sep='\t').set_index('Unnamed: 0')
    # dist_df_jc = pd.read_csv(processed_dir + '/jc_full.csv', sep='\t').set_index('Unnamed: 0')
    dist_df_ham.index = dist_df_ham.index.astype(str).rename('')
    # dist_df_jc.index = dist_df_jc.index.astype(str).rename('')

    dist_df_ham_merged = merge.merge_replicants_in_dataframe(dist_df_ham).transpose()
    dist_df_ham_merged = merge.merge_replicants_in_dataframe(dist_df_ham_merged).transpose()


    # - 3 / 4 * np.log(1 - 4 / 3 * hamming_dist)
    # seq_labels = list(np.loadtxt(processed_dir + '/seq_label.txt', dtype=str))
    query_labels = np.loadtxt(processed_dir + '/query_label.txt', dtype=str)
    backbone_labels = np.loadtxt(processed_dir + '/backbone_label.txt', dtype=str)
    dist_filtered_df_ham = dist_df_ham_merged.reindex(query_labels, axis=0).reindex(backbone_labels, axis=1)
    dist_filtered_df_jc = - 3 / 4 * np.log(1 - 4 / 3 * dist_filtered_df_ham)
    # dist_df_jc = dist_df_jc.reindex(seq_labels, axis=0).reindex(seq_labels, axis=1)


    # dist_filtered_df_ham = dist_df_ham.filter(query_labels,axis=0).filter(backbone_labels, axis=1)
    # dist_filtered_df_jc = dist_df_jc.filter(query_labels,axis=0).filter(backbone_labels, axis=1)

    dist_filtered_df_ham.to_csv(output_dir + '/hamming.csv', sep='\t')
    dist_filtered_df_jc.to_csv(output_dir + '/jc.csv', sep='\t')

    print('\thamming/jc completed')


def find_and_scale_tree(data_dir, output_dir, scale=1, verbose=True):
    processed_dir = os.path.join(data_dir, 'processed_data')

    tree_file = ''
    for file in os.listdir(data_dir):
        if 'tree' in file:
            tree_file = data_dir + '/' + file
    if tree_file == '':
        for file in os.listdir(os.path.dirname(data_dir)):
            if 'tree' in file:
                tree_file = data_dir + '/' + file

    tree = treeswift.read_tree_newick(tree_file)
    num_nodes = tree.num_nodes(internal=False)
    dist_dir = output_dir + '/jc.csv'

    seq_labels = list(np.loadtxt(processed_dir + '/seq_label.txt', dtype=str))
    if num_nodes < len(seq_labels):
        jplace_file = processed_dir + '/true_tree.jplace'
        command = ['run_apples.py',
                   '-d', dist_dir,
                   '-t', tree_file,
                   '-o', jplace_file
                   ]

        if not verbose:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(command)

        command2 = ['gappa', 'examine', 'graft',
                   '--jplace-path', jplace_file,
                   '--out-dir', processed_dir,
                   '--allow-file-overwriting', '--fully-resolve'
                   ]

        if not verbose:
            subprocess.run(command2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(command2)
        tree_true = treeswift.read_tree_newick(processed_dir + '/true_tree.newick')
    else:
        tree_true = tree
        tree_true.write_tree_newick(processed_dir + '/true_tree.newick')

    backbone_file = processed_dir + '/backbone_label.txt'
    backbone_labels = np.loadtxt(backbone_file, dtype=str)
    tree_backbone = tree_true.extract_tree_with(backbone_labels)
    tree_backbone.write_tree_newick(processed_dir + '/backbone_tree.newick')

    tree_scaled = tree_backbone
    tree_scaled.scale_edges(scale)
    tree_scaled.write_tree_newick(processed_dir + '/scaled_tree.newick')


def create_baselines_from_tree(data_dir, output_dir):
    processed_dir = os.path.join(data_dir, 'processed_data')
    tree_dir = processed_dir + '/true_tree.newick'
    tree = treeswift.read_tree_newick(tree_dir)

    seq_labels = list(np.loadtxt(processed_dir + '/seq_label.txt', dtype=str))
    query_labels = list(np.loadtxt(processed_dir + '/query_label.txt', dtype=str))
    backbone_labels = list(np.loadtxt(processed_dir + '/backbone_label.txt', dtype=str))
    num_nodes = tree.num_nodes(internal=False)
    if num_nodes < len(seq_labels):
        print('\tNot a complete tree file')
        if num_nodes == len(backbone_labels):
            print('\tOnly has backbone labels')
        print('\tExiting tree baseline')
        assert num_nodes == len(seq_labels),'Not a complete tree file'

    dist = tree.distance_matrix()

    dist_df = pd.DataFrame.from_dict(dist)

    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    dist_df = dist_df.fillna(0)

    dist_filtered_df = dist_df.reindex(query_labels, axis=0).reindex(backbone_labels, axis=1)
    # dist_filtered_df = dist_df.filter(query_labels, axis=0).filter(backbone_labels, axis=1)

    dist_filtered_df.to_csv(output_dir + '/true_tree.csv', sep='\t')


    print('\ttree completed')

def create_distances_from_model(data_dir, output_dir, scale, verbose=True):
    models_dir = os.path.join(data_dir, 'models')
    processed_dir = os.path.join(data_dir, 'processed_data')
    backbone_seq = processed_dir + '/backbone_seq.fa'
    query_seq = processed_dir + '/query_seq.fa'

    for model_type in os.listdir(models_dir):
        if model_type != 'log.csv':
            type_path = os.path.join(models_dir,model_type)
            output_type_dir = os.path.join(output_dir, model_type)
            if not os.path.exists(output_type_dir):
                os.mkdir(output_type_dir)
            for model in os.listdir(type_path):
                print('\t' + model)
                model_dir = os.path.join(type_path,model)

                command = ['depp_distance.py',
                           'backbone_seq_file=' + backbone_seq,
                           'query_seq_file=' + query_seq,
                           'model_path=' + model_dir,
                           'outdir=' + output_dir]

                if not verbose:
                    subprocess.run(command, stdout=subprocess.DEVNULL)
                else:
                    subprocess.run(command)



                dist_df = pd.read_csv(os.path.join(output_dir,'depp.csv'), sep='\t').set_index('Unnamed: 0')/scale
                dist_df = merge.merge_replicants_in_dataframe(dist_df, '_')

                dist_df.index = dist_df.index.astype(str).rename('')
                dist_df.columns = dist_df.columns.astype(str)
                query_labels = np.loadtxt(processed_dir + '/query_label.txt', dtype=str)
                backbone_labels = np.loadtxt(processed_dir + '/backbone_label.txt', dtype=str)
                dist_sorted_df = dist_df.reindex(query_labels, axis=0).reindex(backbone_labels, axis=1)

                dist_sorted_df.to_csv(os.path.join(output_type_dir, model[:-5]+'.csv'), sep='\t')

                os.remove(os.path.join(output_dir,'depp.csv'))
    if os.path.exists(output_dir + '/backbone_embeddings.pt'):
        os.remove(output_dir + '/backbone_embeddings.pt')
    if os.path.exists(output_dir + '/backbone_names.pt'):
        os.remove(output_dir + '/backbone_names.pt')

def evaluate_distances(distance_dir, verbose=True):
    results_dict = {}
    baseline_dir = os.path.join(distance_dir,'baselines')
    for baseline in os.listdir(baseline_dir):
        if verbose:
            print('\t' + baseline)
        results_dict[baseline] = {}
        depp_dir = os.path.join(distance_dir, 'depp')
        if 'training' not in baseline:
            baseline_mat = np.genfromtxt(baseline_dir + '/' + baseline, delimiter='\t')[1:,1:]
            for depp_type in os.listdir(depp_dir):
                if 'training' not in depp_type:
                    depp_mat_dir = os.path.join(depp_dir, depp_type)
                    for depp_dist in os.listdir(depp_mat_dir):
                        if verbose:
                            print('\t\t' + depp_dist)
                        depp_mat = np.genfromtxt(depp_mat_dir + '/' + depp_dist, delimiter='\t')[1:,1:]

                        dist = utils.mse_loss(torch.from_numpy(depp_mat), torch.from_numpy(baseline_mat), 'square_root_be')
                        results_dict[baseline][depp_dist] = float(dist)

    results_df = pd.DataFrame.from_dict(results_dict)
    output_file_raw = distance_dir + '/results_raw.csv'
    results_df.to_csv(output_file_raw, sep='\t')

    results_merged_df = merge.merge_replicants_in_dataframe(results_df, '-')
    output_file = distance_dir + '/results_avg.csv'
    results_merged_df.to_csv(output_file, sep='\t')











