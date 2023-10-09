#!/home/user/packages/anaconda3/envs/depp_env/bin/python
import os

import numpy as np
import pandas as pd
import torch

import treeswift
from Bio import SeqIO
import subprocess

from depp import utils

def create_baselines_from_seq(data_dir, output_dir):
    processed_dir = os.path.join(data_dir, 'processed_data')
    seq_file = processed_dir + '/seq.fa'
    seq = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))

    seq_len = len(seq[list(seq.keys())[0]])
    dist_dict = {}
    for row in seq.keys():
        dist_dict[row] = {}
        for col in seq.keys():
            if col in dist_dict.keys() and row in dist_dict[col].keys():
                dist_dict[row][col] = dist_dict[col][row]
            else:
                dist_dict[row][col] = sum(seq[row][i] != seq[col][i] for i in range(len(seq[row])))

            # find diistance

    dist_df = pd.DataFrame.from_dict(dist_dict)
    seq_labels = list(np.loadtxt(processed_dir + '/seq_label.txt', dtype=str))
    dist_df = dist_df.reindex(seq_labels, axis=0).reindex(seq_labels, axis=1)

    dist_df = dist_df/seq_len
    dist_df_jc = (-3/4) * np.log(1- (4/3) * dist_df)


    query_labels = np.loadtxt(processed_dir + '/query_label.txt', dtype=str)
    backbone_labels = np.loadtxt(processed_dir + '/backbone_label.txt', dtype=str)
    dist_filtered_df = dist_df.filter(query_labels,axis=0).filter(backbone_labels, axis=1)
    dist_filtered_df_jc = dist_df_jc.filter(query_labels,axis=0).filter(backbone_labels, axis=1)

    dist_filtered_df.to_csv(output_dir + '/hamming.csv', sep='\t')
    dist_filtered_df_jc.to_csv(output_dir + '/jc.csv', sep='\t')


    print('\thamming/jc completed')

def create_baselines_from_tree(data_dir, output_dir):
    processed_dir = os.path.join(data_dir, 'processed_data')
    tree_dir = processed_dir + '/true_tree.nwk'
    tree = treeswift.read_tree_newick(tree_dir)
    dist = tree.distance_matrix()

    dist_df = pd.DataFrame.from_dict(dist)

    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    dist_df = dist_df.fillna(0)
    seq_labels = list(np.loadtxt(processed_dir + '/seq_label.txt', dtype=str))
    dist_df = dist_df.reindex(seq_labels, axis=0).reindex(seq_labels, axis=1)

    query_labels = np.loadtxt(processed_dir + '/query_label.txt', dtype=str)
    backbone_labels = np.loadtxt(processed_dir + '/backbone_label.txt', dtype=str)
    dist_filtered_df = dist_df.filter(query_labels, axis=0).filter(backbone_labels, axis=1)

    dist_filtered_df.to_csv(output_dir + '/true_tree.csv', sep='\t')


    print('\ttree completed')

def create_distances_from_model(data_dir, output_dir, scale, verbose=True):
    models_dir = data_dir + '/models/'
    backbone_seq = data_dir + '/processed_data/backbone_seq.fa'
    query_seq = data_dir + '/processed_data/query_seq.fa'

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

                dist_df = pd.read_csv(os.path.join(output_dir,'depp.csv'), sep='\t').set_index('Unnamed: 0') / scale
                dist_df.index.name = ''
                # print(dist_df.index.name)
                # print(dist_df.columns)
                dist_df.to_csv(os.path.join(output_type_dir, model[:-5]+'.csv'), sep='\t')

                os.remove(os.path.join(output_dir,'depp.csv'))
                # os.rename(os.path.join(output_dir,'depp.csv'), os.path.join(output_type_dir, model[:-5]+'.csv'))
    if os.path.exists(output_dir + '/backbone_embeddings.pt'):
        os.remove(output_dir + '/backbone_embeddings.pt')
    if os.path.exists(output_dir + '/backbone_names.pt'):
        os.remove(output_dir + '/backbone_names.pt')

def evaluate_distances(distance_dir, run_amount=1, verbose=True):
    results_dict = {}
    baseline_dir = os.path.join(distance_dir,'baselines')
    for baseline in os.listdir(baseline_dir):
        results_dict[baseline] = {}
        depp_dir = os.path.join(distance_dir, 'depp')

        for depp_type in os.listdir(depp_dir):
            if depp_type != 'training' and baseline != 'training':
                depp_mat_dir = os.path.join(depp_dir, depp_type)
                # print(depp_dir)
                # print(depp_type)
                # print(depp_mat_dir)
                # print(os.listdir(depp_mat_dir))
                for depp_dist in os.listdir(depp_mat_dir):
                    if not verbose:
                        print('\t' + depp_dist)
                    baseline_mat = np.genfromtxt(baseline_dir + '/' + baseline, delimiter='\t')[1:,1:]
                    depp_mat = np.genfromtxt(depp_mat_dir + '/' + depp_dist, delimiter='\t')[1:,1:]

                    dist = utils.mse_loss(torch.from_numpy(depp_mat), torch.from_numpy(baseline_mat), 'square_root_be')
                    results_dict[baseline][depp_dist] = float(dist)

    results_df = pd.DataFrame.from_dict(results_dict)
    output_file_raw = distance_dir + '/results_raw.csv'
    results_df.to_csv(output_file_raw, sep='\t')

    results_summed_df = pd.DataFrame()
    for row in results_df.index:
        hyphen_location = -(row[::-1].find('-')+1)
        row_new = row[:hyphen_location]
        if row_new not in results_summed_df.index:
            results_summed_df = results_summed_df.append(results_df.loc[row])
            results_summed_df = results_summed_df.rename(index={row:row_new})
        else:
            results_summed_df.loc[row_new] += results_df.loc[row]

    results_averaged_df = results_summed_df/run_amount
    output_file = distance_dir + '/results_avg.csv'
    results_averaged_df.to_csv(output_file, sep='\t')










