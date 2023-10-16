#!/home/user/packages/anaconda3/envs/depp_env/bin/python
import os
import pandas as pd
import subprocess

import merge_replicants as merge

def create_placements(data_dir, output_dir, verbose=True):
    dist_dir = os.path.join(data_dir, 'distances', 'depp')
    apple_dir = os.path.join(output_dir, 'apples')

    if not os.path.exists(apple_dir):
        os.mkdir(apple_dir)

    placement_tree_dir = os.path.join(output_dir, 'placement_trees')
    if not os.path.exists(placement_tree_dir):
        os.mkdir(placement_tree_dir)

    tree_file = os.path.join(data_dir, 'processed_data') + '/backbone_tree.newick'
    for dist_type in os.listdir(dist_dir):
        dist_type_dir = os.path.join(dist_dir, dist_type)
        if os.path.isdir(dist_type_dir):
            for dist_mat in os.listdir(dist_type_dir):
                print('\t' + dist_mat)

                jplace_file = apple_dir + '/' + dist_mat[:-4] + '.jplace'
                command = ['run_apples.py',
                           '-d', dist_type_dir + '/' + dist_mat,
                           '-t', tree_file,
                           '-o', jplace_file
                ]
                #command = ['run_apples.py', '-d', dist_type_dir + '/' + dist_mat, '-t', tree_file, '-o', jplace_file]

                if not verbose:
                    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(command)

                command = ['gappa', 'examine', 'graft',
                           '--jplace-path', jplace_file,
                           '--out-dir', placement_tree_dir,
                           '--allow-file-overwriting', '--fully-resolve'
                ]
                if not verbose:
                    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(command)



def evaluate_placements(data_dir, placement_dir, run_amount=1):
    tree_dir = os.path.join(placement_dir, 'placement_trees')
    processed_dir = os.path.join(data_dir, 'processed_data')
    true_tree_file = processed_dir + '/true_tree.newick'
    # query_file = processed_dir + '/query_label.txt'
    query_file = processed_dir + '/query_seq.fa'
    backbone_tree_file = processed_dir + '/backbone_tree.newick'

    script_file = os.getcwd() + '/evaluate_placement.sh'
    output_dir = os.path.join(placement_dir, 'evaluations')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for tree_file in os.listdir(tree_dir):
        print('\t' + tree_file)
        tree_file_full = tree_dir + '/' + tree_file
        command = ['bash',
                   script_file,
                   true_tree_file,
                   tree_file_full,
                   query_file,
                   backbone_tree_file
        ]
        output_file = output_dir + '/' + tree_file[:-7] + '.csv'
        with open(output_file, 'w') as f:
            subprocess.run(command, stdout=f)
        with open(output_file, 'r+') as f:
            content = f.read()
            f.seek(0,0)
            f.write('Query Bipartition Error Error/Bipartition\n' + content)


    results_file = placement_dir + '/results_raw.csv'
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, 'x') as f:
        f.write('Model\tError Amount\tError/Bipartition\n')
        for file in os.listdir(output_dir):
            result_df = pd.read_csv(output_dir + '/' + file, sep=' ')
            error = list(result_df.sum(axis=0))
            f.write(file[:-4]+ '\t' + str(error[2]) + '\t' + str(error[3]) + '\n')

    results_df = pd.read_csv(results_file, sep='\t').set_index('Model')

    results_merged_df = merge.merge_replicants_in_dataframe(results_df, '-')
    results_merged_df.index.name = 'Models'
    results_merged_df.to_csv(placement_dir + '/results_avg.csv', sep='\t')