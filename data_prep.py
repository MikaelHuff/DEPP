
import os
import shutil
import numpy as np
from Bio import SeqIO
import dendropy
import treeswift


# seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
# args.sequence_length = len(list(seq.values())[0])
# tree = dendropy.Tree.get(path=backbone_tree_file, schema='newick')
def group_data(data_dir, output_dir):
    seq_file = data_dir + '/seq.fa'
    shutil.copy(seq_file, output_dir+'/seq.fa')

    seq = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))
    seq_labels = np.array(list(seq.keys()), dtype=str)
    np.savetxt(output_dir + '/seq_label.txt',seq_labels, delimiter='\n', fmt='%s')


    if 'query_label.txt' in os.listdir(data_dir):
        query_file = data_dir + '/query_label.txt'
    else:
        query_file = os.path.dirname(data_dir) + '/query_label.txt'
    queries = np.loadtxt(query_file, dtype=str)
    backbone_labels = [label for label in seq_labels if label not in queries]
    np.savetxt(output_dir + '/backbone_label.txt', backbone_labels, delimiter='\n', fmt='%s')
    shutil.copy(query_file,output_dir+'/query_label.txt')


    tree_file = ''
    for file in os.listdir(data_dir):
        if 'tree' in file and 'best' not in file:
            tree_file = data_dir + '/' + file
    if tree_file == '':
        for file in os.listdir(os.path.dirname(data_dir)):
            if 'tree' in file and 'best' not in file:
                tree_file = data_dir + '/' + file
    shutil.copy(tree_file,output_dir + '/true_tree.nwk')

def split_sequences(output_dir):
    seq_file = output_dir + '/seq.fa'
    seq = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))

    query_file = output_dir + '/query_label.txt'
    queries = np.loadtxt(query_file, dtype=str)

    backbone_seq = {}
    query_seq = {}
    for key in seq.keys():
        if key in queries:
            query_seq[key] = seq[key]
        else:
            backbone_seq[key] = seq[key]


    with open(output_dir + '/backbone_seq.fa', 'w') as handle:
        SeqIO.write(backbone_seq.values(), handle, 'fasta')

    with open(output_dir + '/query_seq.fa', 'w') as handle:
        SeqIO.write(query_seq.values(), handle, 'fasta')

def find_and_scale_tree(output_dir, scale=1):
    #tree_denropy = dendropy.Tree.get(path=tree_file, schema='newick')
    tree_file = output_dir + '/true_tree.nwk'
    tree = treeswift.read_tree_newick(tree_file)

    backbone_file = output_dir + '/backbone_label.txt'
    backbone = np.loadtxt(backbone_file, dtype=str)
    tree = tree.extract_tree_with(backbone)
    tree.write_tree_newick(output_dir + '/backbone_tree.nwk')

    tree.scale_edges(scale)
    tree.write_tree_newick(output_dir + '/scaled_tree.nwk')


def copy_data_to_training(data_dir, output_dir):

    old_data_dir = os.path.join(data_dir, 'processed_data')

    shutil.copy(old_data_dir + '/backbone_label.txt', output_dir + '/backbone_label.txt')
    shutil.copy(old_data_dir + '/backbone_seq.fa', output_dir + '/backbone_seq.fa')

    shutil.copy(old_data_dir + '/backbone_label.txt', output_dir + '/query_label.txt')
    shutil.copy(old_data_dir + '/backbone_seq.fa', output_dir + '/query_seq.fa')

    shutil.copy(old_data_dir + '/seq_label.txt', output_dir + '/seq_label.txt')
    shutil.copy(old_data_dir + '/seq.fa', output_dir + '/seq.fa')

    shutil.copy(old_data_dir + '/true_tree.nwk', output_dir + '/true_tree.nwk')

    shutil.copy(old_data_dir + '/hamming_full.csv', output_dir + '/hamming_full.csv')
    shutil.copy(old_data_dir + '/jc_full.csv', output_dir + '/jc_full.csv')


    model_dir_old = os.path.join(data_dir, 'models')
    model_dir_new = os.path.join(data_dir, 'training', 'models')
    if os.path.exists(model_dir_new):
        shutil.rmtree(model_dir_new)
    shutil.copytree(model_dir_old,model_dir_new)




