
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def merge_replicates_in_list(replicate_list, replicate_char='_'):
    new_list_dict = {}
    for name in replicate_list:
        char_location = -(name[::-1].find(replicate_char) + 1)
        new_name = name[:char_location]
        new_list_dict[new_name] = 1

    new_list = list(new_list_dict.keys())
    return new_list


def merge_replicants_in_dataframe(replicate_df, replicate_char='_'):
    new_df = pd.DataFrame()
    repeat_am = {}
    for row in replicate_df.index:
        char_location = -(row[::-1].find(replicate_char) + 1)
        row_new = row[:char_location]
        if row_new not in new_df.index:
            new_df = new_df.append(replicate_df.loc[row])
            new_df = new_df.rename(index={row: row_new})
            repeat_am[row_new] = 1
        else:
            new_df.loc[row_new] *= repeat_am[row_new]
            new_df.loc[row_new] += replicate_df.loc[row]
            new_df.loc[row_new] /= repeat_am[row_new]+1
            repeat_am[row_new] += 1

    return new_df


def merge_replicants_in_seq(seq_file, output_file, replicate_char='_'):
    seq_dict = SeqIO.to_dict(SeqIO.parse(seq_file, "fasta"))
    seq_df = pd.DataFrame.from_dict(seq_dict).transpose()

    new_list = []
    new_dict = {}
    for row in seq_df.index:
        char_location = -(row[::-1].find(replicate_char) + 1)
        row_new = row[:char_location]
        if row_new not in list(new_dict.keys()):
            seq = SeqRecord(Seq(''.join(list(seq_df.loc[row]))), id=row_new, name=row_new, description=row_new)
            new_list.append(seq)
            new_dict[row_new] = 1

    SeqIO.write(new_list, output_file, 'fasta')

seq_file = '/home/user/DEPP/cur_data/processed_data/query_seq.fa'
output = '/home/user/DEPP/cur_data/processed_data/query_seq_merged.fa'
merge_replicants_in_seq(seq_file, output)

