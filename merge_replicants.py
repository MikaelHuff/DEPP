
import pandas as pd

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