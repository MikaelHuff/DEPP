
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_ecdfs(placement_dir):

    def average_and_get_data(dir):
        error_dict = {}
        for file in os.listdir(dir):
            model_type = file[0:4]
            df = pd.read_csv(dir + '/' + file, sep=' ')
            df = df.set_index('Query').sort_index()
            np_arr = df.to_numpy()
            err_arr = np_arr[:,1]
            if model_type not in error_dict.keys():
                error_dict[model_type] = err_arr/repeat_am
            else:
                error_dict[model_type] = error_dict[model_type] + err_arr/repeat_am
        return error_dict


    repeat_am = len(os.listdir(placement_dir))/4
    error_dict = average_and_get_data(placement_dir)
    query_am = len(error_dict[list(error_dict.keys())[0]])
    print(query_am)
    for model_type in error_dict.keys():
        if model_type in ['dual', 'full']:
            x = np.sort(error_dict[model_type])
            y = np.arange(len(x))/float(len(x))
            plt.plot(x,y, label=model_type)
        pass
    plt.legend()
    plt.show()




if __name__ == '__main__':
    placement_dir = '/home/user/DEPP/cur_data/results/placement_evaluations'
    find_ecdfs(placement_dir)