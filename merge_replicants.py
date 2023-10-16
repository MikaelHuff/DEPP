


replicate_char = '_'
def merge_replicates_in_list(replicate_list):
    new_list_dict = {}
    for name in replicate_list:
        new_name = name[0:name.find(replicate_char)]
        new_list_dict[new_name] = 1

    new_list = list(new_list_dict.keys())
    return new_list


    pass