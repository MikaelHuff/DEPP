import os
import sys

import train_depp


def train(args, data_dir, model_type='full', amount=1, base_models=None, log=sys.stdout, verbose=True):
    if base_models == None:
        base_models = [None] * amount

    model_dir = data_dir + '/models/'
    output_dir = model_dir + model_type
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    args.model_dir = model_dir
    args.backbone_tree_file = data_dir + '/processed_data/scaled_tree.newick'
    args.backbone_seq_file = data_dir + '/processed_data/backbone_seq.fa'

    models = []
    for run in range(amount):
        print('\t' + model_type + '-' + str(run))
        if not verbose:
            void = open(os.devnull, 'w')
            sys.stdout = void
        epoch, final_loss, model = train_depp.main(args, model_type, prev_model=base_models[run])
        sys.stdout = sys.__stdout__

        new_file_name = model_type + '-' + str(run) + '.ckpt'
        for file in os.listdir(model_dir):
            if file[-4:] == 'ckpt':
                os.rename(os.path.join(model_dir,file),os.path.join(output_dir,new_file_name))
                break
        log.write(new_file_name[:-5] + '\t' + str(epoch) + '\t' + str(final_loss) + '\n')

        if model_type == 'base':
            models.append(model.encoder.linear)

        for file in os.listdir(model_dir):
            if file[-3:] == 'pth':
                os.remove(os.path.join(model_dir,file))

    return models

