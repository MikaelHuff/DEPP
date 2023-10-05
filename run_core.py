#!/home/user/packages/anaconda3/envs/depp_env/bin/python
import os
import sys
import shutil
from omegaconf import OmegaConf

import data_prep
import distances
import train_models
import placements

from depp import default_config


def run_all(args, selection, training=False):

    if not training:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir,'training')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

    tree_scaling = args.tree_scaling
    verbose = args.verbose
    #split sequences
    #find/scale tree
    if selection['prep_data']:
        output_dir = os.path.join(data_dir, 'processed_data')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not training:
            data_prep.group_data(data_dir, output_dir)
            data_prep.split_sequences(output_dir)
        else:
            old_data_dir = os.path.dirname(data_dir)
            data_prep.copy_data_to_training(old_data_dir, output_dir)

        data_prep.find_and_scale_tree(output_dir, scale=tree_scaling)
    print('data preperation completed')


    run_amount = args.run_amount
    # train models
    if selection['create_models']:
        model_dir = data_dir + '/models/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # base, full, dual, no-5mer
        log_dir = model_dir + 'log.csv'
        if os.path.exists(log_dir):
            os.remove(log_dir)

        with open(log_dir, 'x') as f:
            f.write('model name\tfinal epoch\tfinal loss\n')
            train_models.train(args, data_dir, 'full', amount=run_amount, log=f, verbose=verbose)
            train_models.train(args, data_dir, 'no5-mer', amount=run_amount, log=f, verbose=verbose)
            models = train_models.train(args, data_dir, 'base', amount=run_amount, log=f, verbose=verbose)
            train_models.train(args, data_dir, model_type='dual', amount=run_amount, base_models=models, log=f, verbose=verbose)
    print('depp models trained')


    distance_dir = data_dir + '/distances/'
    if not os.path.exists(distance_dir):
        os.mkdir(distance_dir)

    # create baselines
    if selection['create_base_distances']:
        output_dir = os.path.join(distance_dir, 'baselines')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        distances.create_baselines_from_seq(data_dir, output_dir)
        pass
        distances.create_baselines_from_tree(data_dir, output_dir)
    print('baseline distances created')



    #createt depp diisttances
    if selection['create_depp_distances']:
        output_dir = distance_dir + '/depp/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        distances.create_distances_from_model(data_dir, output_dir, tree_scaling, verbose=verbose)
    print('depp distances created')

    # evaluate disttances
    if selection['evaluate_distances']:
        distances.evaluate_distances(distance_dir, run_amount=run_amount, verbose=verbose)
        print('distances evaluated')

    placement_dir = os.path.join(data_dir, 'placements')
    # crreate placements
    if selection['create_placements']:
        if not os.path.exists(placement_dir):
            os.mkdir(placement_dir)
        placements.create_placements(data_dir, placement_dir, verbose=verbose)
        print('placements created')


    # evaluuate placements
    if selection['evaluate_placements']:
        placements.evaluate_placements(data_dir, placement_dir, run_amount=run_amount)
        print('placements evaluated')


    # gather results to one place
    if selection['compile_results']:
        result_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        shutil.copy(os.path.join(data_dir, 'models') + '/log.csv', result_dir + '/model_training_log.csv')

        shutil.copy(os.path.join(data_dir, 'distances') + '/results_raw.csv', result_dir + '/distance_evaluations_all.csv')
        shutil.copy(os.path.join(data_dir, 'distances') + '/results_avg.csv', result_dir + '/distance_evaluations_avg.csv')

        shutil.copy(os.path.join(data_dir, 'placements') + '/results_raw.csv', result_dir + '/placement_evaluations_all.csv')
        shutil.copy(os.path.join(data_dir, 'placements') + '/results_avg.csv', result_dir + '/placement_evaluations_avg.csv')
        print('results grouped in ' + result_dir)



def main():

    args_base = OmegaConf.create(default_config.default_config)
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_base, args_cli)

    # args.selection = [0,0, 0,0, 0, 0, 1, 1]

    selection = {
        'prep_data': args.selection[0],
        'create_models': args.selection[1],
        'create_base_distances': args.selection[2],
        'create_depp_distances': args.selection[3],
        'evaluate_distances': args.selection[4],
        'create_placements': args.selection[5],
        'evaluate_placements': args.selection[6],
        'compile_results': args.selection[7]
    }
    training = args.training

    run_all(args, selection)
    if training:
        selection['create_models'] = False
        selection['create_placements'] = False
        selection['evaluate_placements'] = False

        run_all(args, selection, training=True)


if __name__ == '__main__':
    main()

#run_core.py data_dir=/home/user/DEPP/simulated_data/hgt_data/rep.01/005/ gpus=0 patience=4 val_freq=10 min_delta=.1 run_amount=2 training=True selection=[1,1,1,1,1,1,1,1]
# dir=/depp_test/simulated_data/ils_data/model.200.500000.0.000001/06/1/
# python run_core.py data_dir=$dir gpus=0 patience=4 val_freq=10 min_delta=.1 run_amount=2 training=False selection=[1,1,1,1,1,1,1,1]
