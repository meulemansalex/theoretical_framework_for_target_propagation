#!/usr/bin/env python3
# Copyright 2019 Alexander Meulemans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a script to run a certain hyper parameter setting for various random
seeds, to test random seed robustness
"""

import importlib
import argparse
import os
import sys
import numpy as np
import main
import pandas as pd


def _override_cmd_arg(config, fixed_space):
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)
    for k, v in fixed_space.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)

def run_training(config, config_fixed={}):
    """ Run the mainfile with the given config file and save the results of the
    alignment angles in the angle_dict"""

    _override_cmd_arg(config, config_fixed)
    summary = main.run()

    return summary

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str,
                        default='logs/seed_robustness/',
                        help='directory where the results will be saved.')
    parser.add_argument('--name', type=str, default='default',
                        help='Name of the file in which the results will be '
                             'saved')
    parser.add_argument('--config_module', type=str,
                        default='figure_scripts.config_toy_examples',
                        help='The name of the module containing the configs.')
    parser.add_argument('--nb_seeds', type=int, default=20,
                        help='number of random seeds to run the hp config on.')
    parser.add_argument('--regression', action='store_true',
                        help='Flag indicating that it is a regression and not '
                             'a classification. Hence, no accuracies will be '
                             'saved.')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    config_module = importlib.import_module(args.config_module)

    if args.regression:
        columns = ['loss_test_val_best',
                   'loss_train_val_best',
                   'loss_val_best',
                   'loss_test_best',
                   'loss_train_last',
                   'loss_train_best',
                   'epoch_best_loss',
                   ]
    else:
        columns = ['acc_test_val_best',
                   'acc_test_last',
                   'acc_train_val_best',
                   'acc_val_best',
                   'acc_test_best',
                   'acc_train_last',
                   'acc_train_best',
                   'acc_val_last',
                   'loss_test_val_best',
                   'loss_train_val_best',
                   'loss_val_best',
                   'loss_test_best',
                   'loss_train_last',
                   'loss_train_best',
                   'epoch_best_loss',
                   'epoch_best_acc',
                   ]
    index = [i for i in range(args.nb_seeds)]

    results = pd.DataFrame(index=index, columns=columns)

    random_seeds = np.random.randint(0, 10000, args.nb_seeds)

    filename = args.name
    if filename[-4:] != '.csv':
        filename += '.csv'

    for i, seed in enumerate(random_seeds):
        print('Initiating run {} ...'.format(i))
        config_module.config['random_seed'] = seed
        summary = run_training(config_module.config)
        for key in columns:
            results.at[i, key] = summary[key]
        results.to_csv(os.path.join(args.out_dir, filename))

    means = results.mean(axis=0)
    stds = results.std(axis=0)

    results.loc['mean'] = means
    results.loc['std'] = stds


    results.to_csv(os.path.join(args.out_dir, filename))

if __name__ == '__main__':
    run()
    

