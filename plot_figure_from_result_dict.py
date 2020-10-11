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

import importlib
from figure_scripts import figure_utils
import argparse
import os
import sys
import pandas as pd
import pickle
import matplotlib



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str,
                        default=None,
                        help='directory where the figures will be saved. '
                             'Default is equal to --result_dir')
    parser.add_argument('--result_dir', type=str,
                        default='logs/figures/default',
                        help='directory where the result dictionary is '
                             'situated.')
    parser.add_argument('--config_module', type=str,
                        default='figure_scripts.config_toy_examples',
                        help='The name of the module containing the configs.')
    parser.add_argument('--show_plots', action='store_true',
                        help='Should the plots be shown or only stored?')
    parser.add_argument('--smooth', type=int, default=30,
                        help='Smoothing window applied to the angle data.')
    parser.add_argument('--show_title', action='store_true',
                        help='display title on top of the figures')

    args = parser.parse_args()
    if not os.path.exists(args.result_dir):
        raise ValueError('provided result_dir does not exist')
    if args.out_dir is None:
        args.out_dir = args.result_dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    config_module = importlib.import_module(args.config_module)

    config_collection = config_module.config_collection
    config_fixed = config_module.config_fixed
    result_keys = config_module.result_keys


    result_dir = os.path.join(args.result_dir, 'result_dict')
    result_dict = figure_utils.read_result_dict(result_dir)

    # make the plots
    for result_key in result_keys:
        if 'angle' in result_key:
            xlabel = 'iteration'
            ylabel = figure_utils.make_ylabel(result_key)
        else:
            xlabel = 'epoch'
            ylabel = result_key

        result_key_dict = result_dict[result_key]

        figure_utils.make_plot(result_key_dict=result_key_dict,
                               result_key=result_key,
                               title=result_key,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               out_dir=args.out_dir,
                               save=True,
                               show=args.show_plots,
                               fancyplot=True,
                               smooth=args.smooth,
                               log_interval=config_fixed['log_interval'],
                               no_title=not args.show_title)

if __name__ == '__main__':
    run()
