# Copyright 2020 Alexander Meulemans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import seaborn as sns
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str,
                        default='logs/figure_toy_nullspace_frozen',
                        help='Directory with the result data from the nullspace'
                             'experiment.')

    args = parser.parse_args()
    result_dir = args.result_dir

    filename = os.path.join(result_dir, 'result_dict.pickle')
    with open(filename, 'rb') as f:
        result_dict = pickle.load(f)

    legend = ['DTP', 'DDTP-linear \n (ours)']
    # legend = ['DTP', 'Ours']
    layer_idx = 1


    result_dict_null = result_dict['nullspace_relative_norm_angles']
    result_dict_null_DTP = result_dict_null['DTP_pretrained']
    result_dict_null_DDTPlin = result_dict_null['DMLPDTP2_linear']


    # append the dataframes with labels
    result_dict_null_DTP.insert(result_dict_null_DTP.shape[1], 'type',
                                    [legend[0] for i in
                                     range(result_dict_null_DTP.shape[0])])

    result_dict_null_DDTPlin.insert(result_dict_null_DDTPlin.shape[1], 'type',
                                    [legend[1] for i in
                                     range(result_dict_null_DDTPlin.shape[0])])
    result_dict_joined = pd.concat([result_dict_null_DTP, result_dict_null_DDTPlin])
    # result_dict_joined.rename(columns={layer_idx, r''})

    sns.set(style="ticks", font_scale=1.3, rc={'figure.figsize':(3,3)})
    ax = sns.barplot(x="type", y=layer_idx, data=result_dict_joined, ci='sd')
    ax.set_xlabel('')
    ax.set_ylabel(r'$\|\|\Delta W_%i^{null}\|\|/\|\|\Delta W_%i\|\|$' % (layer_idx+1,
                  layer_idx+1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(axis='x', length=0)
    sns.despine()
    plt.savefig(os.path.join(result_dir, 'barplot_nullspace.pdf'),
                bbox_inches='tight', dpi=400)

if __name__ == '__main__':
    run()
