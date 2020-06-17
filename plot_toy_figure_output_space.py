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

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from lib import utils
import matplotlib.patches as mpatches
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str,
                        default='./logs/output_space_figure',
                        help='Directory with the result data from the output '
                             'space toy experiment.')

    args = parser.parse_args()
    logdir = args.logdir
    fontsize = 22


    methods = [
        'GN2',
        'BP',
    ]

    legend_labels = [
        'GNT',
        'BP',
    ]

    colours = [
        'b',
        'r',
    ]

    output_arrows = {}
    output_arrows_start = {}
    targets = {}
    for key in methods:
        filename1 = 'output_arrow_' + key + '.npy'
        filename2 = 'output_arrow_start_' + key + '.npy'
        filename3 = 'output_space_label_' + key + '.npy'
        output_arrows[key] = np.load(os.path.join(logdir, filename1))
        output_arrows_start[key] = np.load(os.path.join(logdir, filename2))
        targets[key] = np.load(os.path.join(logdir, filename3))

    y = output_arrows_start[methods[0]]
    y = torch.Tensor(y)
    target = targets[methods[0]]
    target = torch.Tensor(target)
    loss_function = torch.nn.MSELoss()

    ax = plt.axes()
    utils.plot_contours(y, target, loss_function, ax)

    for i, key in enumerate(methods):
        ax.arrow(output_arrows_start[key][0], output_arrows_start[key][1],
                  output_arrows[key][0], output_arrows[key][1],
                  width=0.08,
                  head_width=0.4,
                 ec=colours[i],
                 fc=colours[i]
                  )

    ax.plot(output_arrows_start[methods[0]][0], output_arrows_start[methods[0]][1], '*')
    # dimensions
    output_start = y
    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              target.detach().cpu().numpy())
    x_low = target[0].detach().cpu().numpy() - 1.1 * distance
    x_high = target[0].detach().cpu().numpy() + 1.1 * distance
    y_low = target[1].detach().cpu().numpy() - 1.1 * distance
    y_high = target[1].detach().cpu().numpy() + 1.1 * distance
    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    # remove axis numbers
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tick_params(
        axis='both',          # changes apply to the x and y axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False)

    ax.set_aspect('equal', 'box')
    # ax.axis('off')
    # ax labels
    ax.set_ylabel(r'$y_2$', fontsize=fontsize)
    ax.set_xlabel(r'$y_1$', fontsize=fontsize)

    # create ledgend
    patches = []
    for i, key in enumerate(legend_labels):
        colour = colours[i]
        patches.append(mpatches.Patch(color=colour, label=key))

    plt.legend(handles=patches, fontsize=fontsize, loc='lower right')


    file_name = 'output_space_updates_fig_BP_GN.pdf'
    plt.savefig(os.path.join(logdir, file_name))
    plt.close()


if __name__ == '__main__':
    run()




