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

import main
import sys
import os
import json

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="ticks", font_scale=1.6)

def _override_cmd_arg(config, fixed_space):
    sys.argv = [sys.argv[0]]
    for k, v in fixed_space.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)
    for k, v in config.items():
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


def postprocess_summary(summary, name, result_dict, result_keys):
    """ Save the result_keys performances in the result_dict"""

    for key in result_keys:
        if key in summary.keys():
            result_dict[key][name] = summary[key]


def create_layerwise_result_dict(result_key_dict):
    """ Sort the result_key_dict (result dict of only one performance key)
    with layer numbers as keys and as values Dataframes with the training method
    names as column names."""
    layerwise_dict = {}
    nb_layers = len(result_key_dict[list(result_key_dict.keys())[0]].columns)
    for i in range(nb_layers):
        layerwise_dict[i] = pd.DataFrame()

    column_names = []
    for name, dtframe in result_key_dict.items():
        column_names.append(name)
        for i in range(nb_layers):
            layerwise_dict[i] = pd.concat([layerwise_dict[i], dtframe[i]],
                                          axis=1)
    return layerwise_dict, column_names


def make_plot(result_key_dict, result_key, title='plot', xlabel='x',
              ylabel=None, out_dir='logs/figures', save=False, show=True,
              fancyplot=True, smooth=30, log_interval=20, no_title=True):
    """ Make a plot for each layer, comparing the result_key for all the
    training methods used when creating the result_dict"""
    if ylabel is None:
        ylabel = result_key

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], np.ndarray):
        plt.figure()
        legend = []
        for name, array in result_key_dict.items():
            legend.append(name)
            plt.plot(array)
        plt.legend(legend)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()

        if save:
            file_name = title + '.pdf'
            plt.savefig(os.path.join(out_dir, file_name))
            plt.close()
        if show:
            plt.show()

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], pd.DataFrame):

        if fancyplot and "angle" in result_key:
            plot_rolling(result_key_dict=result_key_dict,
                                       result_key=result_key,
                                       title=result_key,
                                       xlabel=xlabel,
                                       ylabel=ylabel,
                                       out_dir=out_dir,
                                       save=True,
                                       show=show,
                                       smooth=smooth,
                                       log_interval=log_interval,
                                       no_title=no_title)
            return

        layerwise_dict, legend = create_layerwise_result_dict(result_key_dict)
        for idx, dtframe in layerwise_dict.items():
            plt.figure()
            figure_title = title + ' layer ' + str(idx)
            ax = dtframe.plot(title=figure_title, style='.')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.legend(legend)

            if save:
                file_name = title + '_layer' + str(idx) + '.pdf'
                plt.savefig(os.path.join(out_dir, file_name))
                plt.close()
            if show:
                plt.show()


def make_plot_smooth(result_key_dict, result_key, title='plot', xlabel='x',
              ylabel=None, out_dir='logs/figures', save=False, show=True, smooth=30, fancyplot=True):
    """
    Make a plot for each layer, comparing the result_key for all the
        training methods used when creating the result_dict. Smooth over a fixed window.
    Args:
        result_dict:
        result_key:
        title:
        xlabel:
        ylabel:
        out_dir:
        save:
        show:

    Returns:

    """
    if ylabel is None:
        ylabel = result_key

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], np.ndarray):
        plt.figure()
        legend = []
        for name, array in result_key_dict.items():
            legend.append(name)
            plt.plot(array)
        plt.legend(legend)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        sns.despine()

        if save:
            file_name = title + '.pdf'
            plt.savefig(os.path.join(out_dir, file_name))
        if show:
            plt.show()

    if isinstance(result_key_dict[list(result_key_dict.keys())[0]], pd.DataFrame):




        layerwise_dict, legend = create_layerwise_result_dict(result_key_dict)
        for idx, df in layerwise_dict.items():

            df.to_pickle(os.path.join(f"df{idx}.pkl"))
            df = pd.read_pickle(os.path.join(f"df{idx}.pkl"))

            ncols = len(df.columns)
            for col in range(ncols):
                df[f'roll_mean{col}'] = df.iloc[:, col].rolling(window=smooth, min_periods=1).mean()
            for col in range(ncols):
                df[f'roll_std{col}'] = df.iloc[:, col].rolling(window=smooth, min_periods=1).std()

            df_points = df.iloc[:, :ncols]
            df_roll_mean = df.iloc[:, ncols:2*ncols].T.reset_index(drop=True).T
            df_roll_std = df.iloc[:, 2*ncols:].T.reset_index(drop=True).T

            df_upper = df_roll_mean.add(df_roll_std, fill_value=0)
            df_lower = df_roll_mean.sub(df_roll_std, fill_value=0)

            plt.figure()
            figure_title = title + ' layer ' + str(idx+1)
            ax = df_points.plot(title=figure_title, style='.', colormap='viridis')
            df_roll_mean.plot(title=figure_title, style='-', ax=ax, colormap='viridis')
            df_upper.plot(title=figure_title, style='--', ax=ax, colormap='viridis')
            df_lower.plot(title=figure_title, style='--', ax=ax, colormap='viridis')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            plt.legend(legend)
            exit()

            if save:
                file_name = title + '_layer' + str(idx) + '.pdf'
                plt.savefig(os.path.join(out_dir, file_name))
            if show:
                plt.show()


def save_list(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write('%s\n' % item)

def save_dict_to_json(dictionary, file_name):
    with open(file_name, 'w') as json_file:
        json.dump(dictionary, json_file)


def read_list(filename):
    lst = []
    with open(filename, 'r') as f:
        for line in f:
            lst.append(line[:-1])
    return lst


def save_result_dict(result_dict, out_dir):
    keys = []
    names = []
    is_array = np.array([])

    for key, sub_dict in result_dict.items():
        keys.append(key)
        np.append(is_array, isinstance(sub_dict[list(sub_dict.keys())[0]],
                                       np.ndarray))
        for name, value in sub_dict.items():
            file_name = os.path.join(out_dir, key + '_' + name)
            if len(names) < len(sub_dict):
                names.append(name)
            if isinstance(value, np.ndarray):
                np.save(file_name + '.npy', value)
            if isinstance(value, pd.DataFrame):
                value.to_csv(file_name + '.csv')

    save_list(keys, os.path.join(out_dir, 'keys.txt'))
    save_list(names, os.path.join(out_dir, 'names.txt'))
    np.save(os.path.join(out_dir, 'is_array.npy'), is_array)


def read_result_dict(result_dir):
    keys = read_list(os.path.join(result_dir, 'keys.txt'))
    names = read_list(os.path.join(result_dir, 'names.txt'))
    is_array = np.load(os.path.join(result_dir, 'is_array.npy'))

    result_dict = {}

    for i, key in enumerate(keys):
        result_dict[key] = {}
        for name in names:
            file_name = os.path.join(result_dir, key + '_' + name)
            if not 'angle' in file_name:
                if os.path.exists(file_name + '.npy'):
                    result_dict[key][name] = np.load(file_name + '.npy')
            else:
                if os.path.exists(file_name + '.csv'):
                    result_dict[key][name] = pd.read_csv(file_name + '.csv',
                                                         index_col=0)

    return result_dict


def plot_rolling(result_key_dict, result_key, title='plot', xlabel='x',
                 ylabel=None, out_dir='logs/figures', save=False, show=True,
                 smooth=30, subplots=True, omit_last=True, log_interval=20,
                 no_title=True):
    """
    Make plots visualizing the mean and standard deviation of a smoothing
    window over the last points.
    """
    # Prepare DataFrame
    experiments = []
    max_idx = 0
    for name, df in result_key_dict.items():
        nlayers = len(df.columns)
        for layer in range(nlayers):
            df[f'roll_mean_{layer}'] = df.iloc[:, layer].rolling(window=smooth, min_periods=1).mean()
            df[f'roll_std_{layer}'] = df.iloc[:, layer].rolling(window=smooth, min_periods=1).std()
            df[f"lower{layer}"] = df[f"roll_mean_{layer}"].sub(df[f"roll_std_{layer}"], fill_value=0)
            df[f"upper{layer}"] = df[f"roll_mean_{layer}"].add(df[f"roll_std_{layer}"], fill_value=0)
        df['batch_idx'] = df.index * log_interval
        df['experiment'] = name
        experiments.append(name)
        if max_idx < df['batch_idx'].max():
            max_idx = df['batch_idx'].max()
    df = pd.concat(result_key_dict.values(), ignore_index=True)


    # Define Experiments and Color Palette
    color_palette = sns.color_palette("Set1", len(experiments))
    if len(experiments) >= 6:
        color_palette[5] = (0.33203125, 0.41796875, 0.18359375)
    if len(experiments) >= 9:
        color_palette[8] = (0., 0.796875, 0.796875)
    sns.set_palette(color_palette)

    # Plot
    if omit_last:
        nlayers = nlayers-1
    plt.close('all')
    max_value = 0
    for layer in range(nlayers):
        ax = sns.lineplot(x="batch_idx", y=f"roll_mean_{layer}", hue="experiment", data=df, estimator=None)
        for i, experiment in enumerate(experiments):
            df_fill = df.loc[df["experiment"]==experiment]
            ax.fill_between(df_fill["batch_idx"], df_fill[f"lower{layer}"], df_fill[f"upper{layer}"], alpha=0.3, lw=0, color=color_palette[i])
            if max_value < df_fill[[f"lower{layer}", f"upper{layer}"]].max().max():
                max_value = df_fill[[f"lower{layer}", f"upper{layer}"]].max().max()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, max_idx])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:], ncol=2,
                  bbox_to_anchor=(0.16, 0.69))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        sns.despine()
        if not no_title:
            plt.title(f"Layer {layer+1}: {result_key}")

        if save:
            file_name = title + '_layer' + str(layer) + '.png'
            plt.savefig(os.path.join(out_dir, file_name), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    if subplots:
        f, axes = plt.subplots(1, nlayers, figsize=(nlayers * 4+4, 5))
        plt.tight_layout()
        for layer in range(nlayers):
            ax = axes[layer]
            sns.lineplot(x="batch_idx", y=f"roll_mean_{layer}", hue="experiment", data=df, estimator=None, ax=ax)
            for i, experiment in enumerate(experiments):
                df_fill = df.loc[df["experiment"] == experiment]
                ax.fill_between(df_fill["batch_idx"], df_fill[f"lower{layer}"], df_fill[f"upper{layer}"], alpha=0.3, lw=0,
                                color=color_palette[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim([-0.005, max_value])
            ax.set_xlim([0, max_idx])
            ax.legend_.remove()
            if layer != 0:
                ax.set_ylabel("")
            ax.set_title(f"Layer {layer+1}")
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            sns.despine()

        handles, labels = ax.get_legend_handles_labels()
        f.legend(handles=handles[1:], labels=labels[1:], loc="center right")
        plt.tight_layout()
        plt.subplots_adjust(right=0.83)


        if save:
            file_name = title + '_subplots.pdf'
            plt.savefig(os.path.join(out_dir, file_name))
        if show:
            plt.show()
        else:
            plt.close()

def make_ylabel(result_key):
    if result_key == 'gnt_angles':
        ylabel = r'$\Delta W_i \angle \Delta W_i^{GNT}$ [$^\circ$]'
    elif result_key == 'bp_angles':
        ylabel = r'$\Delta W_i \angle \Delta W_i^{BP}$ [$^\circ$]'
    else:
        ylabel = 'angle in degrees'
    return ylabel

