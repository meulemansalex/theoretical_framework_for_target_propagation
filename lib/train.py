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
Collection of train and test functions.
"""

import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pandas as pd

from lib import utils
from lib.networks import LeeDTPNetwork, DTPNetwork
import pickle

def train(args, device, train_loader, net, writer, test_loader, summary,
          val_loader):
    """
    Train the given network on the given training dataset with DTP.
    Args:
        args (Namespace): The command-line arguments.
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (DTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
        summary (dict): summary dictionary with the performance measures of the
            training and testing
        val_loader (torch.utils.data.DataLoader): The data handler for the
            validation data
    """
    print('Training network ...')
    net.train()
    if args.save_weights:
        forward_parameters = net.get_forward_parameter_list()
        filename = os.path.join(args.out_dir, 'weights.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(forward_parameters, f)
    if args.load_weights:
        filename = os.path.join(args.out_dir, 'weights.pickle')
        forward_parameters_loaded = pickle.load( open(filename, 'rb'))
        for i in range(len(forward_parameters_loaded)):
            net.layers[i]._weights = forward_parameters_loaded[i]

    # Simple struct that contains the relevant training variables
    train_var = Namespace()
    train_var.summary = summary

    train_var.forward_optimizer, train_var.feedback_optimizer = \
        utils.choose_optimizer(args, net)
    if args.classification:
        if args.output_activation == 'softmax':
            train_var.loss_function = nn.CrossEntropyLoss()
        elif args.output_activation == 'sigmoid':
            train_var.loss_function = nn.MSELoss()
        else:
            raise ValueError('The mnist dataset can only be combined with a '
                             'sigmoid or softmax output activation.')

    elif args.regression:
        train_var.loss_function = nn.MSELoss()
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))
    train_var.batch_idx = 1
    train_var.batch_idx_fb = 1
    train_var.init_idx = 1

    train_var.epoch_losses = np.array([])
    train_var.epoch_reconstruction_losses = np.array([])
    train_var.epoch_reconstruction_losses_var = np.array([])
    train_var.test_losses = np.array([])
    train_var.val_losses = np.array([])

    train_var.val_loss = None
    train_var.val_accuracy = None
    if args.classification:
        train_var.epoch_accuracies = np.array([])
        train_var.test_accuracies = np.array([])
        train_var.val_accuracies = np.array([])

    if args.epochs_fb == 0 or args.freeze_fb_weights:
        print("No initial training of feedback weights.")
    else:
        print('Training the feedback weights ...')

        av_reconstruction_loss_init = -1
        train_var.summary['rec_loss_first'] = -1
        train_var.epochs_init = 0
        for e_fb in range(args.epochs_fb):
            # Train the feedback weights before starting the real training, such
            # that they are aligned with the pseudoinverse of the forward weights.
            train_var.epochs_init = e_fb
            train_var.reconstruction_losses_init = np.array([])
            train_only_feedback_parameters(args, train_var, device, train_loader,
                                           net, writer)
            av_reconstruction_loss_init = np.mean(
                train_var.reconstruction_losses_init)
            if e_fb == 0:
                train_var.summary['rec_loss_first'] = av_reconstruction_loss_init

            print('init epoch {}, reconstruction loss: {}'.format(
                e_fb + 1, av_reconstruction_loss_init
            ))

        # save the final reconstruction loss of the initialization process
        print(f'Initialization feedback weights done after {args.epochs_fb} epochs.')
        print(f'Reconstruction loss: {av_reconstruction_loss_init}')
        train_var.summary['rec_loss_init'] = av_reconstruction_loss_init
        train_var.summary['rec_loss_init_combined'] = \
            0.5 * (train_var.summary['rec_loss_init'] + \
            train_var.summary['rec_loss_first'])

        if args.train_only_feedback_parameters:
            print('Terminating training')
            return train_var.summary

    if args.output_space_plot:
        train_var.forward_optimizer.zero_grad()
        val_loader_iter = iter(val_loader)
        (inputs, targets) = val_loader_iter.next()
        inputs, targets = inputs.to(device), targets.to(device)
        if args.classification:
            if args.output_activation == 'sigmoid':
                targets = utils.int_to_one_hot(targets, 10, device,
                                               soft_target=1.)
            else:
                raise utils.NetworkError("output space plot for classification "
                                         "tasks is only possible with sigmoid "
                                         "output layer.")
        utils.make_plot_output_space(args, net,
                                     args.output_space_plot_layer_idx,
                                     train_var.loss_function,
                                     targets,
                                     inputs,
                                     steps=20)
        return train_var.summary


    train_var.epochs = 0
    for e in range(args.epochs):
        train_var.epochs = e
        if args.classification:
            train_var.accuracies = np.array([])
        train_var.losses = np.array([])
        train_var.reconstruction_losses = np.array([])
        if not args.train_separate:
            train_parallel(args, train_var, device, train_loader, net, writer)
        else:
            train_separate(args, train_var, device, train_loader, net, writer)
        if not args.freeze_fb_weights:
            for extra_e in range(args.extra_fb_epochs):
                train_only_feedback_parameters(args, train_var, device,
                                               train_loader,
                                               net, writer, log=False)

        train_var.test_accuracy, train_var.test_loss = \
            test(args, device, net, test_loader,
                 train_var.loss_function)
        if not args.no_val_set:
            train_var.val_accuracy, train_var.val_loss = \
                test(args, device, net, val_loader,
                     train_var.loss_function)

        # print intermediate results
        train_var.epoch_loss = np.mean(train_var.losses)
        print('Epoch {} -- training loss = {}.'.format(e + 1,
                                                       train_var.epoch_loss))
        if not args.no_val_set:
            print('Epoch {} -- val loss = {}.'.format(e + 1, train_var.val_loss))
        print('Epoch {} -- test loss = {}.'.format(e + 1, train_var.test_loss))

        if args.classification:
            train_var.epoch_accuracy = np.mean(train_var.accuracies)
            print('Epoch {} -- training acc  = {}%'.format(
                e + 1, train_var.epoch_accuracy * 100))
            if not args.no_val_set:
                print('Epoch {} -- val acc  = {}%'.format(
                    e + 1, train_var.val_accuracy * 100))
            print('Epoch {} -- test acc  = {}%'.format(
                e + 1, train_var.test_accuracy * 100))
        else:
            train_var.epoch_accuracy = None
        if args.save_logs:
            utils.save_logs(writer, step=e + 1, net=net,
                            loss=train_var.epoch_loss,
                            accuracy=train_var.epoch_accuracy,
                            test_loss=train_var.test_loss,
                            val_loss=train_var.val_loss,
                            test_accuracy=train_var.test_accuracy,
                            val_accuracy=train_var.val_accuracy)

        # save epoch results in summary dict
        train_var.epoch_losses = np.append(train_var.epoch_losses,
                                           train_var.epoch_loss)
        train_var.test_losses = np.append(train_var.test_losses,
                                          train_var.test_loss)

        if not args.no_val_set:
            train_var.val_losses = np.append(train_var.val_losses,
                                         train_var.val_loss)
        if not args.freeze_fb_weights:
            av_epoch_reconstruction_loss = np.mean(train_var.reconstruction_losses)
            var_epoch_reconstruction_loss = np.var(train_var.reconstruction_losses)
            train_var.epoch_reconstruction_losses = np.append(
                train_var.epoch_reconstruction_losses,
                av_epoch_reconstruction_loss)
            train_var.epoch_reconstruction_losses_var = np.append(
                train_var.epoch_reconstruction_losses_var,
                var_epoch_reconstruction_loss
            )

        if args.classification:
            train_var.epoch_accuracies = np.append(train_var.epoch_accuracies,
                                                   train_var.epoch_accuracy)
            train_var.test_accuracies = np.append(train_var.test_accuracies,
                                                  train_var.test_accuracy)
            if not args.no_val_set:
                train_var.val_accuracies = np.append(train_var.val_accuracies,
                                                     train_var.val_accuracy)

        utils.save_summary_dict(args, train_var.summary)

        if e > 4 and (not args.evaluate):
            # stop unpromising runs
            if args.dataset in ['mnist', 'fashion_mnist']:
                if train_var.epoch_accuracy < 0.4:
                    # error code to indicate pruned run
                    print('writing error code -1')
                    train_var.summary['finished'] = -1
                    break
            if args.dataset in ['cifar10']:
                if train_var.epoch_accuracy < 0.25:
                    # error code to indicate pruned run
                    print('writing error code -1')
                    train_var.summary['finished'] = -1
                    break

        # do a small gridsearch to find the damping constant for GNT angles
        if e == 2:
            if args.gn_damping_hpsearch:
                print('Doing hpsearch for finding ideal GN damping constant'
                      'for computing the angle with GNT updates')
                gn_damping = gn_damping_hpsearch(args, train_var, device,
                                                 train_loader, net, writer)
                args.gn_damping = gn_damping
                print('Damping constants GNT angles: {}'.format(gn_damping))
                train_var.summary['gn_damping_values'] = gn_damping
                return train_var.summary

    if not args.epochs == 0:
        # save training summary results in summary dict
        train_var.summary['loss_train_last'] = train_var.epoch_loss
        train_var.summary['loss_test_last'] = train_var.test_loss
        train_var.summary['loss_train_best'] = train_var.epoch_losses.min()
        train_var.summary['loss_test_best'] = train_var.test_losses.min()
        train_var.summary['loss_train'] = train_var.epoch_losses
        train_var.summary['loss_test'] = train_var.test_losses
        train_var.summary['rec_loss'] = train_var.reconstruction_losses
        if not args.no_val_set:
            train_var.summary['loss_val_last'] = train_var.val_loss
            train_var.summary['loss_val_best'] = train_var.val_losses.min()
            train_var.summary['loss_val'] = train_var.val_losses
        # pick the epoch with best validation loss and save the corresponding
        # test loss
        if not args.no_val_set:
            best_epoch = train_var.val_losses.argmin()
            train_var.summary['epoch_best_loss'] = best_epoch
            train_var.summary['loss_test_val_best'] = \
                train_var.test_losses[best_epoch]
            train_var.summary['loss_train_val_best'] = \
                train_var.epoch_losses[best_epoch]

        if not args.freeze_fb_weights:
            train_var.summary['rec_loss_last'] = av_epoch_reconstruction_loss
            train_var.summary[
                'rec_loss_best'] = train_var.epoch_reconstruction_losses.min()
            train_var.summary['rec_loss_var_av'] = np.mean(
                train_var.epoch_reconstruction_losses_var
            )

        if args.classification:
            train_var.summary['acc_train_last'] = train_var.epoch_accuracy
            train_var.summary['acc_test_last'] = train_var.test_accuracy
            train_var.summary['acc_train_best'] = train_var.epoch_accuracies.max()
            train_var.summary['acc_test_best'] = train_var.test_accuracies.max()
            train_var.summary['acc_train_growth'] = train_var.epoch_accuracies[-1] - train_var.epoch_accuracies[0]
            train_var.summary['acc_test_growth'] = train_var.test_accuracies[-1] - train_var.epoch_accuracies[0]
            train_var.summary['acc_train'] = train_var.epoch_accuracies
            train_var.summary['acc_test'] = train_var.test_accuracies
            if not args.no_val_set:
                train_var.summary['acc_val'] = train_var.val_accuracies
                train_var.summary['acc_val_last'] = train_var.val_accuracy
                train_var.summary['acc_val_best'] = train_var.val_accuracies.max()
                # pick the epoch with best validation acc and save the corresponding
                # test acc
                best_epoch = train_var.val_accuracies.argmax()
                train_var.summary['epoch_best_acc'] = best_epoch
                train_var.summary['acc_test_val_best'] = \
                    train_var.test_accuracies[best_epoch]
                train_var.summary['acc_train_val_best'] = \
                    train_var.epoch_accuracies[best_epoch]

    utils.save_summary_dict(args, train_var.summary)

    print('Training network ... Done')
    return train_var.summary


def train_parallel(args, train_var, device, train_loader, net, writer):
    """
    Train the given network on the given training dataset with DTP. The forward
    and feedback parameters are trained simultaneously for each batch.
    Args:
        args (Namespace): The command-line arguments.
        train_var (Namespace): Structure containing training variables
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (DTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
    """



    for i, (inputs, targets) in enumerate(train_loader):
        if args.double_precision:
            inputs, targets = inputs.double().to(device), targets.to(device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        predictions = net.forward(inputs)
        if args.classification and args.output_activation == 'sigmoid':
            # convert targets to one hot vectors for MSE loss:
            targets = utils.int_to_one_hot(targets, 10, device,
                                           soft_target=args.soft_target)

        train_var.batch_accuracy, train_var.batch_loss = \
            train_forward_parameters(args, net, predictions, targets,
                                     train_var.loss_function,
                                     train_var.forward_optimizer)
        if not args.freeze_fb_weights:
            train_feedback_parameters(args, net, train_var.feedback_optimizer)


        if args.classification:
            train_var.accuracies = np.append(train_var.accuracies,
                                             train_var.batch_accuracy)
        train_var.losses = np.append(train_var.losses,
                                     train_var.batch_loss.item())
        if not args.freeze_fb_weights:
            train_var.reconstruction_losses = np.append(
            train_var.reconstruction_losses,
            net.get_av_reconstruction_loss())

        for l, layer in enumerate(net.layers):
            loss_rec = layer.reconstruction_loss
            if loss_rec is not None and args.plots is not None:
                net.reconstruction_loss.at[train_var.epochs, l] = loss_rec

        if args.save_logs and i % args.log_interval == 0:
            if not args.freeze_fb_weights:
                utils.save_feedback_batch_logs(args, writer,
                                           train_var.batch_idx, net)
            utils.save_forward_batch_logs(args, writer, train_var.batch_idx,
                                          net,
                                          train_var.batch_loss, predictions)
            train_var.batch_idx += 1

        if not args.freeze_fb_weights and args.extra_fb_minibatches > 0:
            train_extra_fb_minibatches(args, train_var, device, train_loader,
                                       net)

        # update the forward parameters
        if not args.freeze_forward_weights:
            if args.train_randomized:
                # Fixme: correct if-else statement to include randomized updates
                raise NotImplementedError('The randomized version of the algorithms'
                                          'is not yet implemented. Select the '
                                          'correct layer to optimize with '
                                          'forward_optimizer.step(i).')
            train_var.forward_optimizer.step()



def train_separate(args, train_var, device, train_loader, net, writer):
    """
    Train the given network on the given training dataset with DTP. For each
    epoch, first the feedback weights are trained on the whole epoch, after
    which the forward weights are trained on the same epoch (similar to Lee2105)
    Args:
        args (Namespace): The command-line arguments.
        train_var (Namespace): Structure containing training variables
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (LeeDTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
    """

    # Train feedback parameters on whole training batch
    if not args.freeze_fb_weights:
        for i, (inputs, targets) in enumerate(train_loader):
            # print("  train fb: ", i)
            if args.double_precision:
                inputs, targets = inputs.double().to(
                    device), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)

            predictions = net.forward(inputs)

            train_feedback_parameters(args, net, train_var.feedback_optimizer)
            train_var.reconstruction_losses = np.append(
                train_var.reconstruction_losses,
                net.get_av_reconstruction_loss())
            if args.save_logs and i % args.log_interval == 0:
                utils.save_feedback_batch_logs(args, writer,
                                               train_var.batch_idx_fb, net)

                train_var.batch_idx_fb += 1


    # Train forward parameters on whole training batch
    for i, (inputs, targets) in enumerate(train_loader):
        if args.double_precision:
            inputs, targets = inputs.double().to(device), targets.to(
                device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        if args.classification and \
                args.output_activation == 'sigmoid':
            # convert targets to one hot vectors for MSE loss:
            targets = utils.int_to_one_hot(targets, 10, device,
                                           soft_target=args.soft_target)

        predictions = net.forward(inputs)

        # print(predictions.shape, targets.shape)
        train_var.batch_accuracy, train_var.batch_loss = \
            train_forward_parameters(args, net, predictions, targets,
                                     train_var.loss_function,
                                     train_var.forward_optimizer)

        if args.classification:
            train_var.accuracies = np.append(train_var.accuracies,
                                             train_var.batch_accuracy)
        train_var.losses = np.append(train_var.losses,
                                     train_var.batch_loss.item())

        for l, layer in enumerate(net.layers):
            loss_rec = layer.reconstruction_loss
            if loss_rec is not None and args.plots is not None:
                net.reconstruction_loss.at[train_var.epochs, l] = loss_rec

        if args.save_logs and i % args.log_interval == 0:
            utils.save_forward_batch_logs(args, writer, train_var.batch_idx,
                                          net,
                                          train_var.batch_loss, predictions)
            train_var.batch_idx += 1

        # update the forward parameters
        if not args.freeze_forward_weights:
            if args.train_randomized:
                # Fixme: correct if-else statement to include randomized updates
                raise NotImplementedError(
                    'The randomized version of the algorithms'
                    'is not yet implemented. Select the '
                    'correct layer to optimize with '
                    'forward_optimizer.step(i).')
            train_var.forward_optimizer.step()


def train_forward_parameters(args, net, predictions, targets, loss_function,
                             forward_optimizer):
    """ Train the forward parameters on the current mini-batch."""
    if predictions.requires_grad == False:
        # we need the gradient of the loss with respect to the network
        # output. If a LeeDTPNetwork is used, this is already the case.
        # The gradient will also be saved in the activations attribute of the
        # output layer of the network
        predictions.requires_grad = True

    save_target = args.save_GN_activations_angle or \
                  args.save_BP_activations_angle

    forward_optimizer.zero_grad()
    loss = loss_function(predictions, targets)
    if not args.train_randomized:
        net.backward(loss, args.target_stepsize, save_target=save_target,
                     norm_ratio=args.norm_ratio)
    else:
        k = np.random.randint(0, net.depth)
        net.backward_random(loss, args.target_stepsize, k,
                            save_target=save_target, norm_ratio=args.norm_ratio)

    if args.classification:
        if args.output_activation == 'sigmoid':
            batch_accuracy = utils.accuracy(predictions,
                                            utils.one_hot_to_int(targets))
        else: #softmax
            batch_accuracy = utils.accuracy(predictions, targets)
    else:
        batch_accuracy = None
    batch_loss = loss

    return batch_accuracy, batch_loss

def train_feedback_parameters(args, net, feedback_optimizer):
    """ Train the feedback parameters on the current mini-batch."""

    feedback_optimizer.zero_grad()

    if args.diff_rec_loss:
        if not args.train_randomized_fb:
            for k in range(1, net.depth):
                net.compute_feedback_gradients(k)
        else:
            k = np.random.randint(1, net.depth)
            net.compute_feedback_gradients(k)
    elif args.direct_fb:
        if not args.train_randomized_fb:
            for k in range(0, net.depth-1):
                net.compute_feedback_gradients(k)
        else:
            k = np.random.randint(0, net.depth-1)
            net.compute_feedback_gradients(k)
    else:
        net.compute_feedback_gradients()

    feedback_optimizer.step()

def test(args, device, net, test_loader, loss_function):
    """
    Compute the test loss and accuracy on the test dataset
    Args:
        args: command line inputs
        net: network
        test_loader (DataLoader): dataloader object with the test dataset

    Returns: Tuple containing:
        - Test accuracy
        - Test loss
    """
    loss = 0
    if args.classification:
        accuracy = 0
    nb_batches = len(test_loader)
    with torch.no_grad():
        for inputs, targets in test_loader:
            if args.double_precision:
                inputs, targets = inputs.double().to(device), targets.to(
                    device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.classification and\
                    args.output_activation == 'sigmoid':
                # convert targets to one hot vectors for MSE loss:
                targets = utils.int_to_one_hot(targets, 10, device,
                                           soft_target=args.soft_target)
            predictions = net.forward(inputs)
            loss += loss_function(predictions, targets).item()
            if args.classification:
                if args.output_activation == 'sigmoid':
                    accuracy += utils.accuracy(predictions,
                                               utils.one_hot_to_int(
                                                        targets))
                else:  # softmax
                    accuracy += utils.accuracy(predictions, targets)
    loss /= nb_batches
    if args.classification:
        accuracy /= nb_batches
    else:
        accuracy = None
    return accuracy, loss


def train_only_feedback_parameters(args, train_var, device, train_loader,
                                   net, writer, log=True):
    """ Train only the feedback parameters for the given amount of epochs.
    This function is used to initialize the network in a 'pseudo-inverse'
    condition. """


    for i, (inputs, targets) in enumerate(train_loader):
        # print("  train fb: ", i)
        if args.double_precision:
            inputs, targets = inputs.double().to(device), targets.to(
                device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)

        predictions = net.forward(inputs)

        train_feedback_parameters(args, net, train_var.feedback_optimizer)

        for l, layer in enumerate(net.layers):
            loss_rec = layer.reconstruction_loss
            if loss_rec is not None and args.plots is not None and log:
                net.reconstruction_loss_init.at[train_var.epochs_init, l] = \
                    loss_rec
        if log:
            train_var.reconstruction_losses_init = np.append(
                train_var.reconstruction_losses_init,
                net.get_av_reconstruction_loss())
        if args.save_logs and i % args.log_interval == 0 and log:
            utils.save_feedback_batch_logs(args, writer,
                                           train_var.init_idx, net,
                                           init=True)
            train_var.init_idx += 1


def train_extra_fb_minibatches(args, train_var, device, train_loader,
                               net):
    train_loader_iter = iter(train_loader)
    for i in range(args.extra_fb_minibatches):
        (inputs, targets) = train_loader_iter.next()
        if args.double_precision:
            inputs, targets = inputs.double().to(device), targets.to(device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        predictions = net.forward(inputs)
        train_feedback_parameters(args, net, train_var.feedback_optimizer)


def train_bp(args, device, train_loader, net, writer, test_loader, summary,
             val_loader):
    print('Training network ...')
    net.train()
    forward_optimizer = utils.OptimizerList(args, net)

    nb_batches = len(train_loader)

    if args.classification:
        if args.output_activation == 'softmax':
            loss_function = nn.CrossEntropyLoss()
        elif args.output_activation == 'sigmoid':
            loss_function = nn.MSELoss()
        else:
            raise ValueError('The mnist dataset can only be combined with a '
                             'sigmoid or softmax output activation.')

    elif args.regression:
        loss_function = nn.MSELoss()
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))

    epoch_losses = np.array([])
    epoch_reconstruction_losses = np.array([])
    epoch_reconstruction_losses_var = np.array([])
    test_losses = np.array([])
    val_losses = np.array([])
    val_loss = None
    val_accuracy = None

    if args.classification:
        epoch_accuracies = np.array([])
        test_accuracies = np.array([])
        val_accuracies = np.array([])

    if args.output_space_plot:
        forward_optimizer.zero_grad()
        val_loader_iter = iter(val_loader)
        (inputs, targets) = val_loader_iter.next()
        inputs, targets = inputs.to(device), targets.to(device)
        if args.classification:
            if args.output_activation == 'sigmoid':
                targets = utils.int_to_one_hot(targets, 10, device,
                                               soft_target=1.)
            else:
                raise utils.NetworkError("output space plot for classification "
                                     "tasks is only possible with sigmoid "
                                     "output layer.")
        utils.make_plot_output_space_bp(args, net,
                                     args.output_space_plot_layer_idx,
                                     loss_function,
                                     targets,
                                     inputs,
                                     steps=20)
        return summary

    for e in range(args.epochs):
        if args.classification:
            running_accuracy = 0
        else:
            running_accuracy = None
        running_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            if args.double_precision:
                inputs, targets = inputs.double().to(
                    device), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'BPConv':
                inputs = inputs.flatten(1, -1)
            if args.classification and \
                    args.output_activation == 'sigmoid':
                # convert targets to one hot vectors for MSE loss:
                targets = utils.int_to_one_hot(targets, 10, device,
                                               soft_target=args.soft_target)

            forward_optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            forward_optimizer.step()

            running_loss += loss.item()

            if args.classification:
                if args.output_activation == 'sigmoid':
                    running_accuracy += utils.accuracy(predictions,
                                                    utils.one_hot_to_int(
                                                        targets))
                else:  # softmax
                    running_accuracy += utils.accuracy(predictions, targets)

        test_accuracy, test_loss = test_bp(args, device, net, test_loader,
                                           loss_function)
        if not args.no_val_set:
            val_accuracy, val_loss = test_bp(args, device, net, val_loader,
                                             loss_function)
        epoch_loss = running_loss/nb_batches
        if args.classification:
            epoch_accuracy = running_accuracy/nb_batches
        else:
            epoch_accuracy = None

        print('Epoch {} -- training loss = {}.'.format(e + 1,
                                                       epoch_loss))
        if not args.no_val_set:
            print('Epoch {} -- val loss = {}.'.format(e + 1, val_loss))
        print('Epoch {} -- test loss = {}.'.format(e + 1, test_loss))

        if args.classification:
            print('Epoch {} -- training acc  = {}%'.format(
                e + 1, epoch_accuracy * 100))
            if not args.no_val_set:
                print('Epoch {} -- val acc  = {}%'.format(
                    e + 1, val_accuracy * 100))
            print('Epoch {} -- test acc  = {}%'.format(
                e + 1, test_accuracy * 100))
        if args.save_logs:
            utils.save_logs(writer, step=e + 1, net=net,
                            loss=epoch_loss,
                            accuracy=epoch_accuracy,
                            test_loss=test_loss,
                            test_accuracy=test_accuracy,
                            val_loss=val_loss,
                            val_accuracy=val_accuracy)

        epoch_losses = np.append(epoch_losses,
                                           epoch_loss)
        test_losses = np.append(test_losses,
                                          test_loss)
        if not args.no_val_set:
            val_losses = np.append(val_losses, val_loss)

        if args.classification:
            epoch_accuracies = np.append(
                epoch_accuracies,
                epoch_accuracy)
            test_accuracies = np.append(test_accuracies,
                                                  test_accuracy)
            if not args.no_val_set:
                val_accuracies = np.append(val_accuracies, val_accuracy)

        utils.save_summary_dict(args, summary)

        if e > 4:
            # stop unpromising runs
            if args.dataset in ['mnist', 'fashion_mnist']:
                if epoch_accuracy < 0.4:
                    # error code to indicate pruned run
                    print('writing error code -1')
                    summary['finished'] = -1
                    break
            if args.dataset in ['cifar10']:
                if epoch_accuracy < 0.25:
                    # error code to indicate pruned run
                    print('writing error code -1')
                    summary['finished'] = -1
                    break

    if not args.epochs == 0:
        # save training summary results in summary dict
        summary['loss_train_last'] = epoch_loss
        summary['loss_test_last'] = test_loss
        summary['loss_train_best'] = epoch_losses.min()
        summary['loss_test_best'] = test_losses.min()
        summary['loss_train'] = epoch_losses
        summary['loss_test'] = test_losses
        if not args.no_val_set:
            summary['loss_val_last'] = val_loss
            summary['loss_val_best'] = val_losses.min()
            summary['loss_val'] = val_losses
            # pick the epoch with best validation loss and save the corresponding
            # test loss
            best_epoch = val_losses.argmin()
            summary['epoch_best_loss'] = best_epoch
            summary['loss_test_val_best'] = \
                test_losses[best_epoch]
            summary['loss_train_val_best'] = \
                epoch_losses[best_epoch]

        if args.classification:
            summary['acc_train_last'] = epoch_accuracy
            summary['acc_test_last'] = test_accuracy
            summary[
                'acc_train_best'] = epoch_accuracies.max()
            summary[
                'acc_test_best'] = test_accuracies.max()
            summary['acc_train'] = epoch_accuracies
            summary['acc_test'] = test_accuracies
            if not args.no_val_set:
                summary['acc_val'] = val_accuracies
                summary['acc_val_last'] = val_accuracy
                summary['acc_val_best'] = val_accuracies.max()
                # pick the epoch with best validation acc and save the corresponding
                # test acc
                best_epoch = val_accuracies.argmax()
                summary['epoch_best_acc'] = best_epoch
                summary['acc_test_val_best'] = \
                    test_accuracies[best_epoch]
                summary['acc_train_val_best'] = \
                    epoch_accuracies[best_epoch]
    utils.save_summary_dict(args, summary)

    print('Training network ... Done')
    return summary


def test_bp(args, device, net, test_loader, loss_function):
    loss = 0
    if args.classification:
        accuracy = 0
    nb_batches = len(test_loader)
    with torch.no_grad():
        for inputs, targets in test_loader:
            if args.double_precision:
                inputs, targets = inputs.double().to(device), targets.to(
                    device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'BPConv':
                inputs = inputs.flatten(1, -1)
            if args.classification and args.output_activation == 'sigmoid':
                # convert targets to one hot vectors for MSE loss:
                targets = utils.int_to_one_hot(targets, 10, device,
                                           soft_target=args.soft_target)
            predictions = net(inputs)
            loss += loss_function(predictions, targets).item()
            if args.classification:
                if args.output_activation == 'sigmoid':
                    accuracy += utils.accuracy(predictions,
                                               utils.one_hot_to_int(
                                                        targets))
                else:  # softmax
                    accuracy += utils.accuracy(predictions, targets)
    loss /= nb_batches
    if args.classification:
        accuracy /= nb_batches
    else:
        accuracy = None
    return accuracy, loss


def gn_damping_hpsearch(args, train_var, device, train_loader, net, writer):
    nb_hidden_layers = len(net.layers)-1
    freeze_forward_weights_copy = args.freeze_forward_weights
    args.freeze_forward_weights = True
    damping_values = np.logspace(-5., 1., num=7, base=10.0)
    damping_values = np.append(0, damping_values)
    average_angles = np.empty((len(damping_values), nb_hidden_layers))

    for k, gn_damping in enumerate(damping_values):
        print('testing damping={}'.format(gn_damping))
        angles_df = pd.DataFrame(columns=[i for i in range(0, nb_hidden_layers)])
        step=0
        for i, (inputs, targets) in enumerate(train_loader):
            # print("  train fw: ", i)
            if args.double_precision:
                inputs, targets = inputs.double().to(
                    device), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.classification and \
                    args.output_activation == 'sigmoid':
                # convert targets to one hot vectors for MSE loss:
                targets = utils.int_to_one_hot(targets, 10, device,
                                               soft_target=args.soft_target)

            predictions = net.forward(inputs)

            # print(predictions.shape, targets.shape)
            acc, loss = \
                train_forward_parameters(args, net, predictions, targets,
                                         train_var.loss_function,
                                         train_var.forward_optimizer)

            if  i % args.log_interval == 0:
                net.save_gnt_angles(writer, step, predictions,
                                    loss, gn_damping,
                                    retain_graph=False,
                                    custom_result_df=angles_df)
                step += 1

        average_angles[k, :] = angles_df.mean(axis=0)

    # select the damping values corresponding with the minimum angles
    optimal_damping_constants = damping_values[average_angles.argmin(axis=0)]
    print('average angles:')
    print(average_angles)

    args.freeze_forward_weights = freeze_forward_weights_copy

    return optimal_damping_constants




