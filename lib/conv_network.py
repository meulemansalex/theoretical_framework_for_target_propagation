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

import torch
from torch import nn
import numpy as np
from lib.networks import DTPNetwork
from lib.conv_layers import DDTPConvLayer, DDTPConvControlLayer
from lib.direct_feedback_layers import DDTPMLPLayer, DDTPControlLayer
import torch.nn.functional as F
from lib import utils
import pandas as pd

class DDTPConvNetwork(nn.Module):
    def __init__(self, bias=True, hidden_activation='tanh',
                 feedback_activation='linear', initialization='xavier_normal',
                 sigma=0.1, plots=None, forward_requires_grad=False):
        nn.Module.__init__(self)
        l1 = DDTPConvLayer(3, 96, (5, 5), 10, [96, 16, 16],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l2 = DDTPConvLayer(96, 128, (5, 5), 10, [128, 8, 8],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l3 = DDTPConvLayer(128, 256, (5, 5), 10, [256, 4, 4],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l4 = DDTPMLPLayer(4 * 4 * 256, 2048, 10, bias=True,
                          forward_requires_grad=forward_requires_grad,
                          forward_activation=hidden_activation,
                          feedback_activation=feedback_activation,
                          size_hidden_fb=None, initialization=initialization,
                          is_output=False,
                          recurrent_input=False)
        l5 = DDTPMLPLayer(2048, 2048, 10, bias=True,
                          forward_requires_grad=forward_requires_grad,
                          forward_activation=hidden_activation,
                          feedback_activation=feedback_activation,
                          size_hidden_fb=None, initialization=initialization,
                          is_output=False,
                          recurrent_input=False)
        l6 = DDTPMLPLayer(2048, 10, 10, bias=True,
                          forward_requires_grad=forward_requires_grad,
                          forward_activation='linear',
                          feedback_activation=feedback_activation,
                          size_hidden_fb=None, initialization=initialization,
                          is_output=True,
                          recurrent_input=False)
        self._layers = nn.ModuleList([l1, l2, l3, l4, l5, l6])
        self._depth = 6
        self.nb_conv = 3
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad

        self._plots = plots
        if plots is not None:
            self.bp_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

            self.reconstruction_loss_init = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def layers(self):
        """Getter for read-only attribute :attr:`layers`."""
        return self._layers

    @property
    def sigma(self):
        """ Getter for read-only attribute sigma"""
        return self._sigma

    @property
    def input(self):
        """ Getter for attribute input."""
        return self._input

    @input.setter
    def input(self, value):
        """ Setter for attribute input."""
        self._input = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    def forward(self, x):
        self.input = x
        y = x
        for i, layer in enumerate(self.layers):
            if i == self.nb_conv: # flatten conv layer
                y = y.view(y.shape[0], -1)
            y = layer.forward(y)
        return y

    def dummy_forward(self, h, i):
        """Propagate the activation of layer i forward through the network
        without saving the activations"""
        y = h
        for k, layer in enumerate(self.layers[i+1:]):
            if k + i + 1 == self.nb_conv:
                y = y.view(y.shape[0], -1)
            y = layer.dummy_forward(y)
        return y

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer

        Returns: Mini-batch of output targets
        """
        output_activations = self.layers[-1].activations

        gradient = torch.autograd.grad(loss, output_activations,
                                       retain_graph=self.forward_requires_grad)\
                                        [0].detach()
        output_targets = output_activations - \
                         target_lr*gradient
        return output_targets

    def propagate_backward(self, output_target, i):
        """
        Propagate the output target backwards to layer i in a DTP-like fashion.
        Args:
            h_target (torch.Tensor): the output target
            i: the layer index to which the target must be propagated

        Returns: the target for layer i

        """
        output_activation = self.layers[-1].activations
        layer_activation = self.layers[i].activations

        layer_target = self.layers[i].backward(output_target, layer_activation,
                                               output_activation)
        return layer_target

    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        output_target = self.compute_output_target(loss, target_lr)
        self.layers[-1].compute_forward_gradients(output_target,
                                                  self.layers[-2].activations)
        if save_target:
            self.layers[-1].target = output_target
        for i in range(self.depth - 1):
            h_target = self.propagate_backward(output_target, i)
            if save_target:
                self.layers[i].target = h_target
            if i == 0:
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         self.forward_requires_grad)
            else:
                previous_activations = self.layers[i - 1].activations
                if i == self.nb_conv:  # flatten conv layer
                    previous_activations = \
                        previous_activations.view(previous_activations.shape[0], -1)
                self.layers[i].compute_forward_gradients(h_target,
                                                         previous_activations,
                                                         self.forward_requires_grad)

    def compute_feedback_gradients(self, i):
        self.reconstruction_loss_index = i
        h_corrupted = self.layers[i].activations + \
                      self.sigma * torch.randn_like(self.layers[i].activations)

        output_corrupted = self.dummy_forward(h_corrupted, i)
        output_noncorrupted = self.layers[-1].activations

        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  output_corrupted,
                                                  output_noncorrupted,
                                                  self.sigma)

    def get_forward_parameter_list(self):
        parameterlist = []
        for layer in self.layers:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_feedback_parameter_list(self):
        parameterlist = []
        for layer in self.layers[0:-1]:
            parameterlist += [p for p in layer.get_feedback_parameters()]
        return parameterlist

    def get_conv_feedback_parameter_list(self):
        parameterlist = []
        for layer in self.layers[0:-1]:
            if isinstance(layer, DDTPConvLayer):
                parameterlist += [p for p in layer.get_feedback_parameters()]
        return parameterlist

    def get_fc_feedback_parameter_list(self):
        parameterlist = []
        for layer in self.layers[0:-1]:
            if isinstance(layer, DDTPMLPLayer):
                parameterlist += [p for p in layer.get_feedback_parameters()]
        return parameterlist

    def save_logs(self, writer, step):
        for i in range(len(self.layers)):
            name = 'layer {}'.format(i + 1)
            self.layers[i].save_logs(writer, step, name,
                                     no_gradient=i == len(self.layers) - 1)

    def save_feedback_batch_logs(self, writer, step, init=False):
        for i in range(len(self.layers)):
            name = 'layer {}'.format(i + 1)
            self.layers[i].save_feedback_batch_logs(writer, step, name,
                                                    no_gradient=i == len(
                                                        self.layers) - 1,
                                                    init=init)

    def get_av_reconstruction_loss(self):
        """ Get the reconstruction loss of the network for the layer of which
        the feedback parameters were trained on the current mini-batch
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        reconstruction_loss = self.layers[self.reconstruction_loss_index]. \
            reconstruction_loss
        return reconstruction_loss

    def compute_bp_angles(self, loss, i, retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).

        """
        bp_gradients = self.layers[i].compute_bp_update(loss,
                                                        retain_graph)
        gradients = self.layers[i].get_forward_gradients()
        if utils.contains_nan(bp_gradients[0].detach()):
            print('bp update contains nan (layer {}):'.format(i))
            # print(bp_gradients[0].detach())
        if utils.contains_nan(gradients[0].detach()):
            print('weight update contains nan (layer {}):'.format(i))
            # print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') < 1e-14:
            print('norm updates approximately zero (layer {}):'.format(i))
            # print(torch.norm(gradients[0].detach(), p='fro'))
            # print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') == 0:
            print('norm updates exactly zero (layer {}):'.format(i))
            # print(torch.norm(gradients[0].detach(), p='fro'))
            # print(gradients[0].detach())

        weights_angle = utils.compute_angle(bp_gradients[0].detach(),
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(bp_gradients[1].detach(),
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )

    def save_bp_angles(self, writer, step, loss, retain_graph=False):

        layer_indices = range(len(self.layers))

        for i in layer_indices:
            name = 'layer {}'.format(i+1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_bp_angles(loss, i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_bp_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.bp_angles.at[step, i] = angles[0].item()


            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_bp_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        if i == 0:
            h_previous = self.input
        elif i == self.nb_conv:
            batchsize = self.layers[i-1].activations.shape[0]
            h_previous = self.layers[i-1].activations.view(batchsize, -1)
        else:
            h_previous = self.layers[i-1].activations

        gnt_updates = self.layers[i].compute_gnt_updates(
            output_activation=output_activation,
            loss=loss,
            h_previous=h_previous,
            damping=damping,
            retain_graph=retain_graph,
            linear=linear
        )

        gradients = self.layers[i].get_forward_gradients()
        weights_angle = utils.compute_angle(gnt_updates[0], gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(gnt_updates[1], gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )

    def save_gnt_angles(self, writer, step, output_activation, loss,
                        damping, retain_graph=False, custom_result_df=None):

        layer_indices = range(len(self.layers)-1)

        # assign a damping constant for each layer for computing the gnt angles
        if isinstance(damping, float):
            damping = [damping for i in range(self.depth)]
        else:
            # print(damping)
            # print(len(damping))
            # print(layer_indices)
            # print(len(layer_indices))
            assert len(damping) == len(layer_indices)

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gnt_angle(output_activation=output_activation,
                                            loss=loss,
                                            damping=damping[i],
                                            i=i,
                                            step=step,
                                            retain_graph=retain_graph_flag)
            if custom_result_df is not None:
                custom_result_df.at[step,i] = angles[0].item()
            else:
                writer.add_scalar(
                    tag='{}/weight_gnt_angle'.format(name),
                    scalar_value=angles[0],
                    global_step=step
                )

                if self._plots is not None:
                    # print('saving gnt angles')
                    # print(angles[0].item())
                    self.gnt_angles.at[step, i] = angles[0].item()

                if self.layers[i].bias is not None:
                    writer.add_scalar(
                        tag='{}/bias_gnt_angle'.format(name),
                        scalar_value=angles[1],
                        global_step=step
                    )


class DDTPConvNetworkCIFAR(DDTPConvNetwork):
    def __init__(self, bias=True, hidden_activation='tanh',
                 feedback_activation='linear', initialization='xavier_normal',
                 sigma=0.1, plots=None,
                 forward_requires_grad=False):
        nn.Module.__init__(self)
        l1 = DDTPConvLayer(3, 32, (5, 5), 10, [32, 16, 16],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l2 = DDTPConvLayer(32, 64, (5, 5), 10, [64, 8, 8],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l3 = DDTPMLPLayer(8 * 8 * 64, 512, 10, bias=True,
                          forward_requires_grad=forward_requires_grad,
                          forward_activation=hidden_activation,
                          feedback_activation=feedback_activation,
                          size_hidden_fb=None, initialization=initialization,
                          is_output=False,
                          recurrent_input=False)
        l4 = DDTPMLPLayer(512, 10, 10, bias=True,
                          forward_requires_grad=forward_requires_grad,
                          forward_activation='linear',
                          feedback_activation=feedback_activation,
                          size_hidden_fb=None, initialization=initialization,
                          is_output=True,
                          recurrent_input=False)
        self._layers = nn.ModuleList([l1, l2, l3, l4])
        self._depth = 4
        self.nb_conv = 2
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad

        self._plots = plots
        if plots is not None:
            self.bp_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

            self.reconstruction_loss_init = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])


class BPConvNetwork(nn.Module):
    def __init__(self, bias=True, hidden_activation='tanh',
                 initialization='xavier_normal'):
        super().__init__()
        l1 = nn.Conv2d(3, 96, (5, 5), stride=1, padding=2,
                       dilation=1, groups=1, bias=bias,
                       padding_mode='zeros')
        l2 = nn.MaxPool2d((3, 3), (2, 2), 1, 1)
        l3 = nn.Conv2d(96, 128, (5, 5), stride=1, padding=2,
                       dilation=1, groups=1, bias=bias,
                       padding_mode='zeros')
        l4 = nn.MaxPool2d((3, 3), (2, 2), 1, 1)
        l5 = nn.Conv2d(128, 256, (5, 5), stride=1, padding=2,
                       dilation=1, groups=1, bias=bias,
                       padding_mode='zeros')
        l6 = nn.MaxPool2d((3, 3), (2, 2), 1, 1)
        l7 = nn.Linear(4*4*256, 2048, bias=bias)
        l8 = nn.Linear(2048, 2048, bias=bias)
        l9 = nn.Linear(2048, 10, bias=bias)
        self.layers = nn.ModuleList([l1,l2,l3,l4,l5,l6,l7,l8,l9])

        for param in self.layers.parameters():
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            elif initialization == 'xavier':
                nn.init.xavier_uniform_(param)
            elif initialization == 'xavier_normal':
                nn.init.xavier_normal_(param)
            else:
                raise ValueError('Provided weight initialization "{}" is not '
                                 'supported.'.format(initialization))

        self.activation = hidden_activation
        self.nb_conv = 3

    @staticmethod
    def nonlinearity(x, nonlinearity):
        if nonlinearity == 'tanh':
            return torch.tanh(x)
        elif nonlinearity == 'relu':
            return F.relu(x)
        elif nonlinearity == 'linear':
            return x
        elif nonlinearity == 'leakyrelu':
            return F.leaky_relu(x, 0.2)
        elif nonlinearity == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(nonlinearity))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if i == self.nb_conv*2:
                x = x.view(x.shape[0], -1)
            x = layer(x)
            if not isinstance(layer, nn.MaxPool2d):
                x = self.nonlinearity(x, self.activation)
        x = self.layers[-1](x)
        return x

    def set_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the all the parameters
        to the given value
        Args:
            value (bool): True or False
        """
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')

        for param in self.parameters():
            param.requires_grad = value


class BPConvNetworkCIFAR(BPConvNetwork):
    def __init__(self, bias=True, hidden_activation='tanh',
                 initialization='xavier_normal'):
        nn.Module.__init__(self)
        l1 = nn.Conv2d(3, 32, (5, 5), stride=1, padding=2,
                       dilation=1, groups=1, bias=bias,
                       padding_mode='zeros')
        l2 = nn.MaxPool2d((3, 3), (2, 2), 1, 1)
        l3 = nn.Conv2d(32, 64, (5, 5), stride=1, padding=2,
                       dilation=1, groups=1, bias=bias,
                       padding_mode='zeros')
        l4 = nn.MaxPool2d((3, 3), (2, 2), 1, 1)
        l5 = nn.Linear(8*8*64, 512, bias=bias)
        l6 = nn.Linear(512, 10, bias=bias)
        self.layers = nn.ModuleList([l1,l2,l3,l4,l5,l6])

        for param in self.layers.parameters():
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            elif initialization == 'xavier':
                nn.init.xavier_uniform_(param)
            elif initialization == 'xavier_normal':
                nn.init.xavier_normal_(param)
            else:
                raise ValueError('Provided weight initialization "{}" is not '
                                 'supported.'.format(initialization))

        self.activation = hidden_activation
        self.nb_conv = 2


class DDTPConvControlNetworkCIFAR(DDTPConvNetworkCIFAR):
    def __init__(self, bias=True, hidden_activation='tanh',
                 feedback_activation='linear', initialization='xavier_normal',
                 sigma=0.1, plots=None,
                 forward_requires_grad=False):
        nn.Module.__init__(self)
        l1 = DDTPConvControlLayer(3, 32, (5, 5), 10, [32, 16, 16],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l2 = DDTPConvControlLayer(32, 64, (5, 5), 10, [64, 8, 8],
                           stride=1, padding=2, dilation=1, groups=1,
                           bias=bias, padding_mode='zeros',
                           initialization=initialization,
                           pool_type='max', pool_kernel_size=(3, 3),
                           pool_stride=(2, 2), pool_padding=1, pool_dilation=1,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation)
        l3 = DDTPControlLayer(8 * 8 * 64, 512, 10, bias=True,
                           forward_requires_grad=forward_requires_grad,
                           forward_activation=hidden_activation,
                           feedback_activation=feedback_activation,
                           size_hidden_fb=None, initialization=initialization,
                           is_output=False,
                           recurrent_input=False)
        l4 = DDTPControlLayer(512, 10, 10, bias=True,
                           forward_requires_grad=forward_requires_grad,
                           forward_activation='linear',
                           feedback_activation=feedback_activation,
                           size_hidden_fb=None, initialization=initialization,
                           is_output=True,
                           recurrent_input=False)
        self._layers = nn.ModuleList([l1, l2, l3, l4])
        self._depth = 4
        self.nb_conv = 2
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad

        self._plots = plots
        if plots is not None:
            self.bp_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

            self.reconstruction_loss_init = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])

            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])






