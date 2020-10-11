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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib import utils
import warnings

class DDTPConvLayer(nn.Module):
    """
    A convolutional layer combined with a pool layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 output_size, feature_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 initialization='xavier_normal', pool_type='max',
                 pool_kernel_size=None, pool_stride=None, pool_padding=0,
                 pool_dilation=1, forward_activation='tanh',
                 feedback_activation='linear'):
        nn.Module.__init__(self)

        self._conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias,
                                     padding_mode)

        if pool_kernel_size is None:
            pool_kernel_size = kernel_size
        self._pool_layer = self.construct_pool_layer(pool_type, pool_kernel_size,
                                                     pool_stride, pool_padding,
                                                     pool_dilation)
        feature_size_flat = feature_size[0]*feature_size[1]*feature_size[2]
        self._feedbackweights = nn.Parameter(torch.Tensor(feature_size_flat,
                                                          output_size),
                                             requires_grad=False)
        if initialization == 'xavier_normal':
            nn.init.xavier_normal_(self._conv_layer.weight.data)
            nn.init.xavier_normal_(self._feedbackweights.data)
        else:
            raise ValueError('initialization type {} not supported yet'
                             'for convolutional layers'.format(initialization))
        if bias:
            nn.init.constant_(self._conv_layer.bias.data, 0)

        self._activations = None
        self._reconstruction_loss = None
        self._forward_activation = forward_activation
        self._feedback_activation = feedback_activation
        self._target = None
        self._feature_size = feature_size

    def construct_pool_layer(self, pool_type, kernel_size, stride, padding, dilation):
        if pool_type == 'max':
            return nn.MaxPool2d(kernel_size, stride, padding, dilation)
        elif pool_type == 'average':
            return nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise ValueError('Pooling type {} not supported'.format(pool_type))

    @property
    def weights(self):
        return self._conv_layer.weight

    @property
    def bias(self):
        return self._conv_layer.bias

    @property
    def feedbackweights(self):
        return self._feedbackweights

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def reconstruction_loss(self):
        """ Getter for attribute reconstruction_loss."""
        return self._reconstruction_loss

    @reconstruction_loss.setter
    def reconstruction_loss(self, value):
        """ Setter for attribute reconstruction_loss."""
        self._reconstruction_loss = value

    @property
    def forward_activation(self):
        """ Getter for read-only attribute forward_activation"""
        return self._forward_activation

    @property
    def feedback_activation(self):
        """ Getter for read-only attribute feedback_activation"""
        return self._feedback_activation

    @property
    def target(self):
        """ Getter for attribute target"""
        return self._target

    def get_forward_parameter_list(self):
        """ Return forward weights and forward bias if applicable"""
        parameterlist = []
        parameterlist.append(self.weights)
        if self.bias is not None:
            parameterlist.append(self.bias)
        return parameterlist

    def get_feedback_parameters(self):
        return [self._feedbackweights]

    def forward_activationfunction(self, x):
        """ Element-wise forward activation function"""
        if self.forward_activation == 'tanh':
            return torch.tanh(x)
        elif self.forward_activation == 'relu':
            return F.relu(x)
        elif self.forward_activation == 'linear':
            return x
        elif self.forward_activation == 'leakyrelu':
            return F.leaky_relu(x, 0.2)
        elif self.forward_activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def feedback_activationfunction(self, x):
        """ Element-wise feedback activation function"""
        if self.feedback_activation == 'tanh':
            return torch.tanh(x)
        elif self.feedback_activation == 'relu':
            return F.relu(x)
        elif self.feedback_activation == 'linear':
            return x
        elif self.feedback_activation == 'leakyrelu':
            return F.leaky_relu(x, 5)
        elif self.feedback_activation == 'sigmoid':
            if torch.sum(x < 1e-12) > 0 or torch.sum(x > 1-1e-12) > 0:
                warnings.warn('Input to inverse sigmoid is out of'
                                 'bound: x={}'.format(x))
            inverse_sigmoid = torch.log(x/(1-x))
            if utils.contains_nan(inverse_sigmoid):
                raise ValueError('inverse sigmoid function outputted a NaN')
            return torch.log(x/(1-x))
        else:
            raise ValueError('The provided feedback activation {} is not '
                             'supported'.format(self.feedback_activation))

    def forward(self, x):
        # x = x.detach()
        x = self._conv_layer(x)
        x = self.forward_activationfunction(x)
        x = self._pool_layer(x)
        self.activations = x
        return self.activations

    def dummy_forward(self, x):
        # x = x.detach()
        x = self._conv_layer(x)
        x = self.forward_activationfunction(x)
        x = self._pool_layer(x)
        return x

    def propagate_backward(self, output_target):
        h = output_target.mm(self.feedbackweights.t())
        h = self.feedback_activationfunction(h)
        return torch.reshape(h, [output_target.shape[0]] + self._feature_size)

    def backward(self, output_target, layer_activation, output_activation):
        layer_target = self.propagate_backward(output_target)
        layer_tilde = self.propagate_backward(output_activation)

        return layer_target + layer_activation - layer_tilde

    def compute_forward_gradients(self, h_target, h_previous,
                                  forward_requires_grad=False):
        local_loss = F.mse_loss(self.activations, h_target.detach())
        if self.bias is not None:
            grads = torch.autograd.grad(local_loss, [self.weights, self.bias],
                                        retain_graph=forward_requires_grad)
            self._conv_layer.bias.grad = grads[1].detach()
        else:
            grads = torch.autograd.grad(local_loss, self.weights,
                                        retain_graph=forward_requires_grad)
        self._conv_layer.weight.grad = grads[0].detach()

    def set_feedback_requires_grad(self, value):
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')
        self._feedbackweights.requires_grad = value

    def compute_feedback_gradients(self, h_corrupted, output_corrupted,
                                   output_activation, sigma):
        self.set_feedback_requires_grad(True)
        h_activation = self.activations
        h_reconstructed = self.backward(output_corrupted, h_activation,
                                        output_activation)
        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))
        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h_reconstructed,
                                         h_corrupted)
        self.save_feedback_gradients(reconstruction_loss)
        self.set_feedback_requires_grad(False)

    def save_feedback_gradients(self, reconstruction_loss):
        self.reconstruction_loss = reconstruction_loss.item()
        grads = torch.autograd.grad(reconstruction_loss,
                                    self.feedbackweights,
                                    retain_graph=False)
        self._feedbackweights.grad = grads[0].detach()

    def save_logs(self, writer, step, name, no_gradient=False,
                  no_fb_param=False):
        forward_weights_norm = torch.norm(self.weights)
        writer.add_scalar(tag='{}/forward_weights_norm'.format(name),
                          scalar_value=forward_weights_norm,
                          global_step=step)
        if self.weights.grad is not None:
            forward_weights_gradients_norm = torch.norm(self.weights.grad)
            writer.add_scalar(tag='{}/forward_weights_gradients_norm'.format(name),
                              scalar_value=forward_weights_gradients_norm,
                              global_step=step)
        if self.bias is not None:
            forward_bias_norm = torch.norm(self.bias)

            writer.add_scalar(tag='{}/forward_bias_norm'.format(name),
                              scalar_value=forward_bias_norm,
                              global_step=step)
        if self.bias.grad is not None:
            forward_bias_gradients_norm = torch.norm(self.bias.grad)
            writer.add_scalar(tag='{}/forward_bias_gradients_norm'.format(name),
                              scalar_value=forward_bias_gradients_norm,
                              global_step=step)
        if not no_fb_param:
            feedback_weights_norm = torch.norm(self.feedbackweights)
            writer.add_scalar(tag='{}/feedback_weights_norm'.format(name),
                              scalar_value=feedback_weights_norm,
                              global_step=step)
            if not no_gradient and self.feedbackweights.grad is not None:
                feedback_weights_gradients_norm = torch.norm(
                    self.feedbackweights.grad)
                writer.add_scalar(
                    tag='{}/feedback_weights_gradients_norm'.format(name),
                    scalar_value=feedback_weights_gradients_norm,
                    global_step=step)

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 init=False):
        if not init:
            if not no_gradient and self.reconstruction_loss is not None:
                writer.add_scalar(
                    tag='{}/reconstruction_loss'.format(name),
                    scalar_value=self.reconstruction_loss,
                    global_step=step)
        else:
            if not no_gradient and self.reconstruction_loss is not None:
                writer.add_scalar(
                    tag='{}/reconstruction_loss_init'.format(name),
                    scalar_value=self.reconstruction_loss,
                    global_step=step)

    def compute_bp_update(self, loss, retain_graph=False):
        """ Compute the error backpropagation update for the forward
        parameters of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """

        if self.bias is not None:
            grads = torch.autograd.grad(loss, [self.weights, self.bias],
                                        retain_graph=retain_graph)
        else:
            grads = torch.autograd.grad(loss, self.weights,
                                        retain_graph=retain_graph)

        return grads

    def compute_gn_activation_updates(self, output_activation, loss,
                                      damping=0., retain_graph=False,
                                      linear=False):
        """
        Compute the Gauss Newton update for activations of the layer. Target
        propagation tries to approximate these updates by the difference between
        the layer targets and the layer activations.
        Args:
            output_activation (torch.Tensor): The tensor containing the output
                activations of the network for the current mini-batch
            loss (torch.Tensor): The 0D tensor containing the loss value of the
                current mini-batch.
            damping (float): the damping coefficient to damp the GN curvature
                matrix J^TJ. Default: 0.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.

        Returns (torch.Tensor): A tensor containing the Gauss-Newton updates
            for the layer activations of the current mini-batch. The size is
            minibatchsize x layersize

        """
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()
        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        activations_updates = torch.Tensor(activations.shape)
        layersize = activations.view(activations.shape[0], -1).shape[1]

        # compute the GN update for each batch sample separate, as we are now
        # computing 'updates' for the activations of the layer instead of the
        # parameters of the layers
        for batch_idx in range(activations.shape[0]):
            # print(batch_idx)
            #  compute jacobian for one batch sample:
            if batch_idx == activations.shape[0] - 1:
                retain_graph_flag = retain_graph
            else:
                # if not yet at the end of the batch, we should retain the graph
                # used for computing the jacobian, as the graph needs to be
                # reused for the computing the jacobian of the next batch sample
                retain_graph_flag = True
            jacobian = utils.compute_jacobian(activations,
                                              output_activation[batch_idx, :],
                                            retain_graph=retain_graph_flag)
            # torch.autograd.grad only accepts the original input tensor,
            # not a subpart of it. Thus we compute the jacobian to all the
            # batch samples from activations and then select the correct
            # part of it
            jacobian = jacobian[:, batch_idx*layersize:
                                   (batch_idx+1)*layersize]

            gn_updates = utils.compute_damped_gn_update(jacobian,
                                                output_error[batch_idx, :],
                                                        damping)
            activations_updates[batch_idx, :] = gn_updates.view(activations.shape[1:])
        return activations_updates

    def compute_gnt_updates(self, output_activation, loss, h_previous=None, damping=0.,
                            retain_graph=False, linear=False):
        """ Compute the angle with the GNT updates for the parameters of the
        network."""
        gn_activation_update = self.compute_gn_activation_updates(output_activation=output_activation,
                                                                  loss=loss,
                                                                  damping=damping,
                                                                  retain_graph=True,
                                                                  linear=linear)

        if self.bias is not None:
            gnt_grads = torch.autograd.grad(outputs=self.activations,
                                            inputs=[self.weights, self.bias],
                                            grad_outputs=gn_activation_update,
                                            retain_graph=retain_graph)
            return gnt_grads
        else:
            gnt_grads = torch.autograd.grad(outputs=self.activations,
                                            inputs=self.weights,
                                            grad_outputs=gn_activation_update,
                                            retain_graph=retain_graph)
            return gnt_grads

    def get_forward_gradients(self):
        """ Return a tuple containing the gradients of the forward
        parameters."""

        if self.bias is not None:
            return (self.weights.grad, self.bias.grad)
        else:
            return (self.weights.grad, )


class DDTPConvControlLayer(DDTPConvLayer):
    def compute_feedback_gradients(self, h_corrupted, output_corrupted,
                                   output_activation, sigma):
        self.set_feedback_requires_grad(True)
        h_reconstructed = self.propagate_backward(output_corrupted)

        reconstruction_loss = F.mse_loss(h_reconstructed,
                                                 h_corrupted)
        self.save_feedback_gradients(reconstruction_loss)
        self.set_feedback_requires_grad(False)









