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
In this python file, the network classes for networks with direct feedback
connections should be implemented.
"""

from lib.direct_feedback_layers import DDTPRHLLayer, \
    DDTPMLPLayer, DDTPControlLayer
from lib.networks import DTPNetwork, DTPDRLNetwork
import torch
import torch.nn as nn
import numpy as np
from lib.utils import NetworkError
import pandas as pd
import warnings
from lib import utils



class DDTPRHLNetwork(DTPDRLNetwork):
    """
    A class for networks that use direct feedback for providing targets
    to the hidden layers by using a shared hidden feedback representation.
    It trains its feedback parameters based on the difference reconstruction
    loss.
    """
    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False, hidden_feedback_dimension=500,
                 hidden_feedback_activation='tanh', initialization='orthogonal',
                 fb_activation='linear', plots=None, recurrent_input=False):
        """

        Args:
            n_in:
            n_hidden:
            n_out:
            activation:
            output_activation:
            bias:
            sigma:
            forward_requires_grad:
            hidden_feedback_dimension: The dimension of the hidden feedback
                layer
            hidden_feedback_activation: The activation function of the hidden
                feedback layer
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
        """
        # we need to overwrite the __init__function, as we need an extra
        # argument for this network class: hidden_feedback_dimension
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out,
                                       activation, output_activation,
                                       bias, forward_requires_grad,
                                       hidden_feedback_dimension,
                                       initialization,
                                       fb_activation,
                                       recurrent_input)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._plots = plots
        self.update_idx = None
        if plots is not None:
            self.bp_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss_init = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.td_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.bp_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
        #TODO: set the requires_grad attribute of the weight and bias of the
        # linear layer to False.
        self._hidden_feedback_layer = nn.Linear(n_out,
                                                hidden_feedback_dimension,
                                                bias=bias)
        self._hidden_feedback_activation_function = \
            self.set_hidden_feedback_activation(
            hidden_feedback_activation)

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, hidden_feedback_dimension,
                   initialization, fb_activation, recurrent_input):
        """
        See documentation of DTPNetwork

        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all)-1):
            if i == len(n_all) - 2:
                hidden_fb_dimension_copy = n_out
                recurrent_input_copy = False
            else:
                hidden_fb_dimension_copy = hidden_feedback_dimension
                recurrent_input_copy = recurrent_input
            layers.append(
                DDTPRHLLayer(n_all[i - 1], n_all[i],
                             bias=bias,
                             forward_requires_grad=forward_requires_grad,
                             forward_activation=activation,
                             feedback_activation=fb_activation,
                             hidden_feedback_dimension=hidden_fb_dimension_copy,
                             initialization=initialization,
                             recurrent_input=recurrent_input_copy)
            )
        layers.append(
            DDTPRHLLayer(n_all[-2], n_all[-1],
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=output_activation,
                         feedback_activation=output_activation,
                         hidden_feedback_dimension=hidden_feedback_dimension,
                         initialization=initialization,
                         recurrent_input=recurrent_input)
        )
        return layers

    def get_feedback_parameter_list(self):
        """

        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the LAST hidden layer does not
            need feedback parameters, so they are not put in the list.

        """
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist.append(layer.feedbackweights)
            if layer.feedbackbias is not None:
                parameterlist.append(layer.feedbackbias)
        return parameterlist

    def set_hidden_feedback_activation(self, hidden_feedback_activation):
        """ Create an activation function corresponding to the
        given string.
        Args:
            hidden_feedback_activation (str): string indicating which
            activation function needs to be created

        Returns (nn.Module): activation function object
        """
        if hidden_feedback_activation == 'linear':
            return nn.Softshrink(lambd=0)
        elif hidden_feedback_activation == 'relu':
            return nn.ReLU()
        elif hidden_feedback_activation == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif hidden_feedback_activation == 'tanh':
            return nn.Tanh()
        elif hidden_feedback_activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("The given hidden feedback activation {} "
                             "is not supported.".format(
                hidden_feedback_activation))

    @property
    def hidden_feedback_layer(self):
        """ getter for attribute hidden_feedback_layer."""
        return self._hidden_feedback_layer

    @property
    def hidden_feedback_activation_function(self):
        """ getter for attribute hidden_feedback_activation"""
        return self._hidden_feedback_activation_function

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target and feed it through the hidden feedback
        layer.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer

        Returns: Mini-batch the hidden_feedback_layer activations resulting from
            the current output target
        """
        # We use the nonlinear activation of the output layer to compute the
        # target, such that the theory with GN optimization is consistent.
        # Note that the current direct TP network implementation expects a
        # Linear target. Therefore, after computing the nonlinear target, we
        # pass it through the exact inverse nonlinearity of the output layer.
        # The exact inverse is only implemented with sigmoid layers and the
        # linear and softmax output layer (as they both use a linear output
        # layer under the hood). For other layers, the exact inverse is not
        # implemented. Hence, when they are used, we'll throw an error to
        # prevent misusage.

        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=self.forward_requires_grad)[0].detach()

        output_targets = output_activations - \
                         target_lr*gradient

        if self.layers[-1].forward_activation in ['sigmoid', 'linear']:
            output_targets = self.layers[-1].feedback_activationfunction(
                output_targets
            )
        else:
            warnings.warn('Forward activation {} not implemented yet.'
                          .format(self.layers[-1].forward_activation))


        hidden_feedback_activations = self.hidden_feedback_layer(output_targets)
        hidden_feedback_activations = self.hidden_feedback_activation_function(
            hidden_feedback_activations
        )

        # print("hfa:", hidden_feedback_activations.shape)

        return hidden_feedback_activations

    def propagate_backward(self, h_target, i):
        """
        Propagate the hidden fb layer target backwards to layer i in a
        DTP-like fashion to provide a nonlinear target for layer i
        Args:
            h_target (torch.Tensor): the hidden feedback activation, resulting
                from the output target
            i: the layer index to which the target must be propagated

        Returns (torch.Tensor): the nonlinear target for layer i

        """

        if i == self.depth - 2:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            h_i_target = self.layers[i].backward(h_target,
                                                 self.layers[i].activations,
                                                 self.layers[-1].linearactivations)
        else:
            a_last = self.layers[-1].linearactivations
            a_feedback_hidden = self.hidden_feedback_layer(a_last)
            h_feedback_hidden = self.hidden_feedback_activation_function(
                a_feedback_hidden
            )
            h_i_target = self.layers[i].backward(h_target,
                                                 self.layers[i].activations,
                                                 h_feedback_hidden)
        return h_i_target

    # def propagate_backward_last_hidden_layer(self, output_target, i):
    #     """ For propagating the output target to the last hidden layer, we
    #     want to have a simple linear mapping (as the real GN target is also
    #     a simple linear mapping) instead of using the random hidden layer."""

    def compute_feedback_gradients(self, i):
        """
        Compute the difference reconstruction loss for layer i and update the
        feedback parameters based on the gradients of this loss.
        """
        self.reconstruction_loss_index = i

        h_corrupted = self.layers[i].activations + \
                      self.sigma * torch.randn_like(
            self.layers[i].activations)

        output_corrupted = self.dummy_forward_linear_output(h_corrupted, i)
        output_noncorrupted = self.layers[-1].linearactivations

        if i == self.depth - 2:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            h_fb_hidden_corrupted = output_corrupted
            h_fb_hidden_noncorrupted = output_noncorrupted
        else:

            a_fb_hidden_corrupted = self.hidden_feedback_layer(
                output_corrupted)  # Voltage!
            h_fb_hidden_corrupted = self.hidden_feedback_activation_function(
                a_fb_hidden_corrupted)


            a_fb_hidden_noncorrupted = self.hidden_feedback_layer(
                output_noncorrupted)  # Voltage!
            h_fb_hidden_noncorrupted = self.hidden_feedback_activation_function(
                a_fb_hidden_noncorrupted)

        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  h_fb_hidden_corrupted,
                                                  h_fb_hidden_noncorrupted,
                                                  self.sigma)

    def compute_dummy_output_target(self, loss, target_lr, retain_graph=False):
        """ Compute a target for the nonlinear activation of the output layer.
        """
        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=(self.forward_requires_grad or retain_graph))[
            0].detach()

        output_targets = output_activations - \
                         target_lr * gradient
        return output_targets

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Propagate the output target backwards through the network until
        layer i. Based on this target, compute the gradient of the forward
        weights and bias of layer i and save them in the parameter tensors.
        Args:
            last:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            i: layer index to which the target needs to be propagated and the
                gradients need to be computed
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): used for the minimal norm update (see theory)
        """

        self.update_idx = i

        if i == self.depth - 1:
            h_target = self.compute_dummy_output_target(loss, target_lr)
            if save_target:
                self.layers[i].target = h_target

            self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[
                                                         i - 1].activations,
                                                     norm_ratio=norm_ratio)
        elif i == self.depth - 2:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            h_target = self.compute_dummy_output_target(loss, target_lr)
            h_target = self.layers[-1].feedback_activationfunction(
                h_target)
            h_target = self.propagate_backward(h_target, i)
            if save_target:
                self.layers[i].target = h_target

            if i == 0: # first hidden layer needs to have the input
                       # for computing gradients
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[i-1].activations,
                                                         norm_ratio=norm_ratio)

        else:
            h_target = self.compute_output_target(loss, target_lr)

            h_target = self.propagate_backward(h_target, i)

            if save_target:
                self.layers[i].target = h_target

            if i == 0: # first hidden layer needs to have the input
                       # for computing gradients
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[i-1].activations,
                                                         norm_ratio=norm_ratio)

    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        """ Compute the targets for all layers and update their forward
         parameters accordingly. """
        # First compute the output target, as that is computed in a different
        # manner from the output target for propagating to the hidden layers.
        output_target = self.compute_dummy_output_target(loss, target_lr,
                                                         retain_graph=True)
        if save_target:
            self.layers[-1].target = output_target

        self.layers[-1].compute_forward_gradients(output_target,
                                                 self.layers[-2].activations,
                                                 norm_ratio=norm_ratio)

        if self.depth > 1:
            # For propagating the output target to the last hidden layer, we
            # want to have a simple linear mapping (as the real GN target is also
            # a simple linear mapping) instead of using the random hidden layer.
            output_target_linear = self.layers[-1].feedback_activationfunction(
                output_target)
            h_target = self.propagate_backward(output_target_linear, self.depth - 2)
            if save_target:
                self.layers[self.depth - 2].target = h_target

            if self.depth - 2 == 0:  # first hidden layer needs to have the input
                # for computing gradients
                self.layers[self.depth - 2].compute_forward_gradients(h_target,
                                                                      self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[self.depth - 2].compute_forward_gradients(h_target,
                                                         self.layers[
                                                             self.depth - 3].activations,
                                                         norm_ratio=norm_ratio)


            # Then compute the hidden feedback layer activation for the output
            # target
            hidden_fb_target = self.compute_output_target(loss, target_lr)
            self.backward_all(hidden_fb_target, save_target=save_target,
                              norm_ratio=norm_ratio)

    def backward_all(self, output_target, save_target=False, norm_ratio=1.):
        """
        Compute the targets for all hidden layers (not output layer) and
        update their forward parameters accordingly.
        """
        for i in range(self.depth - 2):
            h_target = self.propagate_backward(output_target, i)

            if save_target:
                self.layers[i].target = h_target

            if i == 0:  # first hidden layer needs to have the input
                # for computing gradients
                self.layers[i].compute_forward_gradients(h_target, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target,
                                                         self.layers[
                                                             i - 1].activations,
                                                         norm_ratio=norm_ratio)


    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        return DTPNetwork.compute_gn_activation_angle(
            self=self,
            output_activation=output_activation,
            loss=loss,
            damping=damping,
            i=i,
            step=step,
            retain_graph=retain_graph,
            linear=False)

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        return DTPNetwork.compute_bp_activation_angle(self=self,
                                                      loss=loss, i=i,
                                                   retain_graph=retain_graph,
                                                   linear=False)

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        return DTPNetwork.compute_gnt_angle(self=self,
                                            output_activation=output_activation,
                                            loss=loss,
                                            damping=damping,
                                            i=i,
                                            step=step,
                                            retain_graph=retain_graph,
                                            linear=False)


    def dummy_forward_linear_output(self, h, i):
        """ Propagates the nonlinear activation h of layer i forward through
        the network until the linear output activation.
        THE OUTPUT NONLINEARITY IS NOT APPLIED"""
        y = h

        for layer in self.layers[i + 1:-1]:
            y = layer.dummy_forward(y)

        y = self.layers[-1].dummy_forward_linear(y)
        return y


class DDTPMLPNetwork(DDTPRHLNetwork):
    """ Network class for DDTPMLPLayers"""

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False, size_hidden_fb=[100],
                 fb_hidden_activation='tanh', initialization='orthogonal',
                 fb_activation='linear', plots=None, recurrent_input=False
                 ):
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out,
                                       activation, output_activation,
                   bias, forward_requires_grad, size_hidden_fb,
                   initialization, fb_activation, fb_hidden_activation,
                                       recurrent_input)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._plots = plots
        self.update_idx = None
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
            self.td_activation = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.gn_activation = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.bp_activation = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
            self.nullspace_relative_norm = pd.DataFrame(
                columns=[i for i in range(0, self._depth)])
        # TODO: set the requires_grad attribute of the weight and bias of the
        # linear layer to False.

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, size_hidden_fb,
                   initialization, fb_activation, fb_hidden_activation,
                   recurrent_input):
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            if i == len(n_all) - 2:
                hidden_fb_layers = None
                recurrent_input_copy = False
                bias_copy = False
            else:
                hidden_fb_layers = size_hidden_fb
                recurrent_input_copy = recurrent_input
                bias_copy = bias
            layers.append(
                DDTPMLPLayer(n_all[i - 1], n_all[i], n_out,
                             bias=bias_copy,
                             forward_requires_grad=forward_requires_grad,
                             forward_activation=activation,
                             feedback_activation=fb_activation,
                             size_hidden_fb=hidden_fb_layers,
                             fb_hidden_activation=fb_hidden_activation,
                             initialization=initialization,
                             recurrent_input=recurrent_input_copy
                             )
            )
        layers.append(
            DDTPMLPLayer(n_all[-2], n_all[-1], n_out,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=output_activation,
                         feedback_activation=output_activation,
                         size_hidden_fb=size_hidden_fb,
                         fb_hidden_activation=fb_hidden_activation,
                         initialization=initialization,
                         is_output=True
                         )
        )
        return layers

    def get_feedback_parameter_list(self):
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist += [p for p in layer.get_feedback_parameters()]

        return parameterlist

    def compute_output_target(self, loss, target_lr, retain_graph=False):
        """
        Compute the output target for the linear activation of the output
        layer.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the linear activation
                of the output layer
            retain_graph: Flag indicating whether the autograd graph should
                be retained.

        Returns: Mini-batch of output targets
        """
        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=(self.forward_requires_grad or retain_graph))[
            0].detach()

        output_targets = output_activations - \
                         target_lr * gradient

        if self.layers[-1].forward_activation == 'linear':
            output_targets = output_targets

        elif self.layers[-1].forward_activation == 'sigmoid':
            output_targets = utils.logit(
                output_targets)  # apply inverse sigmoid

        else:
            warnings.warn('Forward activation {} not implemented yet.'.format(
                self.layers[-1].forward_activation))

        return output_targets

    def propagate_backward(self, output_target, i):
        """
        Propagate the linear output target backwards to layer i with the
        direct feedback MLP mapping to provide a target for the nonlinear hidden
        layer activation.
        """

        a_output = self.layers[-1].linearactivations

        h_layer_i = self.layers[i].activations

        h_target_i = self.layers[i].backward(output_target, h_layer_i,
                                             a_output)

        return h_target_i

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Compute and propagate the output target to layer i via
        direct feedback MLP connection."""

        self.update_idx = i

        if i != self._depth - 1:
            output_target = self.compute_output_target(loss, target_lr)

            h_target_i = self.propagate_backward(output_target, i)

            if save_target:
                self.layers[i].target = h_target_i

            if i == 0:
                self.layers[i].compute_forward_gradients(h_target_i, self.input,
                                                         norm_ratio=norm_ratio)
            else:
                self.layers[i].compute_forward_gradients(h_target_i,
                                                self.layers[i-1].activations,
                                                         norm_ratio=norm_ratio)

        else:
            output_target = self.compute_dummy_output_target(loss, target_lr)
            h_target_i = output_target
            if save_target:
                self.layers[i].target = h_target_i

            self.layers[i].compute_forward_gradients(h_target_i,
                                                self.layers[i-1].activations,
                                                     norm_ratio=norm_ratio)

    def compute_feedback_gradients(self, i):
        """
        Compute the difference reconstruction loss for layer i of the network
        and compute the gradient of this loss with respect to the feedback
        parameters. The gradients are saved in the .grad attribute of the
        feedback parameter tensors.

        Implementation:
        - get the activation of layer i
        - corrupt the activation of layer i
        - propagate the corrupted activation and noncorrupted activation
          towards the output of the last layer with dummy_forward_linear
        - provide the needed arguments to self.layers[i].compute_feedback_
            gradients
        Args:
            i: the layer index of which the feedback matrices should be updated
        """



        self.reconstruction_loss_index = i

        h_corrupted = self.layers[i].activations + \
                      self.sigma * torch.randn_like(
            self.layers[i].activations)

        output_corrupted = self.dummy_forward_linear_output(h_corrupted, i)
        output_noncorrupted = self.layers[-1].linearactivations


        self.layers[i].compute_feedback_gradients(h_corrupted,
                                                  output_corrupted,
                                                  output_noncorrupted,
                                                  self.sigma)

    def compute_dummy_output_target(self, loss, target_lr, retain_graph=False):
        """ Compute a target for the nonlinear activation of the output layer.
        """
        output_activations = self.layers[-1].activations
        gradient = torch.autograd.grad(
            loss, output_activations,
            retain_graph=(self.forward_requires_grad or retain_graph))[
            0].detach()

        output_targets = output_activations - \
                         target_lr * gradient
        return output_targets

    def dummy_forward_linear_output(self, h, i):
        return DDTPRHLNetwork.dummy_forward_linear_output(self=self,
                                                          h=h,
                                                          i=i)

    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        return DDTPRHLNetwork.compute_gn_activation_angle(
            self=self,
            output_activation=output_activation,
            loss=loss,
            damping=damping,
            i=i,
            step=step,
            retain_graph=retain_graph,
            linear=linear)

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        return DDTPRHLNetwork.compute_bp_activation_angle(self=self,
                                                          loss=loss, i=i,
                                                          retain_graph=retain_graph,
                                                          linear=linear)

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        return DDTPRHLNetwork.compute_gnt_angle(self=self,
                                                output_activation=output_activation,
                                                loss=loss,
                                                damping=damping,
                                                i=i,
                                                step=step,
                                                retain_graph=retain_graph,
                                                linear=linear)

class DDTPControlNetwork(DDTPMLPNetwork):
    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, size_hidden_fb,
                   initialization, fb_activation, fb_hidden_activation,
                   recurrent_input):
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            if i == len(n_all) - 2:
                hidden_fb_layers = None
                recurrent_input_copy = False
            else:
                hidden_fb_layers = size_hidden_fb
                recurrent_input_copy = recurrent_input
            layers.append(
                DDTPControlLayer(n_all[i - 1], n_all[i], n_out,
                             bias=bias,
                             forward_requires_grad=forward_requires_grad,
                             forward_activation=activation,
                             feedback_activation=fb_activation,
                             size_hidden_fb=hidden_fb_layers,
                             fb_hidden_activation=fb_hidden_activation,
                             initialization=initialization,
                             recurrent_input=recurrent_input_copy
                             )
            )
        layers.append(
            DDTPControlLayer(n_all[-2], n_all[-1], n_out,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=output_activation,
                         feedback_activation=output_activation,
                         size_hidden_fb=size_hidden_fb,
                         fb_hidden_activation=fb_hidden_activation,
                         initialization=initialization,
                         is_output=True
                         )
        )
        return layers


