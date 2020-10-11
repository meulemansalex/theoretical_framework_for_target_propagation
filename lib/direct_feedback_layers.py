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
In this python file, the layer classes for layers with direct (skip) feedback
connections should be implemented.

"""

from lib.dtp_layers import DTPLayer
from lib.dtpdrl_layers import DTPDRLLayer
from lib.networks import BPNetwork
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib import utils


class DDTPRHLLayer(DTPDRLLayer):
    """
    A class for target propagation layers with direct feedback connections from
    the output layer, that use a shared large hidden layer to
    blow up the dimension of the feedback.

    Note that the hidden feedback layer will not be
    implemented in the layer class but in the corresponding network class
    instead, as this hidden feedback layer needs to be shared by all layers
    of the network.

    Attributes:
        weights (torch.Tensor): The forward weight matrix :math:`W` of the layer
        bias (torch.Tensor): The forward bias :math:`\mathbf{b}`
            of the layer.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        feedback_weights(torch.Tensor): The feedback weight matrix Q_i of the
            layer. Note that feedback_weights represent Q_i instead of Q_{i-1}.
            The dimension of the feedback weights will be
            out_features x hidden_feedback_dimension, as it will be connected from
            the large hidden feedback layer towards this layer.
        feedback_bias (torch.Tensor): The feedback bias of the layer of
            dimension out_features.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        forward_requires_grad (bool): Flag indicating whether the computational
            graph with respect to the forward parameters should be saved. This
            flag should be True if you want to compute BP or GN updates. For
            TP updates, computational graphs are not needed (custom
            implementation by ourselves)
        reconstruction_loss (float): The reconstruction loss of this layer
            evaluated at the current mini-batch.
        forward_activation (str): String indicating the forward nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        feedback_activation (str): String indicating the feedback nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.

        Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        forward_activation (str): String indicating the forward nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        feedback_activation (str): String indicating the feedback nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        hidden_feedback_dimension (int): Size of the shared hidden feedback layer
            that blows up the dimension of the output targets.
    """

    def __init__(self, in_features, out_features, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 feedback_activation='tanh', hidden_feedback_dimension=500,
                 initialization='orthogonal',
                 recurrent_input = False):
        # Warning: if the __init__ method of DTPLayer gets new/extra arguments,
        # this should also be incorporated here
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=forward_activation,
                         feedback_activation=feedback_activation,
                         initialization=initialization)

        self._recurrent_input = recurrent_input
        # Now we need to overwrite the initialization of the feedback weights,
        # as we now need Q_i instead of Q_{i-1}, and a direct connection from
        # the hidden feedback layer to hi

        if recurrent_input:
            n_fb = out_features + hidden_feedback_dimension
        else:
            n_fb = hidden_feedback_dimension

        self._feedbackweights = nn.Parameter(torch.Tensor(out_features,
                                                          n_fb),
                                                          requires_grad=False)
        if bias:
            self._feedbackbias = nn.Parameter(torch.Tensor(out_features),
                                              requires_grad=False)
        else:
            self._feedbackbias = None

        if initialization == 'orthogonal':
            gain = np.sqrt(6. / (out_features + hidden_feedback_dimension))
            nn.init.orthogonal_(self._feedbackweights, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(self._feedbackweights)
        elif initialization == 'xavier_normal':
            nn.init.xavier_normal_(self._feedbackweights)
        else:
            raise ValueError('Provided weight initialization "{}" is not '
                             'supported.'.format(initialization))

        if bias:
            nn.init.constant_(self._feedbackbias, 0)

    def propagate_backward(self, fb_hidden_layer_activation):
        if self._recurrent_input:
            in_tensor = torch.cat((fb_hidden_layer_activation,
                                   self.activations), dim=1)
        else:
            in_tensor = fb_hidden_layer_activation
        h = in_tensor.mm(self.feedbackweights.t())
        if self.feedbackbias is not None:
            h += self.feedbackbias.unsqueeze(0).expand_as(h)
        return self.feedback_activationfunction(h)

    def backward(self, h_hidden_target, h_current, h_hidden_activation):
        """
        Compute the target activation for the current layer in a DTP-like
        fashion (see pg. 11), based on the output target and the output activation.
        Args:
            h_hidden_target (torch.Tensor): the output target that is already
                propagated to the large shared hidden feedback layer.
            h_current (torch.Tensor): the activations of the current
                layer, used for the DTP correction term
            h_hidden_activation (torch.Tensor): output forward activation
                transformed to the hidden feedback layer (same comment as above)
                , used for the DTP correction term.

        Returns: a_target_current: The mini-batch of target activations for the
            current layer.
        """
        h_target = self.propagate_backward(h_hidden_target)
        h_tilde = self.propagate_backward(h_hidden_activation)
        h_target_current = h_target + h_current - h_tilde

        return h_target_current


    def compute_feedback_gradients(self, h_current_corrupted,
                                   h_fb_hidden_corrupted,
                                   h_fb_hidden_noncorrupted,
                                   sigma):
        """
        Compute the feedback gradients according to the DRL and save the gradients
        in the .grad attributes of the feedback weights.
        Args:
            h_current_corrupted: noise corrupted activation of current layer
            h_fb_hidden_corrupted:  the hidden fb layer activation resulting
                from h_current_corrupted
            h_fb_hidden_noncorrupted: the hidden fb layer activation resulting
                from the uncorrupted current layer activation.
            sigma: std of noise corruption

        Returns: saves gradients in .grad attribute of feedback weights

        """
        self.set_feedback_requires_grad(True)

        h_current_noncorrupted = self.activations

        h_current_reconstructed = self.backward(
            h_fb_hidden_corrupted,
            h_current_noncorrupted,
            h_fb_hidden_noncorrupted
        )

        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))

        scale = 1 / sigma ** 2
        reconstruction_loss = scale * F.mse_loss(h_current_reconstructed,
                                                 h_current_corrupted)

        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)


class DDTPMLPLayer(DTPDRLLayer):
    """ Direct DTP layer with a fully trained multilayer perceptron (MLP) as direct feedback connections.
    DDTP-linear is a special case of DDTPMLP with a single-layer linear MLP as direct
     feedback connection."""

    def __init__(self, in_features, out_features, size_output, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 feedback_activation='tanh', size_hidden_fb=[100],
                 fb_hidden_activation=None,
                 initialization='orthogonal',
                 is_output=False,
                 recurrent_input=False):
        """

        Args:
            in_features:
            out_features:
            size_output:
            bias:
            forward_requires_grad:
            forward_activation:
            feedback_activation:
            size_hidden_fb:
            fb_hidden_activation:
            initialization:
            is_output:
            recurrent_input (bool): flag indicating whether the nonlinear layer
                activation should be used as a (recurrent) input to the feedback
                MLP that propagates the target to this layer.
        """
        # Warning: if the __init__ method of DTPLayer gets new/extra arguments,
        # this should also be incorporated here
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=forward_activation,
                         feedback_activation=feedback_activation,
                         initialization=initialization)

        # Now we need to overwrite the initialization of the feedback weights,
        # as we now need an MLP as feedback connection from the output towards the
        # current layer. We will overwrite the feedback parameters with None and
        # create an MLP feedback connection

        self._feedbackweights = None
        self._feedbackbias = None
        self._recurrent_input = recurrent_input
        self._is_output = is_output
        self._has_hidden_fb_layers = size_hidden_fb is not None

        if fb_hidden_activation is None:
            fb_hidden_activation = feedback_activation

        if not is_output:
            if recurrent_input:
                n_in = size_output + out_features
            else:
                n_in = size_output
            self._fb_mlp = BPNetwork(n_in=n_in,
                                     n_hidden=size_hidden_fb,
                                     n_out=out_features,
                                     activation=fb_hidden_activation,
                                     output_activation=feedback_activation,
                                     bias=bias,
                                     initialization=initialization)
            self._fb_mlp.set_requires_grad(False)
        else:
            self._fb_mlp = None # output does not need to have a feedback path

    @property
    def feedbackweights(self):
        if not self._has_hidden_fb_layers:
            return self._fb_mlp.layers[0].weight
        else:
            return None

    @property
    def feedbackbias(self):
        if not self._has_hidden_fb_layers:
            return self._fb_mlp.layers[0].bias
        else:
            return None


    def set_feedback_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the all the parameters in the
        feedback MLP to the given value
        Args:
            value (bool): True or False
        """
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')

        self._fb_mlp.set_requires_grad(value)

    def propagate_backward(self, output_target):
        if self._recurrent_input:
            in_tensor = torch.cat((output_target, self.activations), dim=1)
        else:
            in_tensor = output_target
        h = self._fb_mlp(in_tensor)
        return h

    def backward(self, output_target, h_current, output_activation):
        """
        Compute the target linear activation for the current layer in a DTP-like
        fashion, based on the linear output target and the output linear
        activation
        Args:
            output_target: output target
            h_current: activation of the current layer
            output_activation: Output activation

        Returns: target for linear activation of this layer

        """
        h_target = self.propagate_backward(output_target)
        h_tilde = self.propagate_backward(output_activation)
        h_target_current = h_target + h_current - h_tilde

        return h_target_current

    def compute_feedback_gradients(self, h_current_corrupted,
                                   output_corrupted,
                                   output_noncorrupted,
                                   sigma):

        """
        Compute the gradients of the feedback weights and bias, based on the
        difference reconstruction loss.
        The gradients will be saved in the .grad attributes of the feedback
        parameters.

        """

        self.set_feedback_requires_grad(True)

        h_current_noncorrupted = self.activations

        h_current_reconstructed = self.backward(output_corrupted,
                                                h_current_noncorrupted,
                                                output_noncorrupted)

        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))
        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h_current_reconstructed,
                                         h_current_corrupted)

        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

    def save_feedback_gradients(self, reconstruction_loss):
        """
        Compute the gradients of the reconstruction_loss with respect to the
        feedback parameters by help of autograd and save them in the .grad
        attribute of the feedback parameters
        Args:
            reconstruction_loss: the reconstruction loss

        """
        self.reconstruction_loss = reconstruction_loss.item()
        grads = torch.autograd.grad(reconstruction_loss,
                                    self._fb_mlp.parameters(),
                                    retain_graph=False)
        for i, param in enumerate(self._fb_mlp.parameters()):
            param.grad = grads[i].detach()

    def get_feedback_parameters(self):
        return self._fb_mlp.parameters()

    def save_logs(self, writer, step, name, no_gradient=False,
                  no_fb_param=False):
        if self._has_hidden_fb_layers or self._is_output:
            DTPLayer.save_logs(self=self, writer=writer, step=step,
                           name=name, no_gradient=no_gradient,
                           no_fb_param=True)
        else:
            DTPLayer.save_logs(self=self, writer=writer, step=step,
                               name=name, no_gradient=no_gradient,
                               no_fb_param=False)


class DDTPControlLayer(DDTPMLPLayer):
    """ Direct DTP layer that does not use the difference trick in its
    reconstruction loss, as a control for the other methods."""
    def compute_feedback_gradients(self, h_current_corrupted,
                                   output_corrupted,
                                   output_noncorrupted,
                                   sigma):
        self.set_feedback_requires_grad(True)

        h_current_reconstructed = self.propagate_backward(output_corrupted)

        reconstruction_loss = F.mse_loss(h_current_reconstructed,
                                                 h_current_corrupted)

        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)





