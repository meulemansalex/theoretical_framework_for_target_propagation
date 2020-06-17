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

"""
In here, we define classes for fully connected layers in a multilayer perceptron
network that will be trained by difference target propagation with minimal-norm
updates.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from tensorboardX import SummaryWriter
from lib.dtp_layers import DTPLayer
import torch.nn.functional as F


class DTP2Layer(DTPLayer):
    """
    A class for a second type of difference target propagation layers: the
    parameterization of the feedback mapping is changed to
    g_i(h_{i+1}) = Q_i*t_i(h_{i+1}) + d_i. Besides the feedback mapping, the
    DTP2Layer is equal in functionality to the DTPLayer.
    """

    def propagate_backward(self, h):
        """ Propagate the activation h backward through the backward mapping
        function g(h) = Q*t(h) + d
        Args:
            h (torch.Tensor): a mini-batch of activations
        """
        h = self.feedback_activationfunction(h)
        h = h.mm(self.feedbackweights.t())
        if self.feedbackbias is not None:
            h += self.feedbackbias.unsqueeze(0).expand_as(h)
        return h

class MNDTPLayer(DTP2Layer):
    """
    A class for a minimal-norm DTP layer which computes voltage targets by
    putting the rate targets and activations through the feedback activation
    function before computing the parameter updates. The feedback parameters
    are still trained on the local reconstruction loss based on
    rate activations.
    """

    def compute_forward_gradients(self, h_target, h_previous, norm_ratio=1.):
        """
        Compute the gradient of the forward weights and bias, based on the
        minimal-norm updates shown in equation (74) and (75) of the theoretical
        framework. The voltage targets are computed by putting the rate
        targets and activations through the feedback activation function
        (so we take strategy 2 on page 19 of the theoretical framework).
        Args:
            h_target (torch.Tensor): the DTP target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio: The ratio between the regularizer of the norm of the
            forward weights and the regularizer of the norm of the forward bias.
            See the tau/gamma ratio in theorem 3 of the theoretical framework.

        """

        a_target = self.feedback_activationfunction(h_target)
        a_tilde = self.feedback_activationfunction(self.activations)

        # Note that in DTP, the teaching signal is 2*(h - h_target) as a local
        # MSE loss is used. In here we drop the factor 2, to be conform with the
        # theory.
        teaching_signal = (a_tilde - a_target)
        self.compute_minimal_norm_update(teaching_signal, h_previous,
                                         norm_ratio)


    def compute_minimal_norm_update(self, teaching_signal, h_previous,
                                    norm_ratio):
        """
        Compute the minimal norm update according to equation (74) and (75) of
        the theoretical framework. This method is split from
        `compute_forward_gradients` such that we can reuse it in child classes
        with different teaching signals.
        Args:
            teaching_signal: The difference between the voltage target and
                voltage activation (\Delta a in equations 74 and 75)
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio: The ratio between the regularizer of the norm of the
            forward weights and the regularizer of the norm of the forward bias.
            See the tau/gamma ratio in theorem 3 of the theoretical framework.
        """
        h_previous_norm = h_previous.norm(dim=1)
        batch_size = teaching_signal.shape[0]
        if self.bias is not None:
            step_size = 1. / (h_previous_norm ** 2 + norm_ratio)
            bias_grad = norm_ratio * (step_size.unsqueeze(dim=1) \
                .expand_as(teaching_signal) * teaching_signal).mean(0)
            weights_grad = 1. / batch_size * (step_size.unsqueeze(dim=1) \
                .expand_as(teaching_signal) * teaching_signal).t().\
                mm(h_previous)
            self._bias.grad = bias_grad.detach()
        else:
            step_size = 1. / (h_previous_norm ** 2)
            weights_grad = 1. / batch_size * (step_size.unsqueeze(dim=1) \
                                              .expand_as(
                teaching_signal) * teaching_signal).t(). \
                mm(h_previous)


        self._weights.grad = weights_grad.detach()


class MNDTPDRLayer(MNDTPLayer):
    """
    A class for a minimal-norm DTP layer which computes voltage targets by
    putting the rate targets and activations through the feedback activation
    function before computing the parameter updates. The feedback parameters
    are still trained on the local reconstruction loss based on
    rate activations. It uses the difference reconstruction loss to update its
    feedback parameters.
    """

    def compute_feedback_gradients(self, h_previous_corrupted,
                                   h_current_reconstructed,
                                   h_previous, sigma):
        """
        Compute the gradient of the feedback weights and bias, based on the
        difference reconstruction loss (p16 in theoretical framework). The
        gradients are saved in the .grad attribute of the feedback weights and
        feedback bias.
        Args:
            h_previous_corrupted (torch.Tensor): the initial corrupted sample
                of the previous layer that was propagated forward to the output.
            h_current_reconstructed (torch.Tensor): The reconstruction of the
                corrupted sample (by propagating it backward again in a DTP-like
                fashion to this layer)
            h_previous (torch.Tensor): the initial non-corrupted sample of the
                previous layer
        """
        self.set_feedback_requires_grad(True)

        h_previous_reconstructed = self.backward(h_current_reconstructed,
                                             h_previous,
                                             self.activations)
        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))

        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h_previous_corrupted,
                                         h_previous_reconstructed)
        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)


class DTPDRLayer(DTPLayer):
    """ Class for difference target propagation combined with the
    difference reconstruction loss, but without the minimal norm update."""

    def compute_feedback_gradients(self, h_previous_corrupted,
                                   h_current_reconstructed,
                                   h_previous, sigma):
        """
        Compute the gradient of the feedback weights and bias, based on the
        difference reconstruction loss (p16 in theoretical framework). The
        gradients are saved in the .grad attribute of the feedback weights and
        feedback bias.
        Args:
            h_previous_corrupted (torch.Tensor): the initial corrupted sample
                of the previous layer that was propagated forward to the output.
            h_current_reconstructed (torch.Tensor): The reconstruction of the
                corrupted sample (by propagating it backward again in a DTP-like
                fashion to this layer)
            h_previous (torch.Tensor): the initial non-corrupted sample of the
                previous layer
        """
        self.set_feedback_requires_grad(True)

        h_previous_reconstructed = self.backward(h_current_reconstructed,
                                             h_previous,
                                             self.activations)
        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))

        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h_previous_corrupted,
                                         h_previous_reconstructed)
        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

class MNDTP2DRLayer(MNDTPLayer):
    """ Class for difference target propagation layers that use minimal-norm
    updates and for which the feedback mappings are trained with the
    difference reconstruction loss, based on voltage signals instead of
    rate signals like in MNDTPDRLayer. The feedback path outputs directly
    voltage targets instead of rate targets.

    Warning: unlike in DTP2Layer, the feedback_activation should now be the
    (pseudo-)inverse of the forward activation of the previous layer instead of
    the current layer. Thus when initializing the output layer, make sure
    to give a feedback_activation corresponding with the last hidden layer!
    """

    def propagate_backward(self, a):
        """ Propagate the activation h backward through the backward mapping
        function g(a) = t(Q*a + d)
        Args:
            a (torch.Tensor): a mini-batch of linear activations (targets)
        """
        a = a.mm(self.feedbackweights.t())
        if self.feedbackbias is not None:
            a += self.feedbackbias.unsqueeze(0).expand_as(a)
        return self.feedback_activationfunction(a)

    def compute_forward_gradients(self, a_target, h_previous, norm_ratio=1.):
        """
        Compute the gradient of the forward weights and bias, based on the
        minimal-norm updates shown in equation (74) and (75) of the theoretical
        framework. The voltage targets are computed by putting the rate
        targets and activations through the feedback activation function
        (so we take strategy 2 on page 19 of the theoretical framework).
        Args:
            a_target (torch.Tensor): the DTP voltage target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio: The ratio between the regularizer of the norm of the
            forward weights and the regularizer of the norm of the forward bias.
            See the tau/gamma ratio in theorem 3 of the theoretical framework.

        """

        a = self.linearactivations

        teaching_signal = (a - a_target)
        self.compute_minimal_norm_update(teaching_signal, h_previous,
                                         norm_ratio)

    def compute_feedback_gradients(self, a_previous_corrupted,
                                   a_current_reconstructed,
                                   a_previous, sigma):
        """
        Compute the gradient of the feedback weights and bias, based on the
        difference reconstruction loss (p16 in theoretical framework).
        However, now corrupted linear activations (a) are used to compute the
        difference reconstruction loss, as outlined in strategy one on page 19
        of the theoretical framework. The
        gradients are saved in the .grad attribute of the feedback weights and
        feedback bias.
        Args:
            a_previous_corrupted (torch.Tensor): the initial corrupted sample
                of the previous layer that was propagated forward to the output.
            a_current_reconstructed (torch.Tensor): The reconstruction of the
                corrupted sample (by propagating it backward again in a DTP-like
                fashion to this layer)
            a_previous (torch.Tensor): the initial non-corrupted sample of the
                previous layer that was propagated forward to the output during
                the normal feedforward phase of the network.
        """
        self.set_feedback_requires_grad(True)

        a_previous_reconstructed = self.backward(a_current_reconstructed,
                                             a_previous,
                                             self.linearactivations)

        if sigma <= 0:
            raise ValueError('Sigma should be greater than zero when using the'
                             'difference reconstruction loss. Given sigma = '
                             '{}'.format(sigma))
        scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(a_previous_corrupted,
                                         a_previous_reconstructed)
        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

class GNLayer(DTPLayer):
    pass










