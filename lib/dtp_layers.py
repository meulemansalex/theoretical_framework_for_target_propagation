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
network that will be trained by difference target propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import lib.utils as utils
import warnings

class DTPLayer(nn.Module):
    """ An abstract base class for a layer of an MLP that will be trained by the
    differece target propagation method. Child classes should specify which
    activation function is used.

    Attributes:
        weights (torch.Tensor): The forward weight matrix :math:`W` of the layer
        bias (torch.Tensor): The forward bias :math:`\mathbf{b}`
            of the layer.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        feedback_weights (torch.Tensor): The feedback weight matrix :math:`Q`
            of the layer. Warning: if we use the notation of the theoretical
            framework, the feedback weights are actually $Q_{i-1}$ from the
            previous layer!! We do this because this makes the implementation
            of the reconstruction loss and training the feedback weights much
            easier (as g_{i-1} and hence Q_{i-1} needs to approximate
            f_i^{-1}). However for the direct feedback connection layers, it
            might be more logical to let the feedbackweights represent Q_i
            instead of Q_{i-1}, as now only direct feedback connections exist.
        feedback_bias (torch.Tensor): The feedback bias :math:`\mathbf{b}`
            of the layer.
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
        target (torch.Tensor or None): The target for this layer on the current
            minibatch. During normal training, it is not needed
            to save the targets so this attribute will stay None. If the user
            wants to compute the angle between (target - activation) and a
            BP update or GN update, the target needs to be saved in the layer
            object to use later on to compute the angles. The saving happens in
            the backward method of the network.

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
    """

    def __init__(self, in_features, out_features, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 feedback_activation='tanh', initialization='orthogonal'):
        nn.Module.__init__(self)

        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=forward_requires_grad)
        self._feedbackweights = nn.Parameter(torch.Tensor(in_features,
                                                          out_features),
                                             requires_grad=False)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=forward_requires_grad)
            self._feedbackbias = nn.Parameter(torch.Tensor(in_features),
                                              requires_grad=False)
        else:
            self._bias = None
            self._feedbackbias = None

        if initialization == 'orthogonal':
            gain = np.sqrt(6. / (in_features + out_features))
            nn.init.orthogonal_(self._weights, gain=gain)
            nn.init.orthogonal_(self._feedbackweights, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(self._weights)
            nn.init.xavier_uniform_(self._feedbackweights)
        elif initialization == 'xavier_normal':
            nn.init.xavier_normal_(self._weights)
            nn.init.xavier_normal_(self._feedbackweights)
        elif initialization == 'teacher':
            nn.init.xavier_normal_(self._weights, gain=3.)
            nn.init.xavier_normal_(self._feedbackweights)
        else:
            raise ValueError('Provided weight initialization "{}" is not '
                             'supported.'.format(initialization))

        if bias:
            nn.init.constant_(self._bias, 0)
            nn.init.constant_(self._feedbackbias, 0)

        self._activations = None
        self._linearactivations = None
        self._reconstruction_loss = None
        self._forward_activation = forward_activation
        self._feedback_activation = feedback_activation
        self._target = None

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    @property
    def feedbackweights(self):
        """Getter for read-only attribute :attr:`feedbackweights`."""
        return self._feedbackweights

    @property
    def feedbackbias(self):
        """Getter for read-only attribute :attr:`feedbackbias`."""
        return self._feedbackbias

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def linearactivations(self):
        """Getter for read-only attribute :attr:`linearactivations` """
        return self._linearactivations

    @linearactivations.setter
    def linearactivations(self, value):
        """Setter for the attribute :attr:`linearactivations` """
        self._linearactivations = value

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

    @target.setter
    def target(self, value):
        """ Setter for attribute target"""
        self._target = value

    def get_forward_parameter_list(self):
        """ Return forward weights and forward bias if applicable"""
        parameterlist = []
        parameterlist.append(self.weights)
        if self.bias is not None:
            parameterlist.append(self.bias)
        return parameterlist

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

    def compute_vectorized_jacobian(self, a):
        """ Compute the vectorized Jacobian of the forward activation function,
        evaluated at a. The vectorized Jacobian is the vector with the diagonal
        elements of the real Jacobian, as it is a diagonal matrix for element-
        wise functions. As a is a minibatch, the output will also be a
        mini-batch of vectorized Jacobians (thus a matrix).
        Args:
            a (torch.Tensor): linear activations
        """
        if self.forward_activation == 'tanh':
            return 1. - torch.tanh(a)**2
        elif self.forward_activation == 'relu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.
            return J
        elif self.forward_activation == 'leakyrelu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.2
            return J
        elif self.forward_activation == 'linear':
            return torch.ones_like(a)
        elif self.forward_activation == 'sigmoid':
            s = torch.sigmoid(a)
            return s * (1 - s)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def requires_grad(self):
        """ Set require_grad attribute of the activations of this layer to
        True, such that the gradient will be saved in the activation tensor."""
        self._activations.requires_grad = True

    def forward(self, x):
        """Compute the output activation of the layer.

        This method applies first a linear mapping with the
        parameters ``weights`` and ``bias``, after which it applies the
        forward activation function.

        Args:
            x: A mini-batch of size B x in_features with input activations from
            the previous layer or input.

        Returns:
            The mini-batch of output activations of the layer.
        """

        h = x.mm(self.weights.t())
        if self.bias is not None:
            h += self.bias.unsqueeze(0).expand_as(h)
        self.linearactivations = h

        self.activations = self.forward_activationfunction(h)
        return self.activations

    def dummy_forward(self, x):
        """ Same as the forward method, besides that the activations and
        linear activations are not saved in self."""
        h = x.mm(self.weights.t())
        if self.bias is not None:
            h += self.bias.unsqueeze(0).expand_as(h)
        h = self.forward_activationfunction(h)
        return h

    def dummy_forward_linear(self, x):
        """ Propagate the input of the layer forward to the linear activation
        of the current layer (so no nonlinearity applied), without saving the
        linear activations."""
        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)

        return a

    def propagate_backward(self, h):
        """ Propagate the activation h backward through the backward mapping
        function g(h) = t(Q*h + d)
        Args:
            h (torch.Tensor): a mini-batch of activations
        """
        h = h.mm(self.feedbackweights.t())
        if self.feedbackbias is not None:
            h += self.feedbackbias.unsqueeze(0).expand_as(h)
        return self.feedback_activationfunction(h)


    def backward(self, h_target, h_previous, h_current):
        """Compute the target activation for the previous layer, based on the
        provided target.


        Args:
            h_target: a mini-batch of the provided targets for this layer.
            h_previous: the activations of the previous layer, used for the
                DTP correction term.
            h_current: the activations of the current layer, used for the
                DTP correction term.

        Returns:
            h_target_previous: The mini-batch of target activations for
                the previous layer.
        """

        h_target_previous = self.propagate_backward(h_target)
        h_tilde_previous = self.propagate_backward(h_current)
        h_target_previous = h_target_previous + h_previous - h_tilde_previous

        return h_target_previous

    def compute_forward_gradients(self, h_target, h_previous, norm_ratio=1.):
        """ Compute the gradient of the forward weights and bias, based on the
        local mse loss between the layer activation and layer target.
        The gradients are saved in the .grad attribute of the forward weights
        and forward bias.
        Args:
            h_target (torch.Tensor): the DTP target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        vectorized_jacobians = self.compute_vectorized_jacobian(
            self.linearactivations)
        teaching_signal = 2*vectorized_jacobians*(self.activations - h_target)
        batch_size = h_target.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1./batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
        self._weights.grad = weights_grad.detach()

    def set_feedback_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the feedback weights and bias to
        the given value
        Args:
            value (bool): True or False
        """
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')
        self._feedbackweights.requires_grad = value
        if self._feedbackbias is not None:
            self._feedbackbias.requires_grad = value

    def compute_feedback_gradients(self, h_previous_corrupted, sigma):
        """ Compute the gradient of the backward weights and bias, based on the
        local reconstruction loss of a corrupted sample of the previous layer
        activation. The gradients are saved in the .grad attribute of the
        feedback weights and feedback bias."""

        self.set_feedback_requires_grad(True)

        h_current = self.dummy_forward(h_previous_corrupted)
        h = self.propagate_backward(h_current)

        if sigma < 0.02:
            scale = 1/0.02**2
        else:
            scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h, h_previous_corrupted)

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
        if self.feedbackbias is not None:
            grads = torch.autograd.grad(reconstruction_loss, [
                self.feedbackweights, self.feedbackbias], retain_graph=False)
            self._feedbackbias.grad = grads[1].detach()
        else:
            grads = torch.autograd.grad(reconstruction_loss,
                                        self.feedbackweights,
                                        retain_graph=False
                                        )
        self._feedbackweights.grad = grads[0].detach()

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

    def compute_gn_update(self, output_activation, loss, damping=0.,
                          retain_graph=False):
        """
        Compute the Gauss Newton update for the parameters of this layer based
        on the current minibatch.
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

        Returns (tuple): A tuple containing the Gauss Newton updates for the
            forward parameters (at index 0 the weight updates, at index 1
            the bias updates if the layer has a bias)

        """
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()
        parameters = self.get_forward_parameters()
        jacobian = utils.compute_jacobian(parameters, output_activation,
                                          retain_graph=retain_graph)

        gn_updates = utils.compute_damped_gn_update(jacobian, output_error,
                                                    damping)

        if self.bias is not None:
            weight_update_flattened = gn_updates[:self.weights.numel(), :]
            bias_update_flattened = gn_updates[self.weights.numel():, :]
            weight_update = weight_update_flattened.view_as(self.weights)
            bias_update = bias_update_flattened.view_as(self.bias)
            return (weight_update, bias_update)
        else:
            weight_update = gn_updates.view(self.weights.shape)
            return (weight_update, )

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
        layersize = activations.shape[1]

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
                                              output_activation[batch_idx,
                                              :],
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
            activations_updates[batch_idx, :] = gn_updates.view(-1)
        return activations_updates

    def compute_bp_activation_updates(self, loss, retain_graph=False,
                                      linear=False):
        """ Compute the error backpropagation teaching signal for the
        activations of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns (torch.Tensor): A tensor containing the BP updates for the layer
            activations for the current mini-batch.

                """

        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        grads = torch.autograd.grad(loss, activations,
                                    retain_graph=retain_graph)[0].detach()
        return grads

    def compute_gnt_updates(self, output_activation, loss, h_previous, damping=0.,
                            retain_graph=False, linear=False):
        """ Compute the angle with the GNT updates for the parameters of the
        network."""
        gn_activation_update = self.compute_gn_activation_updates(output_activation=output_activation,
                                                                  loss=loss,
                                                                  damping=damping,
                                                                  retain_graph=retain_graph,
                                                                  linear=linear)

        if not linear:
            vectorized_jacobians = self.compute_vectorized_jacobian(
                self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (
                        gn_activation_update)
        else:
            teaching_signal = 2 * gn_activation_update

        batch_size = self.activations.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1. / batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            return (weights_grad, bias_grad)
        else:
            return (weights_grad, )

    def compute_nullspace_relative_norm(self, output_activation, retain_graph=False):
        """ Compute the norm of the components of weights.grad that are in the nullspace
        of the jacobian of the output with respect to weights, relative to the norm of
        weights.grad."""
        J = utils.compute_jacobian(self.weights, output_activation,
                                   structured_tensor=False,
                                   retain_graph=retain_graph)
        weights_update_flat = self.weights.grad.view(-1)
        relative_norm = utils.nullspace_relative_norm(J, weights_update_flat)
        return relative_norm

    def save_logs(self, writer, step, name, no_gradient=False,
                  no_fb_param=False):
        """
        Save logs and plots of this layer on tensorboardX
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            no_fb_param (bool): don't log the feedback parameters

        """
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
            if self.feedbackbias is not None:
                feedback_bias_norm = torch.norm(self.feedbackbias)
                writer.add_scalar(tag='{}/feedback_bias_norm'.format(name),
                                  scalar_value=feedback_bias_norm,
                                  global_step=step)

            if not no_gradient and self.feedbackweights.grad is not None:
                feedback_weights_gradients_norm = torch.norm(
                    self.feedbackweights.grad)
                writer.add_scalar(
                    tag='{}/feedback_weights_gradients_norm'.format(name),
                    scalar_value=feedback_weights_gradients_norm,
                    global_step=step)
                if self.feedbackbias is not None:
                    feedback_bias_gradients_norm = torch.norm(
                        self.feedbackbias.grad)
                    writer.add_scalar(
                        tag='{}/feedback_bias_gradients_norm'.format(name),
                        scalar_value=feedback_bias_gradients_norm,
                        global_step=step)

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 init=False):
        """
        Save logs for one minibatch.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
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

    def get_forward_parameters(self):
        """ Return a list containing the forward parameters."""
        if self.bias is not None:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def get_forward_gradients(self):
        """ Return a tuple containing the gradients of the forward
        parameters."""

        if self.bias is not None:
            return (self.weights.grad, self.bias.grad)
        else:
            return (self.weights.grad, )











