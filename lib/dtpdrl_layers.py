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
from abc import ABC, abstractmethod
import numpy as np
from tensorboardX import SummaryWriter
from lib.dtp_layers import DTPLayer
import torch.nn.functional as F


class DTPDRLLayer(DTPLayer):
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












