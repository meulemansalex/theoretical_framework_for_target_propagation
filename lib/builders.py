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

from lib import networks, direct_feedback_networks
import torch
from torch.utils.data import Dataset
import numpy as np
import random


def regression_cubic_poly(num_train=20, num_test=100, train_domain=(-4,-4),
                          test_domain=(-4, 4), rseed=42):
    r"""Generate a dataset for a 1D regression task with a cubic polynomial.

    The regression task modelled here is :math:`y = x^3 + \epsilon`,
    where :math:`\epsilon \sim \mathcal{N}(0, 9I)`.

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        train_domain (tuple): Input domain for training samples.
        test_domain (tuple): Input domain for training samples.
        rseed (int): To ensure reproducibility, the random seed for the data
            generation should be decoupled from the random seed of the
            simulation. Therefore, a new :class:`numpy.random.RandomState` is
            created for the purpose of generating the data.

    Returns:
        (tuple): Tuple containing:

        - **train_x**: Generated training inputs.
        - **test_x**: Generated test inputs.
        - **train_y**: Generated training outputs.
        - **test_y**: Generated test outputs.

        Data is returned in form of 2D arrays of class :class:`numpy.ndarray`.
    """
    rand = np.random.RandomState(rseed)

    train_x = rand.uniform(low=train_domain[0], high=train_domain[1],
                               size=(num_train, 1))
    test_x = np.linspace(start=test_domain[0], stop=test_domain[1],
                         num=num_test).reshape((num_test, 1))

    map_function = lambda x : (x**3.)
    train_y = map_function(train_x)
    test_y = map_function(test_x)

    # Add noise to training outputs.
    train_eps = rand.normal(loc=0.0, scale=3, size=(num_train, 1))
    train_y += train_eps

    return train_x, test_x, train_y, test_y


def generate_data_from_teacher_network(teacher, n_in, num_train,
                                       num_test=100):
    """
    Generate a dataset by feeding random inputs through the given teacher
    network.
    Args:
        teacher: Teacher network for generating the dataset
        n_in: dimension of the inputs
        num_train: number of needed training samples
        num_test: number of needed test samples

    Returns:
        (tuple): Tuple containing:

        - **train_x**: Generated training inputs.
        - **test_x**: Generated test inputs.
        - **train_y**: Generated training outputs.
        - **test_y**: Generated test outputs.

        Data is returned in form of 2D arrays of class :class:`numpy.ndarray`.

    """
    ### Ensure deterministic computation.

    rand = np.random
    train_x = torch.rand(low=-1, high=1, size=(num_train, n_in))
    test_x = rand.uniform(low=-1, high=1, size=(num_test, n_in))

    train_y = teacher.forward(torch.from_numpy(train_x).float()).detach(). \
        numpy()
    test_y = teacher.forward(torch.from_numpy(test_x).float()).detach(). \
        numpy()

    return train_x, test_x, train_y, test_y


def generate_data_from_teacher(args, num_train=1000, num_test=100, n_in=5, n_out=5,
                               n_hidden=[10,10,10], activation='tanh',
                               device=None, num_val=None):
    """Generate data for a regression task through a teacher model.

    This function generates random input patterns and creates a random MLP
    (fully-connected neural network), that is used as a teacher model. I.e., the
    generated input data is fed through the teacher model to produce target
    outputs. The so produced dataset can be used to train and assess a
    student model. Hence, a learning procedure can be verified by validating its
    capability of training a student network to mimic a given teacher network.

    Input samples will be uniformly drawn from a unit cube.

    .. warning::
        Since this is a synthetic dataset that uses random number generators,
        the generated dataset depends on externally configured random seeds
        (and in case of GPU computation, it also depends on whether CUDA
        operations are performed in a derterministic mode).

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        n_in (int): Passed as argument ``n_in`` to class
            :class:`lib.networks.DTPNetwork`
            when building the teacher model.
        n_out (int): Passed as argument ``n_out`` to class
            :class:`lib.networks.DTPNetwork`
            when building the teacher model.
        n_hidden (list): Passed as argument ``n_hidden`` to class
            :class:`lib.networks.DTPNetwork` when building the teacher model.
        activation (str): Passed as argument ``activation`` to
            class :class:`lib.networks.DTPNetwork` when building the
            teacher model

    Returns:
        See return values of function :func:`regression_cubic_poly`.
    """
    if device is None:
        device = torch.device('cpu')
    if num_val is None:
        num_val = num_test
    # make sure that the same dataset is generated for each run
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    rand = np.random

    train_x = rand.uniform(low=-1, high=1, size=(num_train, n_in))
    test_x = rand.uniform(low=-1, high=1, size=(num_test, n_in))
    val_x = rand.uniform(low=-1, high=1, size=(num_val, n_in))

    # Note: make sure that gain is high, such that the neurons are pushed into
    # nonlinear regime. Otherwise we have a linear dataset
    teacher = networks.DTPNetwork(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                         activation=activation, output_activation='linear',
                         bias=True, initialization='teacher')

    if args.double_precision:
        train_y = teacher.forward(torch.from_numpy(train_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
    else:
        train_y = teacher.forward(torch.from_numpy(train_x).float().to(device))\
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).float().to(device))\
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).float().to(device))\
            .detach().cpu().numpy()

    return train_x, test_x, val_x, train_y, test_y, val_y


def build_network(args):
    """
    Create the network based on the provided command line arguments
    Args:
        args: command line arguments

    Returns: a network

    """
    if args.hidden_layers is None:
        if isinstance(args.size_hidden, list):
            n_hidden = args.size_hidden
        elif isinstance(args.size_hidden, int):
            n_hidden = [args.size_hidden] * args.num_hidden
    else:
        n_hidden = args.hidden_layers

    forward_requires_grad = args.save_BP_angle or args.save_GN_angle or\
                            args.save_GN_activations_angle or \
                            args.save_BP_activations_angle or \
                            args.save_GNT_angle or \
                            args.network_type in ['GN', 'GN2'] or \
                            args.output_space_plot_bp or \
                            args.gn_damping_hpsearch or \
                            args.save_nullspace_norm_ratio
    if args.classification:
        assert (args.output_activation == 'softmax' or \
               args.output_activation == 'sigmoid'), "Output layer should " \
                    "represent probabilities => use softmax or sigmoid!"
        if args.output_activation == 'sigmoid':
            if args.network_type == 'LeeDTP':
                raise ValueError('For the LeeDTP network, only softmax output'
                                 'activtion is supported')
            output_activation = 'sigmoid'
        elif args.output_activation == 'softmax':
            output_activation = 'linear'  # the softmax function is incorporated
                                        # in the loss function in Pytorch
        else:
            assert False
    else:
        output_activation = args.output_activation

    kwargs = {'n_in': args.size_input,
              'n_hidden': n_hidden,
              'n_out': args.size_output,
              'activation': args.hidden_activation,
              'bias': not args.no_bias,
              'sigma': args.sigma,
              'forward_requires_grad': forward_requires_grad,
              'initialization': args.initialization,
              'output_activation': output_activation,
              'fb_activation': args.fb_activation,
              'plots': args.plots,
              }

    if args.network_type == 'LeeDTP':
        net = networks.LeeDTPNetwork(n_in=args.size_input, n_hidden=n_hidden,
                            n_out=args.size_output,
                            activation=args.hidden_activation,
                            bias=not args.no_bias,
                            sigma=args.sigma,
                            initialization=args.initialization,
                            forward_requires_grad=forward_requires_grad)
    elif args.network_type == 'DTP':
        net = networks.DTPNetwork(**kwargs)
    elif args.network_type == 'DTP2':
        net = networks.DTP2Network(**kwargs)
    elif args.network_type == 'MNDTP':
        net = networks.MNDTPNetwork(**kwargs)
    elif args.network_type == 'MNDTPDR':
        net = networks.MNDTPDRNetwork(**kwargs)
    elif args.network_type == 'MNDTP2DR':
        net = networks.MNDTP2DRNetwork(**kwargs)
    elif args.network_type == 'DTPDR':
        net = networks.DTPDRNetwork(**kwargs)
    elif args.network_type == 'DKDTP':
        net = direct_feedback_networks.DirectKernelDTPNetwork(**kwargs,
                          hidden_feedback_activation=args.hidden_fb_activation,
                          hidden_feedback_dimension=args.size_hidden_fb,
                          recurrent_input=args.recurrent_input)
    elif args.network_type == 'DKDTP2':
        net = direct_feedback_networks.DKDTP2Network(**kwargs,
                          hidden_feedback_activation=args.hidden_fb_activation,
                          hidden_feedback_dimension=args.size_hidden_fb,
                          recurrent_input=args.recurrent_input
        )
    elif args.network_type == 'DMLPDTP':
        net = direct_feedback_networks.DMLPDTPNetwork(**kwargs,
                            size_hidden_fb=args.size_mlp_fb,
                            fb_hidden_activation=args.hidden_fb_activation,
                            recurrent_input=args.recurrent_input)
    elif args.network_type == 'DMLPDTP2':
        net = direct_feedback_networks.DMLPDTP2Network(**kwargs,
                              size_hidden_fb=args.size_mlp_fb,
                              fb_hidden_activation=args.hidden_fb_activation,
                              recurrent_input=args.recurrent_input)
    elif args.network_type == 'DMLPDTP3':
        net = direct_feedback_networks.DMLPDTP3Network(**kwargs,
                              size_hidden_fb=args.size_mlp_fb,
                              fb_hidden_activation=args.hidden_fb_activation,
                              recurrent_input=args.recurrent_input)
    elif args.network_type == 'DDTPControl':
        net = direct_feedback_networks.DDTPControlNetwork(**kwargs,
                              size_hidden_fb=args.size_mlp_fb,
                              fb_hidden_activation=args.hidden_fb_activation,
                              recurrent_input=args.recurrent_input)
    elif args.network_type == 'GN':
        net = networks.GNNetwork(**kwargs,
                                 damping=args.gn_damping_training)
    elif args.network_type == 'GN2':
        net = networks.GN2Network(**kwargs,
                                 damping=args.gn_damping_training)
    elif args.network_type == 'BP':
        net = networks.BPNetwork(n_in=args.size_input, n_hidden=n_hidden,
                                 n_out=args.size_output,
                                 activation=args.hidden_activation,
                                 output_activation=output_activation,
                                 bias=not args.no_bias,
                                 initialization=args.initialization)


    else:
        raise ValueError('The provided network type {} is not supported'.format(
            args.network_type
        ))

    return net
