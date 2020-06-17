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

import numpy as np
import torch
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import os
import pandas
import warnings

# from lib import networks, direct_feedback_networks
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt


class RegressionDataset(Dataset):
    """A simple regression dataset.

    Args:
        inputs (numpy.ndarray): The input samples.
        outputs (numpy.ndarray): The output samples.
    """
    def __init__(self, inputs, outputs, double_precision=False):
        assert(len(inputs.shape) == 2)
        assert(len(outputs.shape) == 2)
        assert(inputs.shape[0] == outputs.shape[0])

        if double_precision:
            self.inputs = torch.from_numpy(inputs).to(torch.float64)
            self.outputs = torch.from_numpy(outputs).to(torch.float64)
        else:
            self.inputs = torch.from_numpy(inputs).float()
            self.outputs = torch.from_numpy(outputs).float()

    def __len__(self):
        return int(self.inputs.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_in = self.inputs[idx, :]
        batch_out = self.outputs[idx, :]

        return batch_in, batch_out

def plot_predictions(device, test_loader, net):
    """Plot the predictions of 1D regression tasks.

    Args:
        (....): See docstring of function :func:`main.test`.
    """
    net.eval()

    data = test_loader.dataset

    assert(data.inputs.shape[1] == 1 and data.outputs.shape[1] == 1)

    inputs = data.inputs.detach().cpu().numpy()
    targets = data.outputs.detach().cpu().numpy()
    
    with torch.no_grad():
        # Note, for simplicity, we assume that the dataset is small and we don't
        # have t collect the predictions by iterating over mini-batches.
        predictions = net.forward(data.inputs).detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.title("Predictions in 1D regression task", size=20)

    plt.plot(inputs, targets, color='k', label='Target function',
             linestyle='dashed', linewidth=.5)
    plt.scatter(inputs, predictions, color='r', label='Predictions')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def compute_matrix_angle(C,D):
    """Compute the angle between the two given matrices in degrees.

    Args:
        C:
        D:
    """
    # Assert matrices have same dimensions
    assert C.size() == D.size()

    C_vectorized = C.reshape(-1)
    D_vectorized = D.reshape(-1)

    cosine_angle = C_vectorized.dot(D_vectorized)/\
                   (torch.norm(C_vectorized, p=2)*\
                    torch.norm(D_vectorized, p=2))

    return 180/np.pi*torch.acos(cosine_angle)

def plot_angles(angle_tensor, title, ylabel):
    # angles = torch.var.detach(angle_tensor).numpy()
    rc('text', usetex=True)
    angles = angle_tensor.detach().numpy()
    nb_layers = angles.shape[1]
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    for i in range(nb_layers):
        plt.subplot(nb_layers, 1, i+1)
        plt.plot(angles[:,i])
        plt.ylabel(ylabel % (str(i+1), str(i+1)))
        plt.xlabel('epoch')
        if i==nb_layers-1:
            plt.title('output layer')
        else:
            plt.title('hidden layer {}'.format(i+1))
        plt.subplots_adjust(hspace=0.5)
    plt.show()

def compute_error_angle(net, predictions, targets, linear=False):
    W = net.linear_layers[-1].weights
    B = net.linear_layers[-1].feedbackweights
    output_errors = targets - predictions
    delta_A_BP = output_errors.mm(W)
    delta_A_FA = output_errors.mm(B.t())
    Z = net.linear_layers[-2].activations
    if linear:
        delta_Z_BP = delta_A_BP
        delta_Z_FA = delta_A_FA
    else:
        sigmoid = lambda x: torch.div(1., torch.add(1., torch.exp(-x)))
        delta_Z_BP = torch.mul(delta_A_BP, torch.mul(sigmoid(Z), 1. - sigmoid(Z)))
        delta_Z_FA = torch.mul(delta_A_FA, torch.mul(sigmoid(Z), 1. - sigmoid(Z)))
    inner_products = torch.sum(delta_Z_BP*delta_Z_FA, dim=1)
    delta_Z_BP_norms = torch.norm(delta_Z_BP, p=2, dim=1)
    delta_Z_FA_norms = torch.norm(delta_Z_FA, p=2, dim=1)
    cosines = inner_products/(delta_Z_BP_norms*delta_Z_FA_norms)
    angles = torch.acos(cosines)
    return 180/3.1415*torch.mean(angles)

def accuracy(predictions, labels):
    """
    Compute the average accuracy of the given predictions.
    Inspired on
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    Args:
        predictions (torch.Tensor): Tensor containing the output of the linear
            output layer of the network.
        labels (torch.Tensor): Tensor containing the labels of the mini-batch

    Returns (float): average accuracy of the given predictions
    """

    _, pred_labels = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    return correct/total

def choose_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network

    Returns: optimizer

    """
    forward_optimizer = OptimizerList(args, net)
    feedback_optimizer = choose_feedback_optimizer(args, net)

    return forward_optimizer, feedback_optimizer


def choose_forward_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer

    """
    if args.freeze_BPlayers:
        forward_params = net.get_reduced_forward_parameter_list()
    elif args.network_type == 'BP':
        if args.shallow_training:
            print('Shallow training')
            forward_params = net.layers[-1].parameters()
        elif args.only_train_first_layer:
            print('Only training first layer')
            forward_params = net.layers[0].parameters()
        else:
            forward_params = net.parameters()
    else:
        if args.only_train_first_layer:
            print('Only training first layer')
            forward_params = net.get_forward_parameter_list_first_layer()
        elif args.freeze_output_layer:
            print('Freezing output layer')
            forward_params = net.get_reduced_forward_parameter_list()
        else:
            forward_params = net.get_forward_parameter_list()

    if args.optimizer == 'SGD':
        print('Using SGD optimizer')

        forward_optimizer = torch.optim.SGD(forward_params,
                                            lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.forward_wd)
    elif args.optimizer == 'RMSprop':
        print('Using RMSprop optimizer')

        forward_optimizer = torch.optim.RMSprop(
            forward_params,
            lr=args.lr,
            momentum=args.momentum,
            alpha=0.95,
            eps=0.03,
            weight_decay=args.forward_wd,
            centered=True
        )

    elif args.optimizer == 'Adam':
        print('Using Adam optimizer')

        forward_optimizer = torch.optim.Adam(
            forward_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
            weight_decay=args.forward_wd
        )

    else:
        raise ValueError('Provided optimizer "{}" is not supported'.format(
            args.optimizer
        ))

    return forward_optimizer


def choose_feedback_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network

    Returns: optimizer

    """

    feedback_params = net.get_feedback_parameter_list()


    if args.optimizer_fb == 'SGD':
        feedback_optimizer = torch.optim.SGD(feedback_params,
                                             lr=args.lr_fb,
                                             weight_decay=args.feedback_wd)
    elif args.optimizer_fb == 'RMSprop':

        feedback_optimizer = torch.optim.RMSprop(
            feedback_params,
            lr=args.lr_fb,
            momentum=args.momentum,
            alpha=0.95,
            eps=0.03,
            weight_decay=args.feedback_wd,
            centered=True
        )

    elif args.optimizer_fb == 'Adam':

        feedback_optimizer = torch.optim.Adam(
            feedback_params,
            lr=args.lr_fb,
            betas=(args.beta1_fb, args.beta2_fb),
            eps=args.epsilon_fb,
            weight_decay=args.feedback_wd
        )

    else:
        raise ValueError('Provided optimizer "{}" is not supported'.format(
            args.optimizer
        ))

    return feedback_optimizer


class OptimizerList(object):
    """ A class for stacking a separate optimizer for each layer in a list. If
    no separate learning rates per layer are required, a single optimizer is
    stored in the optimizer list."""

    def __init__(self, args, net):
        if isinstance(args.lr, float):
            forward_optimizer = choose_forward_optimizer(args, net)
            optimizer_list = [forward_optimizer]
        elif isinstance(args.lr, np.ndarray):
            if args.network_type == 'BP':
                raise NetworkError('Multiple learning rates is not yet '
                                   'implemented for BP')
            if args.freeze_BPlayers:
                raise NotImplementedError('freeze_BPlayers not '
                                          'yet supported in '
                                          'OptimizerList')
            else:
                if args.only_train_first_layer:
                    print('Only training first layer')
                    forward_params = \
                        net.get_forward_parameter_list_first_layer()
                elif args.freeze_output_layer:
                    print('Freezing output layer')
                    forward_params = net.get_reduced_forward_parameter_list()
                else:
                    forward_params = net.get_forward_parameter_list()

            if (not args.no_bias and not args.freeze_output_layer and
                len(args.lr)*2 != len(forward_params)) or \
                    (args.no_bias and not args.freeze_output_layer and
                     len(args.lr) != len(forward_params)):
                raise NetworkError('The lenght of the list with learning rates '
                                   'does not correspond with the size of the '
                                   'network.')
            if not args.optimizer == 'SGD':
                raise NetworkError('multiple learning rates are only supported '
                                   'for SGD optimizer')

            optimizer_list = []
            for i, lr in enumerate(args.lr):
                if args.no_bias:
                    parameters = [net.layers[i].weights]
                else:
                    parameters = [net.layers[i].weights, net.layers[i].bias]
                optimizer = torch.optim.SGD(parameters,
                                            lr=lr, momentum=args.momentum,
                                            weight_decay=args.forward_wd)
                optimizer_list.append(optimizer)
        else:
            raise ValueError('Command line argument lr={} is not recognized '
                             'as a float'
                       'or list'.format(args.lr))

        self._optimizer_list = optimizer_list

    def zero_grad(self):
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()

    def step(self, i=None):
        """
        Perform a step on the optimizer of layer i. If i is None, a step is
        performed on all optimizers.
        """
        if i is None:
            for optimizer in self._optimizer_list:
                optimizer.step()
        else:
            self._optimizer_list[i].step()


def save_logs(writer, step, net, loss, accuracy, test_loss, test_accuracy,
              val_loss, val_accuracy):
    """
    Save logs and plots to tensorboardX
    Args:
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net: network
        loss: current loss of the training iteration

    """
    net.save_logs(writer, step)
    writer.add_scalar(tag='training_metrics/loss',
                      scalar_value=loss,
                      global_step=step)
    writer.add_scalar(tag='training_metrics/test_loss',
                      scalar_value=test_loss,
                      global_step=step)
    if val_loss is not None:
        writer.add_scalar(tag='training_metrics/val_loss',
                          scalar_value=val_loss,
                          global_step=step)
    if accuracy is not None:
        writer.add_scalar(tag='training_metrics/accuracy',
                          scalar_value=accuracy,
                          global_step=step)
        writer.add_scalar(tag='training_metrics/test_accuracy',
                          scalar_value=test_accuracy,
                          global_step=step)
        if val_accuracy is not None:
            writer.add_scalar(tag='training_metrics/val_accuracy',
                              scalar_value=val_accuracy,
                              global_step=step)

def save_forward_batch_logs(args, writer, step, net, loss, output_activation):
    """
    Save logs and plots for the current mini-batch on tensorboardX
    Args:
        args (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        loss (torch.Tensor): loss of the current minibatch
        output_activation (torch.Tensor): output of the network for the current
            minibatch
    """
    if args.save_BP_angle:
        retain_graph = args.save_GN_angle or args.save_GN_activations_angle or \
            args.save_BP_activations_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio
        # if we need to compute and save other angles afterwards, the gradient
        # graph should be retained such that it can be reused
        net.save_bp_angles(writer, step, loss, retain_graph)
    if args.save_GN_angle:
        retain_graph = args.save_GN_activations_angle or \
                       args.save_BP_activations_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio
        net.save_gn_angles(writer, step, output_activation, loss,
                           args.gn_damping, retain_graph)
    if args.save_BP_activations_angle:
        retain_graph = args.save_GN_activations_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio
        net.save_bp_activation_angle(writer, step, loss, retain_graph)

    if args.save_GN_activations_angle:
        retain_graph = args.save_GNT_angle or args.save_nullspace_norm_ratio
        net.save_gn_activation_angle(writer, step, output_activation, loss,
                                     args.gn_damping, retain_graph)
    if args.save_GNT_angle:
        retain_graph = args.save_nullspace_norm_ratio
        net.save_gnt_angles(writer, step, output_activation, loss,
                            args.gn_damping, retain_graph)

    if args.save_nullspace_norm_ratio:
        retain_graph = False
        net.save_nullspace_norm_ratio(writer, step, output_activation,
                                  retain_graph)



def save_feedback_batch_logs(args, writer, step, net, init=False):
    """
    Save logs and plots for the current mini-batch on tensorboardX
    Args:
        args (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
    """
    net.save_feedback_batch_logs(writer, step, init=init)


def save_gradient_hook(module, grad_input, grad_output):
    """ A hook that will be used to save the gradients the loss with respect
             to the output of the network. This gradient is used to compute the
              target for the output layer."""
    print('save grad in module')
    module.output_network_gradient = grad_input[0]

def compute_jacobian(input, output, structured_tensor=False,
                     retain_graph=False):
    """
    Compute the Jacobian matrix of output with respect to input. If input
    and/or output have more than one dimension, the Jacobian of the flattened
    output with respect to the flattened input is returned if
    structured_tensor is False. If structured_tensor is True, the Jacobian is
    structured in dimensions output_shape x flattened input shape. Note that
    output_shape can contain multiple dimensions.
    Args:
        input (list or torch.Tensor): Tensor or sequence of tensors
            with the parameters to which the Jacobian should be
            computed. Important: the requires_grad attribute of input needs to
            be True while computing output in the forward pass.
        output (torch.Tensor): Tensor with the values of which the Jacobian is
            computed
        structured_tensor (bool): A flag indicating if the Jacobian
            should be structured in a tensor of shape
            output_shape x flattened input shape instead of
            flattened output shape x flattened input shape.


    Returns (torch.Tensor): 2D tensor containing the Jacobian of output with
        respect to input if structured_tensor is False. If structured_tensor
        is True, the Jacobian is structured in a tensor of shape
        output_shape x flattened input shape.
    """
    # We will work with a sequence of input tensors in the following, so if
    # we get one Tensor instead of a sequence of tensors as input, make a
    # list out of it.
    if isinstance(input, torch.Tensor):
        input = [input]

    output_flat = output.view(-1)
    numel_input = 0
    for input_tensor in input:
        numel_input += input_tensor.numel()
    jacobian = torch.Tensor(output.numel(), numel_input)

    for i, output_elem in enumerate(output_flat):

        if i == output_flat.numel() - 1:
            # in the last autograd call, the graph should be retained or not
            # depending on our argument retain_graph.
            gradients = torch.autograd.grad(output_elem, input,
                                            retain_graph=retain_graph,
                                            create_graph=False,
                                            only_inputs=True)
        else:
            # if it is not yet the last autograd call, retain_graph should be
            # True such that the remainder parts of the jacobian can be
            # computed.
            gradients = torch.autograd.grad(output_elem, input,
                                            retain_graph=True,
                                            create_graph=False,
                                            only_inputs=True)
        jacobian_row = torch.cat([g.view(-1).detach() for g in gradients])
        jacobian[i, :] = jacobian_row

    if structured_tensor:
        shape = list(output.shape)
        shape.append(-1) # last dimension can be inferred from the jacobian size
        jacobian = jacobian.view(shape)

    return jacobian

def compute_damped_gn_update(jacobian, output_error, damping):
    """
    Compute the damped Gauss-Newton update, based on the given jacobian and
    output error.
    Args:
        jacobian (torch.Tensor): 2D tensor containing the Jacobian of the
            flattened output with respect to the flattened parameters for which
            the GN update is computed.
        output_error (torch.Tensor): tensor containing the gradient of the loss
            with respect to the output layer of the network.
        damping (float): positive damping hyperparameter

    Returns: the damped Gauss-Newton update for the parameters for which the
        jacobian was computed.

    """
    if damping < 0:
        raise ValueError('Positive value for damping expected, got '
                         '{}'.format(damping))
    # The jacobian also flattens the  output dimension, so we need to do
    # the same.
    output_error = output_error.view(-1, 1).detach()

    if damping == 0:
        # if the damping is 0, the curvature matrix C=J^TJ can be
        # rank deficit. Therefore, it is numerically best to compute the
        # pseudo inverse explicitly and after that multiply with it.
        jacobian_pinv = torch.pinverse(jacobian)
        gn_updates = jacobian_pinv.mm(output_error)
    else:
        # If damping is greater than 0, the curvature matrix C will be
        # positive definite and symmetric. Numerically, it is the most
        # efficient to use the cholesky decomposition to compute the
        # resulting Gauss-newton updates

        # As (J^T*J + l*I)^{-1}*J^T = J^T*(JJ^T + l*I)^{-1}, we select
        # the one which is most computationally efficient, depending on
        # the number of rows and columns of J (we want to take the inverse
        # of the smallest possible matrix, as this is the most expensive
        # operation. Note that we solve a linear system with cholesky
        # instead of explicitly computing the inverse, as this is more
        # efficient.
        if jacobian.shape[0] >= jacobian.shape[1]:
            G = jacobian.t().mm(jacobian)
            C = G + damping * torch.eye(G.shape[0])
            C_cholesky = torch.cholesky(C)
            jacobian_error = jacobian.t().matmul(output_error)
            gn_updates = torch.cholesky_solve(jacobian_error, C_cholesky)
        else:
            G = jacobian.mm(jacobian.t())
            C = G + damping * torch.eye(G.shape[0])
            C_cholesky = torch.cholesky(C)
            inverse_error = torch.cholesky_solve(output_error, C_cholesky)
            gn_updates = jacobian.t().matmul(inverse_error)

    return gn_updates

def compute_angle(A, B):
    """
     Compute the angle between two tensors of the same size. The tensors will
     be flattened, after which the angle is computed.
    Args:
        A (torch.Tensor): First tensor
        B (torch.Tensor): Second tensor

    Returns: The angle between the two tensors in degrees

    """
    if contains_nan(A):
        print('tensor A contains nans:')
        print(A)
    if contains_nan(B):
        print('tensor B contains nans:')
        print(B)

    inner_product = torch.sum(A*B)  #equal to inner product of flattened tensors
    cosine = inner_product/(torch.norm(A, p='fro')*torch.norm(B, p='fro'))
    if contains_nan(cosine):
        print('cosine contains nans:')
        print('inner product: {}'.format(inner_product))
        print('norm A: {}'.format(torch.norm(A, p='fro')))
        print('norm B: {}'.format(torch.norm(B, p='fro')))

    if cosine > 1 and cosine < 1 + 1e-5:
        cosine = torch.Tensor([1.])
    angle = 180/np.pi*torch.acos(cosine)
    if contains_nan(angle):
        print('angle computation causes NANs. cosines:')
        print(cosine)
    return angle

def compute_average_batch_angle(A, B):
    """
    Compute the average of the angles between the mini-batch samples of A and B.
    If the samples of the mini-batch have more than one dimension (minibatch
    dimension not included), the tensors will first be flattened
    Args:
        A (torch.Tensor):  A tensor with as first dimension the mini-batch
            dimension
        B (torch.Tensor): A tensor of the same shape as A

    Returns: The average angle between the two tensors in degrees.

    """

    A = A.flatten(1, -1)
    B = B.flatten(1, -1)
    inner_products = torch.sum(A*B, dim=1)
    A_norms = torch.norm(A, p=2, dim=1)
    B_norms = torch.norm(B, p=2, dim=1)
    cosines = inner_products/(A_norms*B_norms)
    cosines = torch.min(cosines, torch.ones_like(cosines))
    angles = torch.acos(cosines)
    return 180/np.pi*torch.mean(angles)


class NetworkError(Exception):
    pass

def list_to_str(list_arg, delim=' '):
    """Convert a list of numbers into a string.

    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.

    Returns:
        List converted to string.
    """
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret

def str_to_list(string, delim=',', type='float'):
    """ Convert a str (that originated from a list) back
    to a list of floats."""

    if string[0] in ('[', '(') and string[-1] in (']', ')'):
        string = string[1:-1]
    if type == 'float':
        lst = [float(num) for num in string.split(delim)]
    elif type == 'int':
        lst = [int(num) for num in string.split(delim)]
    else:
        raise ValueError('type {} not recognized'.format(type))

    return lst



def setup_summary_dict(args):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This method adds the keyword "summary" to `shared`.

    Args:
        config: Command-line arguments.
        shared: Miscellaneous data shared among training functions (summary dict
            will be added to this :class:`argparse.Namespace`).
        experiment: Type of experiment. See argument `experiment` of method
            :func:`probabilistic.prob_mnist.train_bbb.run`.
        mnet: Main network.
        hnet (optional): Hypernetwork.
    """
    if args.hpsearch:
        from hpsearch import hpsearch_config

    summary = dict()

    if args.hpsearch:
        summary_keys = hpsearch_config._SUMMARY_KEYWORDS
    else:
        summary_keys = [
                        # 'acc_train',
                        'acc_train_last',
                        'acc_train_best',
                        # 'loss_train',
                        'loss_train_last',
                        'loss_train_best',
                        # 'acc_test',
                        'acc_test_last',
                        'acc_test_best',
                        'acc_val_last',
                        'acc_val_best',
                        'acc_test_val_best',
                        'acc_train_val_best',
                        'loss_test_val_best',
                        'loss_train_val_best',
                        'loss_val_best',
                        'epoch_best_loss',
                        'epoch_best_acc',
                        # 'loss_test',
                        'loss_test_last',
                        'loss_test_best',
                        'rec_loss',
                        'rec_loss_last',
                        'rec_loss_best',
                        'rec_loss_first',
                        'rec_loss_init',
                        # 'rec_loss_var',
                        'rec_loss_var_av',
                        'finished']


    for k in summary_keys:
        if k == 'finished':
            summary[k] = 0

        else:
            summary[k] = -1
    save_summary_dict(args, summary)

    return summary


def save_summary_dict(args, summary):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        args (Namespace): command line inputs
        summary (dict): summary dictionary
    """

    if args.hpsearch:
        from hpsearch import hpsearch_config
        summary_fn = hpsearch_config._SUMMARY_FILENAME
    else:
        summary_fn = 'performance_overview.txt'
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    with open(os.path.join(args.out_dir, summary_fn), 'w') as f:
        for k, v in summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            elif isinstance(v, (np.ndarray, pandas.DataFrame)):
                # we don't want to save arrays and dataframes to text files
                pass
            else:
                f.write('%s %d\n' % (k, v))

def get_av_reconstruction_loss(network):
    """ Get the average reconstruction loss of the network across its layers
    for the current mini-batch.
    Args:
        network (networks.DTPNetwork): network
    Returns (torch.Tensor):
        Tensor containing a scalar of the average reconstruction loss
    """
    reconstruction_losses = np.array([])

    for layer in network.layers[1:]:
        reconstruction_losses = np.append(reconstruction_losses,
                                           layer.reconstruction_loss)

        reconstruction_losses = list(filter(lambda x: x != None, reconstruction_losses))

    return np.mean(reconstruction_losses[:-1])


def int_to_one_hot(class_labels, nb_classes, device, soft_target=1.):
    """ Convert tensor containing a batch of class indexes (int) to a tensor
    containing one hot vectors."""
    one_hot_tensor = torch.zeros((class_labels.shape[0], nb_classes),
                                 device=device)
    for i in range(class_labels.shape[0]):
        one_hot_tensor[i, class_labels[i]] = soft_target

    return one_hot_tensor


def one_hot_to_int(one_hot_tensor):
    return torch.argmax(one_hot_tensor, 1)


def dict2csv(dct, file_path):
    with open(file_path, 'w') as f:
        for key in dct.keys():
            f.write("{}, {} \n".format(key, dct[key]))


def process_lr(lr_str):
    """
    Process the lr provided by argparse.
    Args:
        lr_str (str): a string containing either a single float indicating the
            learning rate, or a list of learning rates, one for each layer of
            the network.
    Returns: a float or a numpy array of learning rates
    """
    if ',' in lr_str:
        return np.array(str_to_list(lr_str, ','))
    else:
        return float(lr_str)

def process_hdim(hdim_str):
    if ',' in hdim_str:
        return str_to_list(hdim_str, ',', type='int')
    else:
        return int(hdim_str)

def process_hdim_fb(hdim_str):
    if ',' in hdim_str:
        return str_to_list(hdim_str, ',', type='int')
    else:
        return [int(hdim_str)]

def check_gpu():
    try:
        name = torch.cuda.current_device()
        print("Using CUDA device {}.".format(torch.cuda.get_device_name(name)))
    except AssertionError:
        print("No CUDA device found.")


def contains_nan(tensor):
    nb_nans = tensor != tensor
    nb_infs = tensor == float('inf')
    if isinstance(nb_nans, bool):
        return nb_nans or nb_infs
    else:
        return torch.sum(nb_nans) > 0 or torch.sum(nb_infs) > 0


def logit(x):
    if torch.sum(x < 1e-12) > 0 or torch.sum(x > 1 - 1e-12) > 0:
        warnings.warn('Input to inverse sigmoid is out of'
                      'bound: x={}'.format(x))
    inverse_sigmoid = torch.log(x / (1 - x))
    if contains_nan(inverse_sigmoid):
        raise ValueError('inverse sigmoid function outputted a NaN')
    return torch.log(x / (1 - x))


def plot_loss(summary, logdir, logplot=False):
    plt.figure()
    plt.plot(summary['loss_train'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss')
    if logplot:
        plt.yscale('log')
    plt.savefig(os.path.join(logdir, 'loss_train.svg'))
    plt.close()
    plt.figure()
    plt.plot(summary['loss_test'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Test loss')
    if logplot:
        plt.yscale('log')
    plt.savefig(os.path.join(logdir, 'loss_test.svg'))
    plt.close()


def make_plot_output_space(args, net, i, loss_function,
                           targets, inputs, steps=20):
    """
    Make a plot of how the output activations would change if the update 
    for the parameters of layer(s) i is applied with a varying stepsize from 
    zero to one. 
    Args:
        args: command line arguments
        net: network
        i: layer index. If None, all layers are updated
        loss_function: loss function
        targets: true labels for the current batch
        inputs: batch with inputs for the network
        steps: amount of interpolation steps

    Returns: Saves a plot and the sequence of activations
    """

    if args.output_space_plot_bp:
        args.network_type = 'BP'

    # take the first input sample from the batch
    inputs = inputs.flatten(1, -1)
    inputs = inputs[0:1, :]
    targets = targets[0:1, :]

    # Get the parameters
    if i is None:
        parameters = net.get_forward_parameter_list()

    else:
        parameters = net.layers[i].get_forward_parameter_list()

    alpha = 1e-5
    sgd_optimizer = torch.optim.SGD(parameters, lr=alpha)
    sgd_optimizer.zero_grad()

    # compute update based on the single input sample
    predictions = net.forward(inputs)
    loss = loss_function(predictions, targets)

    if args.output_space_plot_bp:
        gradients = torch.autograd.grad(loss, parameters)
        for i, param in enumerate(parameters):
            param.grad = gradients[i].detach()
    else:
        net.backward(loss, args.target_stepsize, save_target=False,
                     norm_ratio=args.norm_ratio)


    # compute the start output value
    output_start = net.forward(inputs)

    # compute the output value after a very small step size
    sgd_optimizer.step()
    output_next = net.forward(inputs)

    output_update = (output_next - output_start)[0, 0:2].detach().cpu().numpy()


    # Make the plot
    ax = plt.axes()
    plot_contours(output_start[0, 0:2], targets[0, 0:2], loss_function, ax)

    # dimensions
    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              targets.detach().cpu().numpy())
    x_low = targets[0, 0].detach().cpu().numpy() - 1.1 * distance
    x_high = targets[0, 0].detach().cpu().numpy() + 1.1 * distance
    y_low = targets[0, 1].detach().cpu().numpy() - 1.1 * distance
    y_high = targets[0, 1].detach().cpu().numpy() + 1.1 * distance

    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    # make the output arrow:
    output_arrow = distance / 2 / np.linalg.norm(output_update) * output_update
    output_arrow_start = output_start[0, 0:2].detach().cpu().numpy()

    ax.arrow(output_arrow_start[0], output_arrow_start[1],
              output_arrow[0], output_arrow[1],
              width=0.05,
              head_width=0.3
              )

    file_name = 'output_space_updates_fig_' + args.network_type + '.svg'
    plt.savefig(os.path.join(args.out_dir, file_name))
    plt.close()
    file_name = 'output_arrow_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow)
    file_name = 'output_arrow_start_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow_start)
    file_name = 'output_space_label_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            targets[0, 0:2].detach().cpu().numpy())


def plot_contours(y, label, loss_function, ax, fontsize=26):
    """
    Make a 2D contour plot of loss_function(y, targets)
    """
    gridpoints = 100

    distance = np.linalg.norm(y.detach().cpu().numpy() -
                              label.detach().cpu().numpy())
    y1 = np.linspace(label[0].detach().cpu().numpy() - 1.1*distance,
                     label[0].detach().cpu().numpy() + 1.1*distance,
                     num=gridpoints)
    y2 = np.linspace(label[1].detach().cpu().numpy() - 1.1*distance,
                     label[1].detach().cpu().numpy() + 1.1*distance,
                     num=gridpoints)

    Y1, Y2 = np.meshgrid(y1, y2)

    L = np.zeros(Y1.shape)
    for i in range(gridpoints):
        for j in range(gridpoints):
            y_sample = torch.Tensor([Y1[i,j], Y2[i, j]])
            L[i,j] = loss_function(y_sample, label).item()

    levels = np.linspace(1.01*L.min(), L.max(), num=9)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    CS = ax.contour(Y1, Y2, L, levels=levels)
    # , colors = 'tab:blue'


def make_plot_output_space_bp(args, net, i, loss_function,
                           targets, inputs, steps=20):
    """
    Make a plot of how the output activations would change if the update
    for the parameters of layer(s) i is applied with a varying stepsize from
    zero to one.
    Args:
        args: command line arguments
        net: network
        i: layer index. If None, all layers are updated
        loss_function: loss function
        targets: true labels for the current batch
        inputs: batch with inputs for the network
        steps: amount of interpolation steps

    Returns: Saves a plot and the sequence of activations
    """

    # Get the parameters
    if i is None:
        parameters = net.parameters()

    else:
        parameters = net.layers[i].parameters()

    # create sgd optimizer
    alpha = 1e-5
    sgd_optimizer = torch.optim.SGD(parameters, lr=alpha)
    sgd_optimizer.zero_grad()

    # take the first input sample from the batch
    inputs = inputs.flatten(1, -1)
    inputs = inputs[0:1, :]
    targets = targets[0:1, :]

    # compute update based on the single input sample
    predictions = net(inputs)
    loss = loss_function(predictions, targets)
    loss.backward()



    # Interpolate the output trajectory
    output_start = net(inputs)

    sgd_optimizer.step()
    output_next = net(inputs)

    output_update = (output_next - output_start)[0, 0:2].detach().cpu().numpy()


    # Make the plot
    ax = plt.axes()
    plot_contours(output_start[0, 0:2], targets[0, 0:2], loss_function, ax)

    # dimensions
    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              targets.detach().cpu().numpy())
    x_low = targets[0,0].detach().cpu().numpy() - 1.1 * distance
    x_high = targets[0, 0].detach().cpu().numpy() + 1.1 * distance
    y_low = targets[0, 1].detach().cpu().numpy() - 1.1 * distance
    y_high = targets[0, 1].detach().cpu().numpy() + 1.1 * distance

    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    # make the output arrow:
    output_arrow = distance / 2 / np.linalg.norm(output_update) * output_update
    output_arrow_start = output_start[0, 0:2].detach().cpu().numpy()

    ax.arrow(output_arrow_start[0], output_arrow_start[1],
              output_arrow[0], output_arrow[1],
              width=0.05,
              head_width=0.3
              )

    file_name = 'output_space_updates_fig_' + args.network_type + '.svg'
    plt.savefig(os.path.join(args.out_dir, file_name))
    plt.close()
    file_name = 'output_arrow_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow)
    file_name = 'output_arrow_start_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow_start)
    file_name = 'output_space_label_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            targets[0, 0:2].detach().cpu().numpy())


def nullspace(A, tol=1e-12):
    U, S, V = torch.svd(A, some=False)
    if S.min() >= tol:
        null_start = len(S)
    else:
        null_start = int(len(S) - torch.sum(S<tol))

    V_null = V[:, null_start:]
    return V_null


def nullspace_relative_norm(A, x, tol=1e-12):
    """
    Compute the ratio between the norm
    of components of x that are in the nullspace of A
    and the norm of x
    """

    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    A_null = nullspace(A, tol=tol)
    x_null_coordinates = A_null.t().mm(x)
    ratio = x_null_coordinates.norm()/x.norm()
    return ratio


if __name__ == '__main__':
    pass


