config = {
'beta1': 0.9,
'beta2': 0.999,
'epsilon': [1.7193559220924876e-07, 0.00019809119366335256, 1.4745363724867889e-08, 3.0370342703184836e-05],
'lr': [0.000592992422167547, 0.00038001447767611315, 7.373656030831236e-06, 1.0011818742335523e-06],
'out_dir': 'logs/cifar/BPConv',
'network_type': 'BPConvCIFAR',
'initialization': 'xavier_normal',
'target_stepsize': 1.0,

'dataset': 'cifar10',

# ### Training options ###
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'epochs_fb': 0,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 0,
'epochs': 100,
'double_precision': True,

### Network options ###
# 'num_hidden': 3,
# 'size_hidden': 1024,
# 'size_input': 3072,
# 'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,

### Miscellaneous options ###
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,

### Logging options ###
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.
}