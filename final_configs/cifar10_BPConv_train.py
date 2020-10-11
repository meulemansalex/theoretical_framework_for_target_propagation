config = {
'beta1': 0.9,
'beta2': 0.999,
'epsilon': [2.6463096105390523e-07, 3.715648713579463e-08, 1.5494382092779065e-08, 1.31897234391335e-08],
'lr': [0.000307860008376031, 9.721247612044138e-05, 0.0005367449933863134, 0.0007177383741161405],
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
'epochs': 300,
'double_precision': True,
'no_val_set': True,

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