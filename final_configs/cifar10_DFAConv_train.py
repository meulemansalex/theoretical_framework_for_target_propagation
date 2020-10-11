config = {
'beta1': 0.9,
'beta2': 0.999,
'epsilon': [1.981969257578602e-07, 1.0002025790642068e-08, 1.7871223223982403e-07, 3.847939630472729e-08],
'lr': [0.000261251891103779, 0.0005709027042137494, 0.000671647950588684, 0.0007906186403687093],
'out_dir': 'logs/cifar/DFAConvCIFAR',
'network_type': 'DFAConvCIFAR',
'initialization': 'xavier_normal',
'fb_activation': 'linear',
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
'forward_wd': 0.,

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