config = {
'beta1': 0.9,
'beta2': 0.999,
'epsilon': [1.1064155093495904e-07, 1.3370147141514222e-08, 1.4849488795615625e-08, 6.579432537287141e-06],
'lr': [4.512786947960491e-05, 0.0003925906667713482, 5.3583994422517426e-05, 7.349907938676488e-05],
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
'gn_damping': 0.,
'log_interval': 100,
}