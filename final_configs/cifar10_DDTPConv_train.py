config = {
'lr': (1.5395901937079718e-05, 4.252664987376195e-05, 9.011700881717918e-05, 0.00026653695086486183),
'target_stepsize': 0.07688144983085089,
'feedback_wd': 5.751527315358352e-07,
'beta1': 0.9,
'beta2': 0.999,
'epsilon': (7.952762675272583e-06, 3.573159556208438e-06, 1.0425400798717413e-08, 2.0232644009531115e-08),
'lr_fb': 4.142073343374983e-05,
'sigma': 0.18197929046014408,
'beta1_fb': 0.9,
'beta2_fb': 0.999,
'epsilon_fb': 8.070760899188774e-06,
'out_dir': 'logs/cifar/DDTPConvCIFAR',
'network_type': 'DDTPConvCIFAR',
'initialization': 'xavier_normal',
'fb_activation': 'linear',


'dataset': 'cifar10',
# ### Training options ###
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'epochs_fb': 10,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
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