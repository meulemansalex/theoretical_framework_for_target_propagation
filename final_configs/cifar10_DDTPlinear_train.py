config = {
'lr': 3.5281433603406804e-05,
'target_stepsize': 0.04146495013820349,
'feedback_wd': 6.774110390427096e-07,
'beta1': 0.9,
'beta2': 0.9,
'epsilon': 2.5557577819547774e-08,
'lr_fb': 0.00011360417631565426,
'sigma': 0.030404212370394985,
'beta1_fb': 0.99,
'beta2_fb': 0.9,
'epsilon_fb': 1.9322859261177356e-06,
'out_dir': 'logs/mnist/DMLPDTP2_linear',
'network_type': 'DMLPDTP2',
'recurrent_input': False,
'hidden_fb_activation': 'linear',
'size_mlp_fb': None,
'fb_activation': 'linear',
'initialization': 'xavier_normal',
'dataset': 'cifar10',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 10,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 2,
'epochs': 100,
'double_precision': True,
'no_val_set': True,
'num_hidden': 3,
'size_hidden': 1024,
'size_input': 3072,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'log_interval': 80,
}