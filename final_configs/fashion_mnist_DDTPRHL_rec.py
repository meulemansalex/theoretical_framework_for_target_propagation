config = {
'lr': 2.9793120523254982e-05,
'target_stepsize': 0.05222150562057032,
'beta1': 0.99,
'beta2': 0.999,
'epsilon': 2.64654942058129e-08,
'lr_fb': 0.00022371330692756543,
'sigma': 0.06409847171006239,
'feedback_wd': 0.008109832782142468,
'beta1_fb': 0.99,
'beta2_fb': 0.9,
'epsilon_fb': 7.59182488242157e-06,
'out_dir': 'logs/mnist/DKDTP2',
'network_type': 'DKDTP2',
'recurrent_input': True,
'hidden_fb_activation': 'tanh',
'fb_activation': 'tanh',
'initialization': 'xavier_normal',
'size_hidden_fb': 1024,
'dataset': 'fashion_mnist',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 6,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
'epochs': 100,
'train_only_feedback_parameters': False,
'num_hidden': 5,
'size_hidden': 256,
'size_input': 784,
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
'log_interval': 5,
}