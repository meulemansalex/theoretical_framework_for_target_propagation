config_DMLPDTP2_linear = {
'lr': 0.0000001,
'target_stepsize': 0.0403226567555006,
'feedback_wd': 9.821494271391093e-05,
'lr_fb': 0.0022485520139920064,
'sigma': 0.06086642605203958,
'out_dir': 'logs/STRegression/DMLPDTP2_linear',
'network_type': 'DMLPDTP2',
'recurrent_input': False,
'hidden_fb_activation': 'linear',
'size_mlp_fb': None,
'fb_activation': 'linear',
'initialization': 'xavier_normal',
}

config_DTP_pretrained = {
'lr': 0.0000001,
'target_stepsize': 0.01186235243557516,
'feedback_wd': 1.084514667138376e-05,
'lr_fb': 0.003289433723080337,
'sigma': 0.09999731226483778,
'out_dir': 'logs/mnist/DTP_improved',
'network_type': 'DTP',
'initialization': 'xavier_normal',
'fb_activation': 'tanh',
}

config_collection = {
'DMLPDTP2_linear': config_DMLPDTP2_linear,
'DTP_pretrained': config_DTP_pretrained,
}

result_keys = [
'loss_train',
'loss_test',
'bp_angles',
'nullspace_relative_norm_angles',
'rec_loss',
]

config_fixed = {
'dataset': 'student_teacher',
'optimizer': 'SGD',
'optimizer_fb': 'SGD',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 1,
'forward_wd': 0.0,
'epochs_fb': 30,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
'num_train': 400,
'num_test': 200,
'epochs': 5,
'train_only_feedback_parameters': False,
'freeze_forward_weights': True,
'num_hidden': 2,
'size_hidden': 6,
'size_input': 6,
'size_output': 2,
'hidden_activation': 'tanh',
'output_activation': 'linear',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'save_logs': True,
'save_BP_angle': True,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'save_nullspace_norm_ratio': True,
'gn_damping': 0.0,
'hpsearch': False,
'plots': 'compute',
'log_interval': 10,
}

if __name__ == '__main__':
	 pass