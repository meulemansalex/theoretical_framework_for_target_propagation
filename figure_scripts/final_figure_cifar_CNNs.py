config_DDTPConv = {
'beta1': 0.9,
'beta1_fb': 0.9,
'beta2': 0.999,
'beta2_fb': 0.999,
'epsilon': [2.7867895625009e-08, 1.9868935703787622e-08, 4.515242618159344e-06, 4.046144976139705e-05],
'epsilon_fb': 7.529093372180766e-07,
'feedback_wd': 6.169295107849636e-05,
'lr': [0.00025935571806476586, 0.000885500279951265, 0.0001423047695105589, 3.3871035558126015e-06],
'lr_fb': 0.0045157498494467095,
'sigma': 0.00921040366516759,
'target_stepsize': 0.015962099947441903,
'dataset': 'cifar10',
'out_dir': 'logs/DDTPConv_cifarCIFAR_figure',
'network_type': 'DDTPConvCIFAR',
'initialization': 'xavier_normal',
'fb_activation': 'linear',
'gn_damping': [0.1, 0.1, 0.0001],
}

config_DDTPControlConv = {
'lr': (0.0007058630744815441, 0.00045976866944110974, 1.4142718290111939e-05, 1.9346548814094763e-05),
'target_stepsize': 0.09962995994670613,
'feedback_wd': 6.731653502569897e-07,
'beta1': 0.9,
'beta2': 0.999,
'epsilon': (1.8847484181047474e-07, 1.3177870193660672e-08, 1.2706140743156817e-06, 1.5886518322735786e-05),
'lr_fb': 0.00012520797481978556,
'sigma': 0.07482483561645029,
'beta1_fb': 0.9,
'beta2_fb': 0.999,
'epsilon_fb': 0.00020685070303063725,
'out_dir': 'logs/cifar/DDTPConvControlCIFAR',
'network_type': 'DDTPConvControlCIFAR',
'initialization': 'xavier_normal',
'fb_activation': 'linear',
'gn_damping': [0.001, 0.01, 0.0001],
}

config_DFAConv = {
'beta1': 0.9,
'beta2': 0.999,
'epsilon': [1.1064155093495904e-07, 1.3370147141514222e-08, 1.4849488795615625e-08, 6.579432537287141e-06],
'lr': [4.512786947960491e-05, 0.0003925906667713482, 5.3583994422517426e-05, 7.349907938676488e-05],
'out_dir': 'logs/cifar/DFAConvCIFAR',
'network_type': 'DFAConvCIFAR',
'initialization': 'xavier_normal',
'fb_activation': 'linear',
'target_stepsize': 1.0,
'gn_damping': [10., 10., 10.],
}

config_collection = {
'DDTP-linear': config_DDTPConv,
'DDTP-control': config_DDTPControlConv,
'DFA': config_DFAConv,
}

result_keys = [
'loss_train',
'loss_test',
'acc_train',
'acc_test',
'bp_angles',
'gnt_angles',
'rec_loss'
]


config_fixed = {
'dataset': 'cifar10',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'epochs_fb': 10,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
'epochs': 100,
'double_precision': True,
'output_activation': 'softmax',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'save_logs': True,
'save_BP_angle': True,
'save_GN_angle': False,
'save_GNT_angle': True,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'plots': 'compute',
'log_interval': 80,
}

if __name__ == '__main__':
	 pass