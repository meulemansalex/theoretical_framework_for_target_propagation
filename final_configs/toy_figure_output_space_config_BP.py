config = {
    'dataset': 'student_teacher',
    'batch_size': 1,
    'num_val': 1,
    'size_output': 2,
    'num_hidden': 2,
    'size_hidden': 4,
    'size_input': 4,
    'hidden_activation': 'tanh',
    'no_bias': True,
    'output_space_plot': True,
    'out_dir': 'logs/output_space_figure',
    'network_type': 'GN2',  # the output_space_plot_bp flag overrules this one
    'random_seed': 93,
    'output_space_plot_bp': True,
}