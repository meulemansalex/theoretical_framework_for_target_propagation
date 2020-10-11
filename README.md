# A Theoretical Framework for Target Propagation
This repository is the official implementation of  ['*A Theoretical Framework for
Target Propagation*'](https://arxiv.org/abs/2006.14331).

## Install Python packages
All the needed Python libraries can be installed with conda by running:
```
$ conda env create -f environment.yml
```

## Running the methods
You can run the various methods on any feedforward fully connected neural
network architecture and the covered CNN architectures by calling the
`main.py` script and specifying the needed
command-line arguments. E.g. for running the DTP method on MNIST for 100 epochs
with a network of 5 hidden layers of size 256, run the following command:
```
$ python3 main.py --dataset=mnist --num_hidden=5 --size_hidden=256 --epochs=100
```
Run `python3 main.py --help` for a documentation on all command-line arguments.

We provide the hyperparameter configurations and command-line arguments that
were used to generate the results in the NeurIPS submission in the directory
`final_configs`. To directly run such a config file, without needing to
copy all the command-line arguments, run the `run_config.py` script with the
config file (in this case DTP on MNIST):
```
$ python3 run_config.py --config_module=final_configs.mnist_DTP
```

## Generating the results
The results in Table 1, 2, S6, S8, S9 and S10 of the paper were computed
over 10 random weight initializations. To automatically run a config file in
10 random configurations and save the results in a `.csv` file, run the
`seed_robustness.py` script (in this case DTP on MNIST for 10 random seeds):
```
$ python3 seed_robustness.py --config_module=final_configs.mnist_DTP --name=mnist_DTP.csv
```
The config files corresponding to Table 2 and S10 have an extension `_train` in
their filename.

## Creating the alignment figures
We created figure scripts that run all the methods on a certain dataset,
compute the alignment angles with BP and GNT and save the plots
(see `figure_scripts`). E.g. for
generating the alignment figures on Fashion-MNIST, run the `create_figures.py`
script:
```
$ python3 create_figures.py --out_dir=logs/figure_fashion_mnist --config_module=figure_scripts.final_figure_fashion_mnist
```
Figures 4, S4, S5, S6 and S7 of the NeurIPS submission can be created by the
above command with their corresponding config file in `figure_scripts`.

## Creating the toy-experiment figures
In the NeurIPS submission, we did two toy experiments (Fig. 2 and S3). The
generation of the barplot in the nullspace toy experiment of Fig. 2 goes in two
steps. First, we need to run the DTP and DDTP-linear method on the synthetic
dataset and save the nullspace component norm ratios:
```
$ python3 create_figures.py --out_dir=logs/toy_experiment_nullspace --config_module=figure_scripts.figure_toy_nullspace
```
Second, the python script `plot_barchart.py` reads the saved results and makes
a bar chart out of it:
```
$ python3 plot_barchart.py --result_dir=logs/toy_experiment_nullspace
```
For the output space toy experiment (Fig. S2), we need first to generate the
output space updates for both GNT and BP:
```
$ python3 run_config.py --config_module=final_configs.toy_figure_output_space_config_GNT
```
```
$ python3 run_config.py --config_module=final_configs.toy_figure_output_space_config_BP
```
Second, the python script `plot_toy_figure_output_space.py` reads the results
and generates the plot:
```
$ python3 plot_toy_figure_output_space.py --logdir=logs/output_space_figure
```

## Method names
Due to historical reasons, the naming of the methods in our code base does not
correspond exactly with the naming in the paper. In the table
below, we link the two name frameworks.

| NeurIPS Submission  | Code Base |
| ------------- | ------------- |
| DTPDRL  | DTPDR  |
| DDTP-linear  | DMLPDTP2 (with fb_activation=linear and size_mlp_fb=None)  |
| DDTP-RHL  | DKDTP2  |
| DTP  | DTP  |
| DTP-control  | DTPControl  |
| DFA  | DFA  |
| BP  | BP |
| GNT  | GN2  |
| DDTP-linear with CNN  | DDTPConvCIFAR|
| DFA with CNN  | DFAConvCIFAR|
| DDTP-control with CNN  | DDTPConvControlCIFAR|
| BP with CNN  | BPConvCIFAR|

