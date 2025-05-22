# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: April 10, 2025

import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)

def cla():
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    # Data parameters
    parser.add_argument('--x_train_file', type=str, default='', help=textwrap.dedent('''Data file containing training samples x, i.e., the inferred field '''))
    parser.add_argument('--y_train_file', type=str, default='', help=textwrap.dedent('''Data file containing training samples y, i.e., the measured field '''))
    parser.add_argument('--n_train'   , type=int, default=4000, help=textwrap.dedent('''Number of training samples to use. Cannot be more than that available.'''))
    
    
    # Network parameters
    parser.add_argument('--g_type'    , type=str, default='resskip', choices=['resskip','denseskip'], help=textwrap.dedent('''Type of generator to use'''))
    parser.add_argument('--d_type'    , type=str, default='res', choices=['res','dense'], help=textwrap.dedent('''Type of critic to use'''))
    parser.add_argument('--g_out'     , type=str, default='x', choices=['x','dx'], help=textwrap.dedent('''x: U-Net generates the target field, 
                                                                                                          dx: U-Net generates a perturbation of the target field'''))
    parser.add_argument('--gp_coef'   , type=float, default=10.0, help=textwrap.dedent('''Gradient penalty parameter'''))
    parser.add_argument('--n_critic'  , type=int, default=4, help=textwrap.dedent('''Number of critic updates per generator update'''))
    parser.add_argument('--n_epoch'   , type=int, default=1000, help=textwrap.dedent('''Maximum number of epochs'''))
    parser.add_argument('--z_dim'     , type=int, default=10, help=textwrap.dedent('''Dimension of the latent variable.'''))
    parser.add_argument('--batch_size', type=int, default=16, help=textwrap.dedent('''Batch size while training'''))
    parser.add_argument('--reg_param' , type=float, default=1e-7, help=textwrap.dedent('''Regularization parameter'''))
    parser.add_argument('--seed_no'   , type=int, default=1008, help=textwrap.dedent('''Set the random seed'''))
    parser.add_argument('--mismatch_param', type=float, default=0.0, help=textwrap.dedent('''Parameter to penalize deviation of ensemble samples from ground-truth'''))

    # Output parameters
    parser.add_argument('--save_freq'    , type=int, default=100, help=textwrap.dedent('''Number of epochs after which a network checkpoint,snapshot and plots are saved'''))
    parser.add_argument('--sdir_suffix'  , type=str, default='', help=textwrap.dedent('''Suffix to directory where trained network/results are saved'''))
    parser.add_argument('--z_n_MC'       , type=int, default=10, help=textwrap.dedent('''Number of z samples used to generate emperical statistics.'''))
    
    # Testing parameters
    parser.add_argument('--GANdir'       , type=str, default=None, help=textwrap.dedent('''Load checkpoint from user specified GAN directory. Else path will be infered from hyperparameters.'''))
    parser.add_argument('--x_test_file', type=str, default='', help=textwrap.dedent('''Data file containing test samples x, i.e., the inferred field '''))
    parser.add_argument('--y_test_file', type=str, default='', help=textwrap.dedent('''Data file containing test samples y, i.e., the measured field '''))
    parser.add_argument('--n_test'       , type=int, default=None, help=textwrap.dedent('''Number of test samples to use. Cannot be more than that available.'''))
    parser.add_argument('--sigma_y'      , type=float, default=None, help=textwrap.dedent('''SD of Gaussian noise to add to y samples before feeding to the generator (in test mode) .'''))
    parser.add_argument('--results_dir'  , type=str, default='Test_results', help=textwrap.dedent('''Directory where test results are saved'''))
    parser.add_argument('--ckpt_id'      , type=int, default=-1, help=textwrap.dedent('''The checkpoint index to load when testing'''))

    assert parser.parse_args().z_dim > 0

    return parser.parse_args()


