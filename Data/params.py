import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
def argparser():
    parser = argparse.ArgumentParser(description='Parameters for generating phantom data', formatter_class=formatter)

    # == parameters == 
    parser.add_argument('--shape_type', type=str, default='ellipse', help=textwrap.dedent(''' type of shape (default: %(default)s)'''), choices=['rectangle', 'ellipse'])
    parser.add_argument('--N_train_base', type=int, default=100, help='number of training base images (without noise) to generated (default: %(default)s)')
    parser.add_argument('--N_test_base', type=int, default=100, help='number of test base images (without noise) to generated (default: %(default)s)')
    parser.add_argument('--N_px', type=int, default=32, help='number of pixels (the same in all directions) (default: %(default)s)')
    parser.add_argument('--N_noisy_per_img', type=int, default=1, help='number of noisy images per single base image (default: %(default)s)')
    parser.add_argument('--data_dir', type=str, default='datasets',help='name of directory to store data (default: %(default)s)')  
    parser.add_argument('--base_density', type=float, default='0.2',help='salt and pepper noise base density (default: %(default)s)')  
    params = parser.parse_args()

    return params
