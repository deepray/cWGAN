# Conditional Wasserstein GAN 
###Created by: Deep Ray, University of Maryland
###Email: deepray@umd.edu
###Webpage: deepray.github.io 
###Date : 21 May, 2025

This repository contains the code to train and test conditional Wasserstein GANs (cWGANs) with a full-gradient penalty. This code is based on the framework described in the following paper:

*Solution of physics-based inverse problems using conditional generative adversarial networks with full gradient penalty* (D. Ray, J. Murgoitio-Esandi, A. Dasgupta, A. A. Oberai); Computer Methods in Applied Mechanics and Engineering, Vol. 417, 2023 [[Article]](https://www.sciencedirect.com/science/article/pii/S0045782523004620?dgcid=rss_sd_all) [[Preprint]](https://arxiv.org/abs/2306.04895)

### Required packages
The code is written using PyTorch and needs the following packages to run the scripts:

* matplotlib
* ProgressBar2
* torchvision
* pytorch (either CPU or GPU version depending on what resources are available)
* torchinfo
* scipy

### Understanding the command line options for the various scripts

The following command line options are available

| Argument                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `-h`, `--help`                    | Show this help message and exit                                            |
| `--x_train_file X_TRAIN_FILE`     | Data file containing training samples x (the inferred field)               |
| `--y_train_file Y_TRAIN_FILE`     | Data file containing training samples y (the measured field)               |
| `--n_train N_TRAIN`               | Number of training samples to use. Cannot exceed available samples         |
| `--g_type {resskip,denseskip}`    | Type of generator to use                                                   |
| `--d_type {res,dense}`            | Type of critic to use                                                      |
| `--g_out {x,dx}`                  | `x`: U-Net generates the target field; `dx`: U-Net generates perturbation       |
| `--gp_coef GP_COEF`               | Gradient penalty parameter                                                 |
| `--n_critic N_CRITIC`             | Number of critic updates per generator update                              |
| `--n_epoch N_EPOCH`               | Maximum number of epochs                                                   |
| `--z_dim Z_DIM`                   | Dimension of the latent variable                                           |
| `--batch_size BATCH_SIZE`         | Batch size while training                                                  |
| `--reg_param REG_PARAM`           | Regularization parameter                                                   |
| `--seed_no SEED_NO`               | Set the random seed                                                        |
| `--mismatch_param MISMATCH_PARAM` | Penalize deviation of ensemble samples from ground-truth                  |
| `--save_freq SAVE_FREQ`           | Epoch interval to save checkpoint, snapshot, and plots                     |
| `--sdir_suffix SDIR_SUFFIX`       | Suffix for the directory to save trained network/results                   |
| `--z_n_MC Z_N_MC`                 | Number of z samples to use (per y) to compute empirical statistics                               |
| `--GANdir GANDIR`                 | Load checkpoint from specified GAN directory or infer from hyperparameters |
| `--x_test_file X_TEST_FILE`       | Data file containing test samples x (the inferred field)                   |
| `--y_test_file Y_TEST_FILE`       | Data file containing test samples y (the measured field)                   |
| `--n_test N_TEST`                 | Number of test samples to use. Cannot exceed available samples             |
| `--sigma_y SIGMA_Y`               | SD of Gaussian noise to add to test y samples before generator input       |
| `--results_dir RESULTS_DIR`       | Directory to save test results                                             |
| `--ckpt_id CKPT_ID`               | Checkpoint index to load when testing                                      |


###Generating noisy data for training and test
The code comes packaged with a way to generate 32x32 images of randomized ellipses or rectangles and adds random salt and pepper noise to it. To create this data, go to ```Data``` and run the following command:

~~~bash
python3 generate_data.py --N_train_base 4000 --N_test_base 100 
~~~

This will create 4000 training labelled samples (stored in ```datasets_ellipse\x_train.npy``` and ```datasets_ellipse\y_train.npy```) and 100 test samples (stored in ```datasets_ellipse\x_train.npy``` and ```datasets_ellipse\y_train.npy```). Each x and y is an image of shape 1x32x32 where an channel dimension has been artificially introduced to be compatible with the data shapes use in PyTorch 2D convolution layers.

###Running the training script
Assuming the above dataset files have been create, you can run the cWGAN training script in the following manner (feel free to change the hyperparameters as needed). Note that the architecture of the generator and critic (which makes use on 2D convolution layers) has been created for an input/output shape of size 1x32x32. If working with different resolutions, you should adapt the architectures described in the file ```Scripts/models.py```

* When the generator directly output samples of x given an input y (and the latent variable)

~~~bash
python3 trainer.py --n_train 4000 --x_train_file ../Data/datasets_ellipse/x_train.npy --y_train_file ../Data/datasets_ellipse/y_train.npy --n_epoch 50 --save_freq 10 --batch_size 25 --g_out x --g_type denseskip --d_type dense
~~~

* When the generator outputs perturbation dx given an input y (and the latent variable) and add y back to it. Thus the precition is y + dx

~~~bash
python3 trainer.py --n_train 4000 --x_train_file ../Data/datasets_ellipse/x_train.npy --y_train_file ../Data/datasets_ellipse/y_train.npy --n_epoch 50 --save_freq 10 --batch_size 25 --g_out dx --g_type denseskip --d_type dense
~~~

In each case, the GAN files, plots, and generator checkpoints will be saved in the directory ```exps```

###Loading and testing a trained genetor
Here we demonstrate how to load a trained generator checkpoint and test it on new data. We will make use of the pre-trained networks available in this repository. 

* When the generator directly output samples of x given an input y (and the latent variable)

~~~bash
python3 test_gan.py --GANdir ../Pretrained_Nets/Learning_x --n_test 10 --x_test_file ../Data/datasets_ellipse/x_test.npy --y_test_file ../Data/datasets_ellipse/y_test.npy --n_epoch 50  --batch_size 25 --g_out x --g_type denseskip --d_type dense --ckpt_id 50
~~~

* When the generator outputs perturbation dx given an input y (and the latent variable) and add y back to it. Thus the precition is y + dx

~~~bash
python3 test_gan.py --GANdir ../Pretrained_Nets/Learning_dx --n_test 10 --x_test_file ../Data/datasets_ellipse/x_test.npy --y_test_file ../Data/datasets_ellipse/y_test.npy --n_epoch 50  --batch_size 25 --g_out dx --g_type denseskip --d_type dense --ckpt_id 50
~~~

In each case, the test plots will be created in the specific GAN directories.

