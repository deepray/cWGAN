# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: April 10, 2025


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import partial
from scipy.linalg import qr
import time,random
from config import cla
from models import *
from data_utils import *
from utils import *

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

PARAMS = cla()

random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
torch.manual_seed(PARAMS.seed_no)

print('\n ============== TESTING GAN GENERATOR =================\n')

print('\n --- Loading data from file\n')

test_data   = LoadDataset(datafiles = [PARAMS.x_test_file,PARAMS.y_test_file],
                           N         = PARAMS.n_test) 
n_test,x_C,x_W,x_H = test_data.x_shape

print('\n --- Loading conditional GAN generator from checkpoint\n')
if PARAMS.GANdir == None:
    GANdir = get_GAN_dir(PARAMS)
else:
    GANdir = PARAMS.GANdir   

savedir = GANdir + "/" + PARAMS.results_dir     

if PARAMS.g_type == 'resskip':
    # Creates generator U-Net with residual blocks
    G_model = generator_resskip(x_shape=(x_C,x_H,x_W),
                                z_dim=PARAMS.z_dim,
                                g_out=PARAMS.g_out)  
elif PARAMS.g_type == 'denseskip':
    # Creates generator U-Net with dense blocks
    G_model = generator_denseskip(x_shape=(x_C,x_H,x_W),
                                  z_dim=PARAMS.z_dim,
                                  denselayers=3,
                                  g_out=PARAMS.g_out,
                                  k0 = 8)    

z_n_MC   = PARAMS.z_n_MC
G_state  = torch.load(f"{GANdir}/checkpoints/G_model_{PARAMS.ckpt_id}.pth",map_location=torch.device(device))
G_model.load_state_dict(G_state)
G_model.eval()  # NOTE: switching to eval mode for generator

# Creating sub-function to generate latent variables
glv          = partial(get_lat_var,z_dim=PARAMS.z_dim)
nimp_samples = 3 # Number of important samples to extract from rank revealing QR

# Creating results directory
results_dir = f"{GANdir}/{PARAMS.results_dir}"
print(f"\n --- Generating results directory: {results_dir}")
if os.path.exists(results_dir):
        print('\n     *** Folder already exists!\n')    
else:
    os.makedirs(results_dir)

print(f"\n --- Computing statistics\n")

for n in range(n_test):
    
    test_x,test_y = test_data[n:n+1]

    print(f"\n ---   for test sample {n}\n")

    #snapshot_mat = np.zeros((x_H*x_W,z_n_MC)) 
    stat_y       = torch.repeat_interleave(test_y,repeats=z_n_MC,dim=0)
    if PARAMS.sigma_y != None:
        y_noise = torch.randn(stat_y.shape)*PARAMS.sigma_y
        stat_y  = stat_y + y_noise

    stat_z       = glv(batch_size=z_n_MC)
    with torch.no_grad(): 
        pred_  = G_model(stat_y,stat_z).detach().numpy().squeeze()
    
    #snapshot_mat[:,i] = np.reshape(pred_,(x_H*x_W))
    sample_mean = np.mean(pred_,axis=0)
    sample_std  = np.std(pred_,axis=0)
    print(f"     ... performing rrqr")
    # _,_,p = qr(snapshot_mat,pivoting=True, mode='economic')
    # imp_samples = np.reshape(snapshot_mat[:,p[0:nimp_samples]].T,(-1,x_W,x_H))     

    print(f"     ... plotting")
    fig,axs = plt.subplots(1, 4, dpi=100, figsize=(4*5,5))

    imgy = test_y.numpy().squeeze()
    pcm  = axs[0].imshow(imgy,
                         aspect='equal',
                         cmap='copper',
                         vmin=np.min(imgy),
                         vmax=np.max(imgy))
    fig.colorbar(pcm,ax=axs[0])
    axs[0].set_title(f'measurement',fontsize=30)
    axs[0].axis('off')

    imgx = test_x.numpy().squeeze()
    pcm = axs[1].imshow(imgx,
                     aspect='equal',
                     cmap='copper',
                     vmin=np.min(imgx),
                     vmax=np.max(imgx))
    fig.colorbar(pcm,ax=axs[1])
    axs[1].set_title(f'target',fontsize=30)
    axs[1].axis('off')

    pcm = axs[2].imshow(sample_mean,
                     aspect='equal',
                     cmap='copper',
                     vmin=np.min(sample_mean),
                     vmax=np.max(sample_mean))
    fig.colorbar(pcm,ax=axs[2])
    axs[2].set_title(f'Mean',fontsize=30)
    axs[2].axis('off')
    
    pcm = axs[3].imshow(sample_std,
                     aspect='equal',
                     cmap='copper',
                     vmin=np.min(sample_std),
                     vmax=np.max(sample_std))
    fig.colorbar(pcm,ax=axs[3])
    axs[3].set_title(f'SD',fontsize=30)
    axs[3].axis('off')
    
    fig.tight_layout()
    fig.savefig(f"{savedir}/test_sample_{n+1}.png")
    plt.close('all')
        


print("----------------------- DONE --------------------------")



 