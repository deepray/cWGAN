# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: April 10, 2025


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from torchinfo import summary
from functools import partial
import time,random
from config import cla
from utils import *
from data_utils import *
from models import *



if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

# Loading parameters
PARAMS = cla()

random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
torch.manual_seed(PARAMS.seed_no)

print('\n ============== LAUNCHING TRAINING SCRIPT =================\n')


print('\n --- Creating network folder \n')
savedir  = make_save_dir(PARAMS)


print('\n --- Loading training data from file\n')
train_data   = LoadDataset(datafiles = [PARAMS.x_train_file,PARAMS.y_train_file],
                           N         = PARAMS.n_train)  

# X represents the inferred field while Y is the measured field
_,x_C,x_W,x_H = train_data.x_shape
_,y_C,y_W,y_H = train_data.y_shape

# Creating sub-function to generate latent variables
glv    = partial(get_lat_var,z_dim=PARAMS.z_dim)
    
print('\n --- Creating conditional GAN models\n')

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

if PARAMS.d_type == 'res':    
    # Creates critic with residual blocks
    D_model = critic_res(x_shape=(x_C,x_H,x_W))
elif PARAMS.d_type == 'dense':    
    # Creates critic with dense blocks
    D_model = critic_dense (x_shape=(x_C,x_H,x_W),
                            denselayers=3)

summary(G_model,input_size=[(2,x_C,x_H,x_W),(2,PARAMS.z_dim,1,1)])  
summary(D_model,input_size=[(2,x_C,x_H,x_W),(2,x_C,x_H,x_W)])     

z_n_MC = PARAMS.z_n_MC
  
# Moving models to correct device and adding optimizers
G_model.to(device)
D_model.to(device)
G_optim = torch.optim.Adam(G_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=PARAMS.reg_param)
D_optim = torch.optim.Adam(D_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=PARAMS.reg_param)

# Creating data loader and any other dummmy/display variables
loader = DataLoader(train_data, batch_size=PARAMS.batch_size, shuffle=True)

# Creating objects for plotting every few epochs
n_view = 5 # Number of training samples to save plots for every few epochs
stat_z = glv(batch_size=z_n_MC*n_view)
view_x,view_y = train_data[0:n_view]
stat_y = torch.repeat_interleave(view_y,repeats=z_n_MC,dim=0)
stat_y = stat_y.to(device)
stat_z = stat_z.to(device)


# ============ Training ==================
print('\n --- Initiating GAN training\n')

n_iters = 1
G_loss_log    = []
D_loss_log    = []
mismatch_log  = []
wd_loss_log   = []


for i in range(PARAMS.n_epoch):
    for true_X,true_Y in loader:

        true_X = true_X.to(device)
        true_Y = true_Y.to(device)

        # ---------------- Updating critic -----------------------
        D_optim.zero_grad()
        z = glv(batch_size=PARAMS.batch_size)
        z = z.to(device)
        fake_X     = G_model(true_Y,z).detach()
        fake_val   = D_model(fake_X,true_Y)
        true_val   = D_model(true_X,true_Y)
        gp_val     = full_gradient_penalty(fake_X=fake_X, 
                                           true_X=true_X, 
                                           true_Y=true_Y,
                                           model =D_model, 
                                           device=device)
        fake_loss  = torch.mean(fake_val)
        true_loss  = torch.mean(true_val)
        wd_loss    = true_loss - fake_loss
        D_loss     = -wd_loss + PARAMS.gp_coef*gp_val

        D_loss.backward()
        D_optim.step()
        D_loss_log.append(D_loss.item())
        wd_loss_log.append(wd_loss.item())
        print(f"     *** (epoch,iter):({i},{n_iters}) ---> d_loss:{D_loss.item():.4e}, gp_term:{gp_val.item():.4e}, wd:{wd_loss.item():.4e}")

        # ---------------- Updating generator -----------------------
        if n_iters%PARAMS.n_critic == 0:
            G_optim.zero_grad()
            z = glv(batch_size=PARAMS.batch_size)
            z = z.to(device)
            fake_X     = G_model(true_Y,z)
            fake_val   = D_model(fake_X,true_Y)
            G_loss     = -torch.mean(fake_val)
            mismatch   = torch.mean(torch.square(true_X-fake_X))/(x_C*x_W*x_H)
            if PARAMS.mismatch_param > 0.0:
                G_loss += PARAMS.mismatch_param*mismatch
            G_loss.backward()
            G_optim.step()
            G_loss_log.append(G_loss.item())
            mismatch_log.append(mismatch.item())
            print(f"     ***           ---> g_loss:{G_loss.item():.4e}, mismatch:{mismatch.item():.4e}") 

        
        n_iters += 1    

    # Saving intermediate output and generator checkpoint
    if (i+1) % PARAMS.save_freq == 0: 
        with torch.no_grad():
            print(f"     *** generating sample plots and saving generator checkpoint")
            G_model.eval()  # NOTE: switching to eval mode for generator  
            pred_ = G_model(stat_y,stat_z).cpu().detach().numpy().squeeze()
            G_model.train() # NOTE: switching back to train mode 

        fig1,axs1 = plt.subplots(n_view, 4, dpi=100, figsize=(4*5,n_view*5))
        ax1 = axs1.flatten()
        ax_ind = 0
        for t in range(n_view):
            axs = ax1[ax_ind]
            imgy = view_y[t:t+1].numpy().squeeze()
            pcm  = axs.imshow(imgy,
                              aspect='equal',
                              cmap='copper',
                              vmin=np.min(imgy),
                              vmax=np.max(imgy))
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'measurement',fontsize=30)
            axs.axis('off')
            ax_ind +=1

            axs = ax1[ax_ind]
            imgx = view_x[t:t+1].numpy().squeeze()
            pcm = axs.imshow(imgx,
                             aspect='equal',
                             cmap='copper',
                             vmin=np.min(imgx),
                             vmax=np.max(imgx))
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'target',fontsize=30)
            axs.axis('off')
            ax_ind +=1

            sample_mean = np.mean(pred_[t*z_n_MC:(t+1)*z_n_MC],axis=0)
            
            axs = ax1[ax_ind]
            pcm = axs.imshow(sample_mean,
                             aspect='equal',
                             cmap='copper',
                             vmin=np.min(sample_mean),
                             vmax=np.max(sample_mean))
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'Mean',fontsize=30)
            axs.axis('off')
            ax_ind +=1

            

            sample_std  = np.std(pred_[t*z_n_MC:(t+1)*z_n_MC],axis=0)
            axs = ax1[ax_ind]
            pcm = axs.imshow(sample_std,
                             aspect='equal',
                             cmap='copper',
                             vmin=np.min(sample_std),
                             vmax=np.max(sample_std))
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'SD',fontsize=30)
            axs.axis('off')
            ax_ind +=1

        fig1.tight_layout()
        fig1.savefig(f"{savedir}/stats_sample_{i+1}.png")
        plt.close('all')

        torch.save(G_model.state_dict(),f"{savedir}/checkpoints/G_model_{i+1}.pth")

save_loss(G_loss_log,'g_loss',savedir,PARAMS.n_epoch)
save_loss(D_loss_log,'d_loss',savedir,PARAMS.n_epoch)  
save_loss(wd_loss_log,'wd_loss',savedir,PARAMS.n_epoch) 



print('\n ============== DONE =================\n')



