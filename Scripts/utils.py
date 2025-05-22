# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: April 10, 2025


import os,shutil
import numpy as np
import matplotlib.pyplot as plt

def make_save_dir(PARAMS):
    ''' This function creates the results save directory'''

    savedir = get_GAN_dir(PARAMS)

    if os.path.exists(savedir):
        print('\n     *** Folder already exists!\n')    
    else:
        os.makedirs(savedir)

    # Creating directory to save generator checkpoints
    if os.path.exists(f"{savedir}/checkpoints"):
        print('\n     *** Checkpoints directory already exists\n')    
    else:
        os.makedirs(f"{savedir}/checkpoints")    

    print('\n --- Saving parameters to file \n')
    param_file = savedir + '/parameters.txt'
    with open(param_file,"w") as fid:
        for pname in vars(PARAMS):
            fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")    

    return savedir 
 
def get_GAN_dir(PARAMS): 
  
    savedir = f"../exps/g_{PARAMS.g_type}"\
              f"_{PARAMS.g_out}"\
              f"_d_{PARAMS.d_type}"\
              f"_Nsamples{PARAMS.n_train}"\
              f"_Ncritic{PARAMS.n_critic}_Zdim{PARAMS.z_dim}"\
              f"_BS{PARAMS.batch_size}_Nepoch{PARAMS.n_epoch}"\
              f"_GP{PARAMS.gp_coef}{PARAMS.sdir_suffix}"      
    return savedir    

def save_loss(loss,loss_name,savedir,n_epoch):
    
    np.savetxt(f"{savedir}/{loss_name}.txt",loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1,len(loss)])


    ax2 = ax1.twiny()
    ax2.set_xlim([0,n_epoch])
    ax2.set_xlabel('Epochs')


    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png",dpi=200)    
    plt.close()           
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    


    
