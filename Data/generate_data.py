# This script is used to generate noisy shapes
# Created by: Deep Ray, University of Maryland
# Date: April 10, 2025

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import shapes
import utils
import params

PARAMS = params.argparser()

data_dir = f"{PARAMS.data_dir}_{PARAMS.shape_type}"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)  

param_file = data_dir + '/parameters.txt'
with open(param_file,"w") as fid:
    for pname in vars(PARAMS):
        fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")

# Creating sampling points in 2D space
domain_width = 10
N_px = PARAMS.N_px
x = np.linspace(-domain_width/2,domain_width/2,PARAMS.N_px)
y = np.linspace(-domain_width/2,domain_width/2,PARAMS.N_px)
xcoord = np.ones((PARAMS.N_px,1))*x
ycoord = y.reshape(-1,1)*np.ones((1,PARAMS.N_px))

# Creating training samples
print("Creating training samples")
N_train = PARAMS.N_train_base*PARAMS.N_noisy_per_img
bar1 = utils.make_prog_bar(N_train)        
x_train = np.zeros((N_train,1,N_px,N_px)) # Artifically adding channel dimension
y_train = np.zeros((N_train,1,N_px,N_px)) # Artifically adding channel dimension
idx = 0
for n in range(PARAMS.N_train_base):
	shape_obj  = shapes.sup_shape(shape_type = PARAMS.shape_type)
	image      = shape_obj.discrete_shape(xcoord=xcoord,ycoord=ycoord)

	for j in range(PARAMS.N_noisy_per_img):
		noisy_image = utils.add_noise(image=image,
									  density0=PARAMS.base_density,
									  pert=True)
		x_train[idx,:,:] = np.copy(image)
		y_train[idx,:,:] = np.copy(noisy_image)
		bar1.update(idx)
		idx+=1
print('\n')
print(f'Generate {N_train} training pairs')

# Saving data
np.save(data_dir+'/x_train.npy',x_train)
np.save(data_dir+'/y_train.npy',y_train)

# Creating test samples
print("Creating test samples")
N_test = PARAMS.N_test_base*PARAMS.N_noisy_per_img
bar2 = utils.make_prog_bar(N_test)     
x_test = np.zeros((N_test,1,N_px,N_px)) # Artifically adding channel dimension
y_test = np.zeros((N_test,1,N_px,N_px)) # Artifically adding channel dimension
idx = 0
for n in range(PARAMS.N_test_base):
	shape_obj  = shapes.sup_shape(shape_type = PARAMS.shape_type)
	image      = shape_obj.discrete_shape(xcoord=xcoord,ycoord=ycoord)

	for j in range(PARAMS.N_noisy_per_img):
		noisy_image = utils.add_noise(image=image,
									  density0=PARAMS.base_density,
									  pert=True)
		x_test[idx,:,:] = np.copy(image)
		y_test[idx,:,:] = np.copy(noisy_image)
		bar2.update(idx)
		idx+=1
print('\n')
print(f'Generate {N_test} test pairs')

# Saving data
np.save(data_dir+'/x_test.npy',x_test)
np.save(data_dir+'/y_test.npy',y_test)

# Plotting a few random training samples
random_index = np.random.choice(np.arange(N_train),16)
fig1, ax1 = plt.subplots(nrows=4,ncols=4,num=1, figsize=(10,10))
fig2, ax2 = plt.subplots(nrows=4,ncols=4,num=2, figsize=(10,10))
ax1 = ax1.flatten()
ax2 = ax2.flatten()
for i in range(16):
	idx = random_index[i]
	
	pcm = ax1[i].imshow(x_train[idx,0,:,:],
		                aspect='equal',
		                vmin=0,
		                vmax=1)
	fig1.colorbar(pcm,ax=ax1[i])
	ax1[i].axis('off')

	pcm = ax2[i].imshow(y_train[idx,0,:,:],
		                aspect='equal',
		                vmin=np.min(y_train[idx,0,:,:]),
		                vmax=np.max(y_train[idx,0,:,:]))
	fig2.colorbar(pcm,ax=ax2[i])
	ax2[i].axis('off')

fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(data_dir+'/x_samples.pdf',dpi=300,bbox_inches='tight')
fig2.savefig(data_dir+'/y_samples.pdf',dpi=300,bbox_inches='tight')
plt.close('all')
