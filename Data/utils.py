import numpy as np
import progressbar 
import os
import random



def make_prog_bar(N):
    widgets = [' [',progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
                 '] ', 
                   progressbar.Bar('>'),' (', 
                   progressbar.ETA(), ') ', 
                  ] 
    bar = progressbar.ProgressBar(widgets=widgets).start(max_value=N)  

    return bar  

def add_noise(image,density0,pert=True):

    if pert:
        density = density0*(1+ 0.2*(np.random.rand(1)-0.5))
    else:
        density = density0


    noise = np.random.binomial(1,density,image.shape)

    noisy_image = np.minimum(image+noise,1.0)

    return noisy_image    




