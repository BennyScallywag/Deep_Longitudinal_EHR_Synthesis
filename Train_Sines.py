from checkpoint_torch_timegan import timegan
#from parallel_timegan import parallel_timegan
from WTimeGAN import w_timegan
import numpy as np
from torch_dataloading import sine_data_generation
import matplotlib.pyplot as plt
from Plotting_and_Visualization import plot_original_vs_generated

no = 400            #change this
sine_data = sine_data_generation(no, 24, 5)

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 400
parameters['batch_size'] = 128
#generated_sine_data, e, r, s, d, g = w_timegan(sine_data, parameters, checkpoint_file='TEST3_sines_checkpoint.pth')

generated_sine_data, e, r, s, d, g = w_timegan(sine_data, parameters, checkpoint_file=f'w_e{parameters["iterations"]}_no{no}.pth')
plot_original_vs_generated(sine_data, generated_sine_data, filename=f'w_e{parameters["iterations"]}_no{no}.pdf', num_samples=15)
#plt.savefig("updatedwass_test1")        #put this inside of the plotting function - otherwise will make blank screen?

