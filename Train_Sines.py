from checkpoint_torch_timegan import timegan
from parallel_timegan import parallel_timegan
from WTimeGAN import w_timegan
import numpy as np
from torch_dataloading import sine_data_generation
import matplotlib.pyplot as plt
from Plotting_and_Visualization import plot_original_vs_generated

sine_data = sine_data_generation(100, 24, 5)

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 700
parameters['batch_size'] = 128
#generated_sine_data, e, r, s, d, g = w_timegan(sine_data, parameters, checkpoint_file='TEST3_sines_checkpoint.pth')
generated_sine_data, e, r, s, d, g = parallel_timegan(sine_data, parameters, checkpoint_file='pTEST1_sines_checkpoint.pth')

plot_original_vs_generated(sine_data, generated_sine_data, num_samples=15)

