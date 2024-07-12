from checkpoint_torch_timegan import timegan
import numpy as np
from torch_dataloading import sine_data_generation

sine_data = sine_data_generation(100, 24, 5)

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 500
parameters['batch_size'] = 128
generated_ckd_data_nc, e, r, s, d, g = timegan(sine_data, parameters, checkpoint_file='TEST_sines_checkpoint.pth')