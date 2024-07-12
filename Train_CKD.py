from checkpoint_torch_timegan import timegan
import numpy as np
import pandas as pd
import os

# Get the directory of the current script, make relative path to CSV data
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'data', 'timeseries2_CKD_dataset.csv')

#Loading and filtering the Japanese CKD dataset
df_ckd = pd.read_csv(data_path, delimiter=",")
columns_used = ['age', 'BMI', 'Hb', 'Alb', 'Cr', 'UPCR', 'eGFR']
df_ckd = df_ckd[columns_used].dropna(axis=0)

#Convert to Numpy array and reshape into time series sub-arrays
ckd_array = df_ckd.values
ckd_sequences_array = ckd_array.reshape(-1,7,len(columns_used))

#Identify & remove sub-arrays where any value in the final position (eGFR) is zero
#(currently only considering full sequences)
mask = np.all(ckd_sequences_array[:, :, -1] != 0, axis=1)            #middle index is sequence number?
filtered_array = ckd_sequences_array[mask]

#print(ckd_sequences_array[:,:,-1])     #only the eGFRs
#print(ckd_sequences_array[:, 6, :])        #original
print(filtered_array.shape)

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 500
parameters['batch_size'] = 128

#Train TimeGAN and generate data
generated_ckd_data_nc, e, r, s, d, g = timegan(filtered_array, parameters, checkpoint_file='TEST_ckd_checkpoint.pth')
