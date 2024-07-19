from checkpoint_torch_timegan import timegan
#from parallel_timegan import parallel_timegan
from WTimeGAN import w_timegan
import numpy as np
from torch_dataloading import sine_data_generation
import matplotlib.pyplot as plt
from Plotting_and_Visualization import plot_original_vs_generated
from metrics.torch_discriminative_metric import discriminative_score_metrics
from metrics.torch_predictive_metric import predictive_score_metrics
from metrics.torch_visualization_metric import visualization

no = 150            #change this
sine_data = sine_data_generation(no, 24, 5)

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 500
parameters['batch_size'] = 128
#generated_sine_data, e, r, s, d, g = w_timegan(sine_data, parameters, checkpoint_file='TEST3_sines_checkpoint.pth')

generated_sine_data, e, r, s, d, g = timegan(sine_data, parameters, checkpoint_file=f'w_e{parameters["iterations"]}_no{no}.pth')

#Discrim & Predictive Metrics
print("Start Discrim & Predictive Score Training")
discrim = discriminative_score_metrics(sine_data, generated_sine_data)
pred = predictive_score_metrics(sine_data, generated_sine_data)
print("End Discrim & Predictive Score Training")
print(f"Discriminative Score (Lower Better): {discrim}, Predictive Score (Higher Better): {pred}")

#Plotting a sample of generated and real data
plot_original_vs_generated(sine_data, generated_sine_data, filename=f'3genstep_e{parameters["iterations"]}_no{no}.pdf', num_samples=15)

#Visualivation Metrics
visualization(sine_data, generated_sine_data, 'pca')
visualization(sine_data, generated_sine_data, 'tsne')


