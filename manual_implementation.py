import numpy as np
#from timegan_old import Timegan
#from TT_old import train, test, dp_train
from Train_and_Test import train, test, dp_train
from torch_dataloading import real_data_loading, sine_data_generation
import os
import pandas as pd

class Options:
    def __init__(self, epochs, sine_no, hidden_dim, num_layer, lr, seq_len, data_name, gamma, batch_size, filename_prefix, 
                 module='gru', synth_size=100, metric_iteration=2, noise_sd = 0.2, use_dp=False, eps=15, delta=1e-5,
                 sample_to_excel=False, num_samples_plotted = 15):
        self.iterations = epochs
        self.sine_no = sine_no
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lr = lr
        self.seq_len = seq_len
        self.data_name = data_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.filename_prefix  = filename_prefix
        self.module = module
        self.synth_size = synth_size
        self.metric_iteration = metric_iteration
        self.noise_sd = noise_sd
        self.use_dp = use_dp
        self.eps = eps
        self.delta = delta
        self.sample_to_excel = sample_to_excel
        self.num_samples_plotted = num_samples_plotted

# ----------------CHANGE THESE (INPUTS)------------------
opt = Options(
    epochs=39,              #number of epochs
    sine_no=1000,            #number of training time-series's if using the 'sines' data
    hidden_dim=24,           #dimension of the latent space
    num_layer=3,             #number of hidden layers in each network
    lr=0.001,                #learning rate (shouldnt make much of a difference)
    seq_len = 24,            #sequence length
    data_name = 'ckd',    #which dataset to use, options are 'sines', 'stocks', and 'ckd'
    noise_sd=0.2,            #standard deviation of noise added to the disriminator inputs during training
    use_dp = False,           #whether to use differentially private training
    eps = 15,                 #epsilon value for differential privacy
    delta = 1e-5,             #delta value for differential privacy
    gamma = 1,                #relative weight of generator loss to discriminator loss during training
    num_samples_plotted=15,   #number of samples to plot in the 4-pane plot
    batch_size = 128,
    filename_prefix = ''      #IMPORTANT: prefix to add to checkpoint filename, this ensures that you dont overwrite an existing checkpoint if using the same parameters
)
#---------------------------------------------------------

#no need the change this unless code does not work:
if __name__ == "__main__":
    checkpoint_filename = f'test{opt.filename_prefix}_e{opt.iterations}_seq{opt.seq_len}_{opt.data_name}'
    # Load your data
    if opt.data_name in ['stocks', 'energy']:
        ori_data = real_data_loading(opt.data_name, opt.seq_len)
    elif opt.data_name == 'sines':
        # Set number of samples and its dimensions
        ori_data = sine_data_generation(opt.sine_no, opt.seq_len, 5)
    elif opt.data_name == 'ckd':
        # Get the directory of the current script, make relative path to CSV data
        script_dir = os.path.dirname(__file__)
        data_path = os.path.join(script_dir, 'data', 'timeseries2_CKD_dataset.csv')

        #Loading and filtering the Japanese CKD dataset
        df_ckd = pd.read_csv(data_path, delimiter=",")
        columns_used = ['eGFR', 'age', 'BMI', 'Hb', 'Alb', 'Cr', 'UPCR']
        df_ckd = df_ckd[columns_used].dropna(axis=0)

        #Convert to Numpy array and reshape into time series sub-arrays
        ckd_array = df_ckd.values
        ckd_sequences_array = ckd_array.reshape(-1,7,len(columns_used))

        #Identify & remove sub-arrays where any value in the final position (eGFR) is zero
        #(currently only considering full sequences)
        mask = np.all(ckd_sequences_array[:, :, -1] != 0, axis=1)            #middle index is sequence number?
        ori_data = ckd_sequences_array[mask]
    print(f'{opt.data_name} dataset is ready.')
    
    if opt.use_dp:
        privacy_params = dp_train(ori_data, opt, checkpoint_filename, delta=1e-5)
        test(ori_data, opt, checkpoint_filename, privacy_params)
    else:
        train(ori_data, opt, checkpoint_filename)
        test(ori_data, opt, checkpoint_filename)