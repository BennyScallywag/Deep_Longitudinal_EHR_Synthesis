import argparse
import numpy as np

# 1. Training model
from Train_and_Test import train, dp_train, test
# 2. Data loading
from torch_dataloading import real_data_loading, sine_data_generation
import os
import pandas as pd


def main(opt, checkpoint_filename):
    # Data loading
    ori_data = None
    if opt.data_name in ['stocks', 'energy']:
        ori_data = real_data_loading(opt.data_name, opt.seq_len)
        #print(ori_data)
    elif opt.data_name == 'sines':
        # Set number of samples and its dimensions
        ori_data = sine_data_generation(opt.sine_no, opt.seq_len, opt.sine_dim)
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
        mask = np.all(ckd_sequences_array[:, :, 0] != 0, axis=1)            #middle index is sequence number? last index 0 for egfr?
        ori_data = ckd_sequences_array[mask]

    print(opt.data_name + ' dataset is ready.')

    # Training or Testing
    if opt.test_only:
        test(ori_data, opt, checkpoint_filename)
    else:
        if opt.use_dp:
            dp_train(ori_data, opt, checkpoint_filename)
            test(ori_data, opt, checkpoint_filename)
        else:
            train(ori_data, opt, checkpoint_filename)
            test(ori_data, opt, checkpoint_filename)


if __name__ == '__main__':
    """Main function for timeGAN experiments.
    Args:
      - data_name: sine, stock, or energy
      - seq_len: sequence length
      - Network parameters (should be optimized for different datasets)
        - module: gru, lstm
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
      - metric_iteration: number of iterations for metric computation
    Returns:
      - ori_data: original data
      - gen_data: generated synthetic data
      - metric_results: discriminative and predictive scores
    """
    # Args for the main function
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('--data_name', type=str, default='sines', choices=['sines', 'stocks', 'energy','ckd'], )
    parser.add_argument('--seq_len', type=int, default=24, help='sequence length')
    parser.add_argument('--sine_no', type=int, default=500, help='number of sine data samples')
    parser.add_argument('--sine_dim', type=int, default=5, help='dim of  sine data')
    # Network parameters (should be optimized for different datasets)
    parser.add_argument('--module', choices=['gru', 'lstm'], default='gru', type=str)
    parser.add_argument('--hidden_dim', type=int, default=24, help='hidden state dimensions')
    parser.add_argument('--num_layer', type=int, default=3, help='number of layers')
    # Model training and testing parameters
    parser.add_argument('--gamma', type=float, default=1, help='gamma weight for G_loss and D_loss')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--iterations', type=int, default=400, help='Training iterations')    #changed from 50k
    parser.add_argument('--print_times', type=int, default=10, help='Print times when Training')
    parser.add_argument('--batch_size', type=int, default=128, help='the number of samples in mini-batch')
    parser.add_argument('--synth_size', type=int, default=500, help='the number of samples in synthetic data, '
                                                                  '0--len(ori_data)')
    parser.add_argument('--metric_iteration', type=int, default=5, help='iterations of the metric computation')
    parser.add_argument('--noise_sd', type=float, default=0.2, help='Standard deviation of discriminator noise injection')
    # Save and Load
    parser.add_argument('--data_dir', type=str, default="./data", help='path to stock and energy data')   #can remove?
    parser.add_argument('--networks_dir', type=str, default="./trained_networks", help='path to checkpoint')    #can remove?
    parser.add_argument('--output_dir', type=str, default="./output", help='folder to output metrics and images')   #remove and replace with regular plot folder logic
    parser.add_argument('--filename_additions', type=str, default="", help='prefix for checkpoint filename')
    # Model running parameters
    parser.add_argument('--test_only', action="store_true", help='iterations of the metric computation')
    parser.add_argument('--use_dp', action="store_true", help='include this if you want to use differential privacy in training')
    parser.add_argument('--only_visualize_metric', type=bool, default=False, help='only compute visualization metrics')
    parser.add_argument('--sample_to_excel', type=bool, default=False, help='whether to save a sample of the generated data to an excel file')

    # Call main function
    opt = parser.parse_args()
    no = f'_no{opt.sine_no}' if opt.data_name == 'sines' else ''
    dp = 'DP_' if opt.use_dp else ''
    checkpoint_filename = f'{dp}e{opt.iterations}{no}_noise{opt.noise_sd}_{opt.data_name}_{str(opt.filename_additions)}'

    main(opt, checkpoint_filename)
