import numpy as np
#from timegan import Timegan
from DP_timegan import DP_Timegan
from timegan import Timegan
import matplotlib.pyplot as plt
from metrics.torch_discriminative_metric import discriminative_score_metrics
from metrics.torch_predictive_metric import predictive_score_metrics
#from metrics.torch_visualization_metric import visualization
import Plotting_and_Visualization as pv
import torch_utils as tu
import pandas as pd
import torch
import os

def train(ori_data, opt, checkpoint_file):
    """
    Train the TimeGAN model using a three-phase training process: 
    embedding network training, supervised loss training, and joint training.

    Args:
        - ori_data (np.ndarray): The original training data.
        - opt (Namespace): The options/parameters for training, including the number of iterations.
        - checkpoint_file (str): The file path for saving the model checkpoints.
    """
    model = Timegan(ori_data, opt, checkpoint_file)

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(model.start_epoch['embedding'], opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_embedder()
        if (i) % 100 == 0:
            print(f'step: {str(i)}/{str(opt.iterations)}, e_loss: {str(np.round(np.sqrt(model.E_loss_T0.item()), 4))}')
        
        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase1_epochnum = {'embedding': i+1, 'supervisor': 0, 'joint': 0}
            model.save_checkpoint(phase1_epochnum, checkpoint_file)

            #epsilon, best_alpha = model.privacy_engine.accountant.get_privacy_spent(delta=1e-5)
            #print(f"(ε = {epsilon:.2f}, δ = {1e-5}) for α = {best_alpha}")
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(model.start_epoch['supervisor'],opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_supervisor()
        if (i) % 100 == 0:
            print(f'step: {str(i)}/{str(opt.iterations)},  g_loss_s: {str(np.round(np.sqrt(model.G_loss_S.item()), 4))}')
        
        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase2_epochnum = {'embedding': opt.iterations, 'supervisor': i+1, 'joint': 0}
            model.save_checkpoint(phase2_epochnum, checkpoint_file)
    print('Finish Supervised-Only Training')

    # 3. Joint Training
    print('Start Joint Training')
    for i in range(model.start_epoch['joint'], opt.iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            model.gen_batch()
            model.batch_forward()
            model.train_generator(join_train=True)
            model.batch_forward()
            model.train_embedder(join_train=True)
        # # Discriminator training
        model.gen_batch()
        model.batch_forward()
        model.train_discriminator()

        # Print multiple checkpoints
        if (i) % 100 == 0:
            print(
                f'step: {i}/{opt.iterations}, '
                f'd_loss: {np.round(model.D_loss.item(), 4)}, '
                f'g_loss_u: {np.round(model.G_loss_U.item(), 4)}, '
                f'g_loss_v: {np.round(model.G_loss_V.item(), 4)}, '
                f'e_loss_t0: {np.round(np.sqrt(model.E_loss_T0.item()), 4)}'
            )

        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase3_epochnum = {'embedding': opt.iterations, 'supervisor': opt.iterations, 'joint': i+1}
            model.save_checkpoint(phase3_epochnum, checkpoint_file)
    print('Finish Joint Training')

def dp_train(ori_data, opt, checkpoint_file, delta=1e-5):
    """
    Train the TimeGAN model using a three-phase training process: 
    embedding network training, supervised loss training, and joint training.

    Args:
        - ori_data (np.ndarray): The original training data.
        - opt (Namespace): The options/parameters for training, including the number of iterations.
        - checkpoint_file (str): The file path for saving the model checkpoints.
    """
    model = DP_Timegan(ori_data, opt, checkpoint_file)

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(model.start_epoch['embedding'], opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_embedder()
        if (i) % 100 == 0:
            print(f'step: {str(i)}/{str(opt.iterations)}, e_loss: {str(np.round(np.sqrt(model.E_loss_T0.item()), 4))}')
            print(f"Memory Allocated: {torch.cuda.memory_allocated(model.device) / 1024**2} MB")
        
        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase1_epochnum = {'embedding': i+1, 'supervisor': 0, 'joint': 0}
            model.save_checkpoint(phase1_epochnum, checkpoint_file)
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(model.start_epoch['supervisor'],opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_supervisor()
        if (i) % 100 == 0:
            print(f'step: {str(i)}/{str(opt.iterations)},  g_loss_s: {str(np.round(np.sqrt(model.G_loss_S.item()), 4))}')
            print(f"Memory Allocated: {torch.cuda.memory_allocated(model.device) / 1024**2} MB")
        
        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase2_epochnum = {'embedding': opt.iterations, 'supervisor': i+1, 'joint': 0}
            model.save_checkpoint(phase2_epochnum, checkpoint_file)
    print('Finish Supervised-Only Training')

    #model.reset_gradients()
    model.reinitialize_discriminator()
    model.initialize_privacy_engine()
    #3. Joint Training
    print('Start Joint Training')
    model.discriminator.train()
    model.generator.train()
    for i in range(model.start_epoch['joint'], opt.iterations):
        for X_batch, _ in model.dataloader:
            model.train_joint(X_batch)

        # Print multiple checkpoints
        if (i) % 5 == 0:
            print(
                f'step: {i}/{opt.iterations}, '
                f'd_loss: {np.round(model.D_loss.item(), 4)}, '
                f'g_loss_u: {np.round(model.G_loss.item(), 4)}, '
                f'g_loss_v: {np.round(model.G_loss_V.item(), 4)}, '
                #f'e_loss_t0: {np.round(np.sqrt(model.E_loss_T0.item()), 4)}'
            )
            print(f"Memory Allocated: {torch.cuda.memory_allocated(model.device) / 1024**2} MB")

        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase3_epochnum = {'embedding': opt.iterations, 'supervisor': opt.iterations, 'joint': i+1}
            model.save_checkpoint(phase3_epochnum, checkpoint_file)

            epsilon = model.privacy_engine.get_epsilon(delta=1e-5)
            print(f"(ε = {epsilon:.2f}, δ = {1e-5})")
    print('Finish Joint Training')
    return {'epsilon': opt.eps, 'delta': delta}

def test(ori_data, opt, filename, privacy_params=None):
    """
    Test the TimeGAN model by generating synthetic data and evaluating its performance using discriminative 
    and predictive scores, followed by visualization using PCA and t-SNE.

    Args:
        ori_data (np.ndarray): The original data used for generating synthetic data and evaluation.
        opt (Namespace): The options/parameters for testing, including synthetic data size and metric iterations.
        filename (str): The file path for saving the visualization output.
    """

    print('Start Testing')
    if opt.use_dp:
        model = DP_Timegan(ori_data, opt, filename)
    else:
        model = Timegan(ori_data, opt, filename)

    # Synthetic data generation
    #synth_size = opt.synth_size if opt.synth_size != 0 else len(ori_data)
    synth_size = min(opt.synth_size, len(ori_data))
    generated_data = model.gen_synth_data(synth_size)
    generated_data = generated_data.cpu().detach().numpy()
    gen_data = list()
    gen_data = [generated_data[i, :opt.seq_len, :] for i in range(synth_size)]
    gen_data = np.array([(data * model.max_val) + model.min_val for data in gen_data])
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    # 1. Discriminative Score
    discriminative_score = list()
    print('Start discriminative_score_metrics')
    for i in range(opt.metric_iteration):
        print('discriminative_score iteration: ', i)
        temp_disc = discriminative_score_metrics(ori_data, gen_data)
        discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)
    print('Finish discriminative_score_metrics compute')

    # 2. Predictive score
    predictive_score = list()
    print('Start predictive_score_metrics')
    for i in range(opt.metric_iteration):
        print('predictive_score iteration: ', i)
        temp_predict = predictive_score_metrics(ori_data, gen_data)
        predictive_score.append(temp_predict)
    metric_results['predictive'] = np.mean(predictive_score)
    print('Finish predictive_score_metrics compute')

    # Save first 5 entries of generated data to an Excel file
    if opt.sample_to_excel:
        first_five_entries = gen_data[:5]
        m, n, p = first_five_entries.shape
        reshaped_data = first_five_entries.reshape(m * n, p)
        
        # Create a DataFrame and add a column to indicate the series ID
        column_names = ['eGFR', 'age', 'BMI', 'Hb', 'Alb', 'Cr', 'UPCR'] if opt.data_name == 'ckd' else [f'Feature_{i+1}' for i in range(p)]
        df = pd.DataFrame(reshaped_data, columns=column_names)
        df['Series_ID'] = np.repeat(np.arange(1, m + 1), n)
        
        # Reorder the columns to place 'Series_ID' at the beginning
        df = df[['Series_ID'] + column_names]

        excel_file = f'{filename}_first_five_entries.xlsx'
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, '..', 'Excel Results')
        excel_path = os.path.join(results_dir, excel_file)
        
        # Save to Excel
        df.to_excel(excel_path, index=False)
        print('First 5 entries of generated data saved to Excel')
    
    tu.save_results_to_excel(f'{filename}', metric_results, opt)

    print(metric_results)
    print(f'Iterations: {opt.iterations}, Data Name: {opt.data_name}, DP Enabled: {opt.use_dp}')
    if opt.use_dp and privacy_params:
        epsilon = privacy_params.get("epsilon", "N/A")
        delta = privacy_params.get("delta", 1e-5)
        print(f"(ε = {epsilon:.2f}, δ = {delta})")
    
    # 3. Visualization (PCA and tSNE)
    pv.plot_4pane(ori_data, gen_data, filename=f'{filename}.pdf')


