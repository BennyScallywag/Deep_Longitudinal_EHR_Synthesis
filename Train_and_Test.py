import numpy as np
#from timegan import Timegan
from DP_timegan import DP_Timegan
from timegan import Timegan
import matplotlib.pyplot as plt
from metrics.torch_discriminative_metric import discriminative_score_metrics
from metrics.torch_predictive_metric import predictive_score_metrics
#from metrics.torch_visualization_metric import visualization
import Plotting_and_Visualization as pv
import latent_representation as latent
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
        Etotal_loss, Ebatch_count = 0, 0
        for X_batch in model.dataloader:
            model.X = X_batch.float().to(model.device)
            model.gen_batch()
            model.forward_embedder_recovery()
            model.train_embedder()

            Etotal_loss += np.sqrt(model.E_loss_T0.item())
            Ebatch_count += 1
        Eaverage_loss = Etotal_loss / Ebatch_count
        if (i) % 100 == 0:
            #print(f'step: {str(i)}/{str(opt.iterations)}, e_loss: {str(np.round(np.sqrt(model.E_loss_T0.item()), 4))}')
            print(f'step: {str(i)}/{str(opt.iterations)}, average_e_loss: {str(np.round(Eaverage_loss, 4))}')
        
        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase1_epochnum = {'embedding': i+1, 'supervisor': 0, 'joint': 0}
            model.save_checkpoint(phase1_epochnum, checkpoint_file)

            #epsilon, best_alpha = model.privacy_engine.accountant.get_privacy_spent(delta=1e-5)
            #print(f"(ε = {epsilon:.2f}, δ = {1e-5}) for α = {best_alpha}")
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(model.start_epoch['supervisor'],opt.iterations):
        Stotal_loss, Sbatch_count = 0, 0
        for X_batch in model.dataloader:
            model.X = X_batch.float().to(model.device)
            model.gen_batch()
            model.forward_supervisor()
            model.train_supervisor()

            Stotal_loss += np.sqrt(model.G_loss_S.item())
            Sbatch_count += 1
        Saverage_loss = Stotal_loss / Sbatch_count
        if (i) % 100 == 0:
            #print(f'step: {str(i)}/{str(opt.iterations)},  g_loss_s: {str(np.round(np.sqrt(model.G_loss_S.item()), 4))}')
            print(f'step: {str(i)}/{str(opt.iterations)}, average_s_loss: {str(np.round(Saverage_loss, 4))}')

        
        if (i+1) % 1000 == 0 or i==(opt.iterations-1):
            phase2_epochnum = {'embedding': opt.iterations, 'supervisor': i+1, 'joint': 0}
            model.save_checkpoint(phase2_epochnum, checkpoint_file)
    print('Finish Supervised-Only Training')

    # 3. Joint Training
    print('Start Joint Training')
    for i in range(model.start_epoch['joint'], opt.iterations):
        Gtotal_loss, Dtotal_loss, Jbatch_count = 0, 0, 0
        for X_batch in model.dataloader:
            model.X = X_batch.float().to(model.device)
            for kk in range(2):
                model.gen_batch()
                model.forward_generator_discriminator()
                model.train_generator()
                model.forward_embedder_recovery()
                model.train_embedder()
            # # Discriminator training
            model.gen_batch()
            model.forward_generator_discriminator()
            model.train_discriminator()
            
            Gtotal_loss += model.G_loss.item()
            Dtotal_loss += model.D_loss.item()
            Jbatch_count += 1
        Gaverage_loss = Gtotal_loss / Jbatch_count if Jbatch_count > 0 else 0
        Daverage_loss = Dtotal_loss / Jbatch_count if Jbatch_count > 0 else 0

        # Print multiple checkpoints
        if (i) % 10 == 0:
            print(
                f'step: {i}/{opt.iterations}, '
                f'd_loss: {np.round(Daverage_loss, 4)}, '
                f'g_loss: {np.round(Gaverage_loss, 4)}, '
                #f'g_loss_u: {np.round(model.G_loss_U.item(), 4)}, '
                #f'g_loss_v: {np.round(model.G_loss_V.item(), 4)}, '
                #f'e_loss_t0: {np.round(np.sqrt(model.E_loss_T0.item()), 4)}'
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
        #model.batch_forward()
        model.forward_embedder_recovery()
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
        #model.batch_forward()
        model.forward_supervisor()
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
            X_batch = X_batch.to(model.device)
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
    """Test the TimeGAN model by generating synthetic data and evaluating its performance using discriminative 
    and predictive scores, followed by visualization using PCA and t-SNE.
    ----------Inputs-----------
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
    synth_size = min(opt.synth_size, len(ori_data))
    generated_data = model.gen_synth_data()
    generated_data = generated_data.cpu().detach().numpy()
    gen_data = [generated_data[i, :opt.seq_len, :] for i in range(synth_size)]
    gen_data = np.array([(data * model.max_val) + model.min_val for data in gen_data])
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    predictive_score, discriminative_score = list(), list()
    # 1. Discriminative Score
    print('Start discriminative_score_metrics')
    for i in range(opt.metric_iteration):
        print('discriminative_score iteration: ', i)
        temp_disc = discriminative_score_metrics(ori_data, gen_data)
        discriminative_score.append(temp_disc)
    metric_results['discriminative'] = np.mean(discriminative_score)
    print('Finish discriminative_score_metrics compute')

    # 2. Predictive score
    print('Start predictive_score_metrics')
    for i in range(opt.metric_iteration):
        print('predictive_score iteration: ', i)
        temp_predict = predictive_score_metrics(ori_data, gen_data)
        predictive_score.append(temp_predict)
    metric_results['predictive'] = np.mean(predictive_score)
    print('Finish predictive_score_metrics compute')

    # Latent Representation Metrics
    print('Start latent representation metrics')
    latent_metric_results = {'MMD': 0, 'Mahalanobis Distance': 0, 'DCR': 0}
    if opt.metric_iteration > 0:
        ori_data, gen_data = torch.tensor(np.array(ori_data), dtype=torch.float32), torch.tensor(np.array(gen_data), dtype=torch.float32)
        latent_metric_results = latent.evaluate_latent_metrics(ori_data, gen_data, opt.hidden_dim, training_epochs=5000)
    print('Finish latent representation metrics')

    if opt.sample_to_excel:
        tu.sample_synthetic_table_to_excel(gen_data, opt, filename, num_series=5)
    
    tu.save_results_to_excel(f'{filename}', metric_results, latent_metric_results, opt)

    print(f'Utility Results: {metric_results}')
    print(f'Latent Results: {latent_metric_results}')
    print(f'Iterations: {opt.iterations}, Data Name: {opt.data_name}, DP Enabled: {opt.use_dp}')
    if opt.use_dp and privacy_params:
        epsilon = privacy_params.get("epsilon", "N/A")
        delta = privacy_params.get("delta", 1e-5)
        print(f"(ε = {epsilon:.2f}, δ = {delta})")
    
    # 3. Visualization (PCA and tSNE)
    pv.plot_4pane(ori_data, gen_data, filename=f'{filename}.pdf', num_samples=opt.num_samples_plotted)


