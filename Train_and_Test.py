import numpy as np
from TimeGAN import Timegan
import matplotlib.pyplot as plt
from metrics.torch_discriminative_metric import discriminative_score_metrics
from metrics.torch_predictive_metric import predictive_score_metrics
from metrics.torch_visualization_metric import visualization
import Plotting_and_Visualization as pv
import torch_utils as tu

def train(ori_data, opt, checkpoint_file):
    #checkpoint_file = opt.checkpoint_file
    # Model Setting
    #print(ori_data)
    model = Timegan(ori_data, opt, checkpoint_file)

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(model.start_epoch['embedding'], opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_embedder()
        if (i) % 100 == 0:
            print(f'step: {str(i)}/{str(opt.iterations)}, e_loss: {str(np.round(np.sqrt(model.E_loss_T0.item()), 4))}')
        
        if (i+1) % 100 == 0:
            #print(f'step: {str(i+1)}/{str(opt.iterations)}, e_loss: {str(np.round(np.sqrt(model.E_loss_T0.item()), 4))}')
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
        
        if (i+1) % 100 == 0:
            #print(f'step: {str(i+1)}/{str(opt.iterations)},  g_loss_s: {str(np.round(np.sqrt(model.G_loss_S.item()), 4))}')
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
        # Discriminator training
        model.gen_batch()
        model.batch_forward()
        model.train_discriminator()

        # Print multiple checkpoints
        if (i) % 100 == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', d_loss: ' + str(np.round(model.D_loss.item(), 4)) +
                  ', g_loss_u: ' + str(np.round(model.G_loss_U.item(), 4)) +
                  ', g_loss_s: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)) +
                  ', g_loss_v: ' + str(np.round(model.G_loss_V.item(), 4)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))

        if (i+1) % 100 == 0:
            # print('step: ' + str(i+1) + '/' + str(opt.iterations) +
            #       ', d_loss: ' + str(np.round(model.D_loss.item(), 4)) +
            #       ', g_loss_u: ' + str(np.round(model.G_loss_U.item(), 4)) +
            #       ', g_loss_s: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)) +
            #       ', g_loss_v: ' + str(np.round(model.G_loss_V.item(), 4)) +
            #       ', e_loss_t0: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
            phase3_epochnum = {'embedding': opt.iterations, 'supervisor': opt.iterations, 'joint': i+1}
            model.save_checkpoint(phase3_epochnum, checkpoint_file)
    print('Finish Joint Training')

    # Save trained networks
    #model.save_trained_networks()

def test(ori_data, opt, filename):

    print('Start Testing')
    # Model Setting
    model = Timegan(ori_data, opt, filename)
    #model.load_trained_networks()

    # Synthetic data generation
    if opt.synth_size != 0:
        synth_size = opt.synth_size
    else:
        synth_size = len(ori_data)
    generated_data = model.gen_synth_data(synth_size)
    generated_data = generated_data.cpu().detach().numpy()
    gen_data = list()
    for i in range(synth_size):
        temp = generated_data[i, :opt.seq_len, :]
        gen_data.append(temp)
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    #if not opt.only_visualize_metric:
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
    
    print(metric_results)
    # 3. Visualization (PCA and tSNE)
    #visualization(ori_data, gen_data, 'pca', opt.output_dir)
    #visualization(ori_data, gen_data, 'tsne', opt.output_dir)
    filename = filename + '.pdf'
    pv.plot_4pane(ori_data, gen_data, filename=filename)
    # Print discriminative and predictive scores


    #for k in range(10):
    #    plt.plot(gen_data[k][:,0])
    #plt.show()