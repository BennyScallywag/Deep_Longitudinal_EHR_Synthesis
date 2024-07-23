import numpy as np
from TimeGAN import Timegan
import matplotlib.pyplot as plt
from metrics.torch_discriminative_metric import discriminative_score_metrics
from metrics.torch_predictive_metric import predictive_score_metrics
from metrics.torch_visualization_metric import visualization
import torch_utils as tu

def train(ori_data, opt, checkpoint_file):

    # Model Setting
    model = Timegan(ori_data, opt)

    # 1. Embedding network training
    print('Start Embedding Network Training')
    for i in range(opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_embedder()
        if i % 100 == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    for i in range(opt.iterations):
        model.gen_batch()
        model.batch_forward()
        model.train_supervisor()
        if i % 100 == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)))

    # 3. Joint Training
    print('Start Joint Training')
    for i in range(opt.iterations):
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
        if i % 100 == 0:
            print('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', d_loss: ' + str(np.round(model.D_loss.item(), 4)) +
                  ', g_loss_u: ' + str(np.round(model.G_loss_U.item(), 4)) +
                  ', g_loss_s: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)) +
                  ', g_loss_v: ' + str(np.round(model.G_loss_V.item(), 4)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
    print('Finish Joint Training')

    # Save trained networks
    model.save_trained_networks()