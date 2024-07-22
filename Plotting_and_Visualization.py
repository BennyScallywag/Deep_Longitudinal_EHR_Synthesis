import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import pickle
import os

def save_figure(fig, filename):
    '''Use Pickle to save all the data associated with a plot 
    such that it may be reloaded and edited'''
    with open(filename, 'wb') as f:
        pickle.dump(fig, f)

def load_figure(filename):
    '''Load figure data using pickle'''
    with open(filename, 'rb') as f:
        fig = pickle.load(f)
    return fig

def plot_original_vs_generated(original_data, generated_data, filename, num_samples=5):
    """
    Plots the original vs generated time-series data for comparison.
    
    Args:
    - original_data: List of original time-series data samples.
    - generated_data: List of generated time-series data samples.
    - num_samples: Number of samples to plot. Default is 5.
    - filename: Name to save the figure under
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Original vs Generated Data Comparison', fontsize=16)
    
    # Plot original data
    for i in range(num_samples):
        axes[0].plot(original_data[i][:, 0], label=f'Original Sample {i+1}')
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    #axes[0].legend()

    # Plot generated data
    for i in range(num_samples):
        axes[1].plot(generated_data[i][:, 0], label=f'Generated Sample {i+1}', linestyle='--')
    axes[1].set_title('Generated Data')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    #axes[1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, '..', 'Plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Create file path and save pickle data for re-obtaining plot later
    plot_path = os.path.join(plot_dir, filename)
    with open(plot_path + '.pkl', 'wb') as f:
        pickle.dump(fig, f)

    plt.savefig(plot_path, format='pdf')
    plt.show()

def get_clustering(ori_data, generated_data):
    """
    Using PCA and t-SNE for generated and original data visualization.
  
    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
  
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
  
    no, seq_len, dim = ori_data.shape  
  
    prep_data = np.mean(ori_data, axis=2).reshape(no, seq_len)
    prep_data_hat = np.mean(generated_data, axis=2).reshape(no, seq_len)
    
    # PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # t-SNE
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results_all = tsne.fit_transform(prep_data_final)
    
    tsne_results = tsne_results_all[:anal_sample_no, :]
    tsne_hat_results = tsne_results_all[anal_sample_no:, :]
    
    return pca_results, pca_hat_results, tsne_results, tsne_hat_results

def plot_4pane(original_data, generated_data, filename, num_samples=5):
    """
    Plots the original vs generated time-series data for comparison.
    
    Args:
    - original_data: List of original time-series data samples.
    - generated_data: List of generated time-series data samples.
    - filename: Name to save the figure under
    - num_samples: Number of samples to plot. Default is 5.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Original vs Generated Data Comparison', fontsize=16)
    
    # Plot original data
    for i in range(num_samples):
        axes[0, 0].plot(original_data[i][:, 0], label=f'Original Sample {i+1}')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    #axes[0, 0].legend(fontsize='small', loc='upper right')

    # Plot generated data
    for i in range(num_samples):
        axes[0, 1].plot(generated_data[i][:, 0], label=f'Generated Sample {i+1}', linestyle='--')
    axes[0, 1].set_title('Generated Data')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value')
    #axes[0, 1].legend(fontsize='small', loc='upper right')

    pca_results, pca_hat_results, tsne_results, tsne_hat_results = get_clustering(original_data, generated_data)
    
    # PCA Plot
    axes[1, 0].scatter(pca_results[:, 0], pca_results[:, 1], c='red', alpha=0.2, label='Original')
    axes[1, 0].scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c='blue', alpha=0.2, label='Synthetic')
    axes[1, 0].set_title('PCA plot')
    axes[1, 0].set_xlabel('x-pca')
    axes[1, 0].set_ylabel('y-pca')
    axes[1, 0].legend(fontsize='small', loc='upper right')

    # t-SNE Plot
    axes[1, 1].scatter(tsne_results[:, 0], tsne_results[:, 1], c='red', alpha=0.2, label='Original')
    axes[1, 1].scatter(tsne_hat_results[:, 0], tsne_hat_results[:, 1], c='blue', alpha=0.2, label='Synthetic')
    axes[1, 1].set_title('t-SNE plot')
    axes[1, 1].set_xlabel('x-tsne')
    axes[1, 1].set_ylabel('y-tsne')
    axes[1, 1].legend(fontsize='small', loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.35, wspace=0.15, top=0.9)  # Adjust the spacing as needed

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, '..', 'Plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Ensure the filename has the correct extensions
    plot_path = os.path.join(plot_dir, filename)

    with open(plot_path, 'wb') as f:
        pickle.dump(fig, f)

    plt.savefig(plot_path, format='pdf')
    plt.show()
