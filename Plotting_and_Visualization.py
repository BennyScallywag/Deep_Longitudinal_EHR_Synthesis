import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_original_vs_generated(original_data, generated_data, num_samples=5):
    """
    Plots the original vs generated time-series data for comparison.
    
    Args:
    - original_data: List of original time-series data samples.
    - generated_data: List of generated time-series data samples.
    - num_samples: Number of samples to plot. Default is 5.
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
    plt.show()