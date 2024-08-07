import matplotlib.pyplot as plt
import numpy as np

# Function to set up consistent plot formatting - this is the most important part,
#can call this wherever I am plotting something
def set_plot_formatting(use_tex=True, font_family='serif', font_size=11):
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": font_family,
        "font.serif": ["Times New Roman"],  # Adjust to the font used in your document
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "axes.grid": False,
        "grid.alpha": 0.5,
        "lines.linewidth": 1
    })

# Function to plot data
def plot_data(x, y, xlabel, ylabel, legend_label, filename, include_top_right_splines=False):
    fig, ax = plt.subplots()

    # Plot data
    ax.plot(x, y, label=legend_label, color='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add legend
    ax.legend()

    # Customize spines
    if not include_top_right_splines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Save figure as PDF
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Set up plot formatting
    set_plot_formatting()

    # Sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot data and save as PDF
    plot_data(x, y, xlabel='Time', ylabel='Amplitude', legend_label='Sine Wave', 
              filename='sine_wave_example', include_top_right_splines=False)
