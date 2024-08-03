from Plotting_and_Visualization import plot_original_vs_generated, plot_4pane
from torch_dataloading import sine_data_generation

ori1 = sine_data_generation(5000, 24, 5)
ori2 = sine_data_generation(5000, 24, 5)

plot_4pane(ori1, ori2, filename='test')
