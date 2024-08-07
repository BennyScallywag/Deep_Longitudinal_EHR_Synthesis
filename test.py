from Plotting_and_Visualization import plot_original_vs_generated, plot_4pane
from torch_dataloading import sine_data_generation
#from sdmetrics.single_table.multi_column.statistical import KSTest
import sdmetrics
from scipy.stats import ks_2samp

ori1 = sine_data_generation(50, 24, 5)
ori2 = sine_data_generation(50, 24, 5)

#plot_4pane(ori1, ori2, filename='test')

# Load the demo data, which includes:
# - A dict containing the real tables as pandas.DataFrames.
# - A dict containing the synthetic clones of the real data.
# - A dict containing metadata about the tables.
real_data, synthetic_data, metadata = sdmetrics.load_demo()

# Obtain the list of multi table metrics, which is returned as a dict
# containing the metric names and the corresponding metric classes.
metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

# Run all the compatible metrics and get a report
#print(sdmetrics.compute_metrics(metrics, real_data, synthetic_data, metadata=metadata))


print(ori1[:][0][0])
print(ks_2samp(ori1[:][0][0], ori2[:][0][0]))
#print(sdmetrics.compute_metrics([sdmetrics.single_table.KSComplement], real_data, synthetic_data, metadata=metadata))
