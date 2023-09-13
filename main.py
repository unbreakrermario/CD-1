import utils.imports as inp
import utils.visuals as visu
import utils.processing as proc

data = inp.read_diabetes_dataset("data/diabetes.tab.txt")
print("archivo txt cargado")
correlation_data = proc.get_correlations(data)

visu.save_histograms(data)
visu.save_scatter_plots(data, correlation_data)
visu.save_histogram_correlations(data)

normalized_data = proc.normalize_diabetes_data(data)


