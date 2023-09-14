import utils.imports as inp
import utils.visuals as visu
import utils.processing as proc

data = inp.read_diabetes_dataset("data/diabetes.tab.txt")
print("archivo txt cargado")
correlation_data = proc.get_correlations(data)

# visu.save_histograms(data)
# visu.save_scatter_plots(data, correlation_data)
# visu.save_histogram_correlations(data)

normalized_data = proc.normalize_diabetes_data(data)

training, test = proc.split_data(normalized_data, 0.7)
training_input = training["BMI"]
training_output = training["Y"]
test_input = test["BMI"]
test_output = test["Y"]

model = proc.simple_linear_regression(training_input,
                                      training_output)
test_predictions = proc.test_predictions(model, test_input)

coefficients = proc.get_coefficients(model)
print("Coefficients: ", coefficients)
MSE = proc.get_mean_squared_error(test_output,
                                  test_predictions)
print("Mean Squared Error: ", MSE)
R2 = proc.get_coefficient_determination(test_output,
                                        test_predictions)
print("R² Score: ", R2)
# var(mean) - var(line) / var(mean)
# R² es  el porcentaje de variación descrito por la relación de dos variables
