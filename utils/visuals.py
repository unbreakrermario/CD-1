import seaborn as sns
import matplotlib.pyplot as plt
import utils.processing as proc
from sklearn import metrics


def save_histogram(data, column):
    proc.check_output_folder("output/histograms")
    fig = sns.displot(data[column], kde=True)
    fig.savefig("output/histograms/histogram_"+column+".png")
    plt.close()


def save_histograms(data):
    for col in data.columns:
        save_histogram(data, col)


def save_scatter_plot(data, var1, var2, corr_value):
    proc.check_output_folder("output/scatterplots")
    new_fig = plt.figure()
    sns.scatterplot(data=data, x=var1, y=var2).set(title=var1+" vs "+var2+" r = "+corr_value)
    plt.savefig("output/scatterplots/scatter_"+var1+"_"+var2+".png")
    plt.close(new_fig)


def save_scatter_plots(data, correlations):
    columns_pending = data.columns
    for var1 in data.columns:
        columns_pending = columns_pending.drop(var1)
        for var2 in columns_pending:
            corr_value = str(correlations.loc[var1][var2])
            save_scatter_plot(data, var1, var2, corr_value)


def save_histogram_correlations(data):
    proc.check_output_folder("output")
    fig = sns.pairplot(data, hue="SEX")
    fig.savefig("output/All_histograms.png")
    plt.close()


def save_correlations_heatmap(data):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt='.2f').set(title="Correlations for Diabetes Dataset")
    plt.savefig("output/correlations_heatmap.png")
    plt.close(new_fig)


def save_confusion_matrix(confusion):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    sns.heatmap(confusion, annot=True, cmap='Blues')
    plt.savefig("output/confusion_matrix.png")
    plt.close(new_fig)


def save_roc_curve(diabetes_y_test, diabetes_y_pred):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    metrics.RocCurveDisplay.from_predictions(diabetes_y_test, diabetes_y_pred)
    plt.savefig("output/curve_ROC.png")
    plt.close(new_fig)


def save_edades_padres_heatmap(data):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    sns.heatmap(data, fmt='.2f').set(title="Edades Padres")
    plt.savefig("output/edades_heatmap.png")
    plt.close(new_fig)


def save_histogram_edades_padres(data):
    proc.check_output_folder("output")
    fig = sns.displot(data, kde=True)
    fig.savefig("output/hsitograma_edades_padres.png")
    plt.close()
