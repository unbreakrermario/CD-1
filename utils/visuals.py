import os
import seaborn as sns
import matplotlib.pyplot as plt


def check_output_folder(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
        print(path+" folder created")


def save_histogram(data, column):
    check_output_folder("output/histograms")
    fig = sns.displot(data[column], kde=True)
    fig.savefig("output/histograms/histogram_"+column+".png")
    plt.close()


def save_histograms(data):
    for col in data.columns:
        save_histogram(data, col)


def save_scatter_plot(data, var1, var2):
    check_output_folder("output/scatterplots")
    new_fig = plt.figure()
    sns.scatterplot(data=data, x=var1, y=var2)
    plt.savefig("output/scatterplots/scatter_"+var1+"_"+var2+".png")
    plt.close(new_fig)


def save_scatter_plots(data):
    for var1 in data.columns:
        for var2 in data.columns:
            if not var1 == var2:
                save_scatter_plot(data, var1, var2)


def save_histogram_correlations(data):
    check_output_folder("output")
    fig = sns.pairplot(data, hue="SEX")
    fig.savefig("output/All_histograms.png")
    plt.close()
