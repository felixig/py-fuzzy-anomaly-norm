import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi

import pandas as pd
import numpy as np

def get_highest_value_indices(df):
    # in case of draw in top positions, take more than one index
    df = df.round(3) # roundint to 3 decimals
    count = []
    for idx, element in df.iterrows():
        arr = element.to_numpy()
        max_value = np.max(arr)
        count.append(np.where(arr == max_value)[0])

    flat = [item for sublist in count for item in sublist]
    return np.array(flat)


def plot_overlapped_spider_plots(df, title="Spider Plot", filename=None):

    num_vars = df.shape[0]
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})

    for column in df.columns:
        values = df[column].tolist()
        values += values[:1] 
        ax.plot(angles, values, label=column)
        ax.fill(angles, values, alpha=0.1)  

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(df.index)
    ax.set_yticks([])
    ax.set_title(title, size=15, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

    #plt.show()
    plt.close()


if __name__ == "__main__":

    results_file  = sys.argv[1]

    df = pd.read_csv(results_file)
    print(df)

    title="ROC-AUC, detection (1)"
    dfaux = df[['1Pauc','1mAauc']]
    dfaux.columns = ['proba','memA']
    arr = get_highest_value_indices(dfaux)
    print(title, dfaux.columns.tolist(), np.unique(arr, return_counts=True))
    filename = "results/unsup_auc.pdf"
    plot_overlapped_spider_plots(dfaux, title=title, filename=filename)

    title="AMF1, detection (1)"
    dfaux = df[['1Pamf1','1mAamf1']]
    dfaux.columns = ['proba','memA']
    arr = get_highest_value_indices(dfaux)
    print(title, dfaux.columns.tolist(), np.unique(arr, return_counts=True))
    filename = "results/unsup_amf1.pdf"
    plot_overlapped_spider_plots(dfaux, title=title, filename=filename)

    title="AAP, detection (1)"
    dfaux = df[['1Paap','1mAaap']]
    dfaux.columns = ['proba','memA']
    arr = get_highest_value_indices(dfaux)
    print(title, dfaux.columns.tolist(), np.unique(arr, return_counts=True))
    filename = "results/unsup_aap.pdf"
    plot_overlapped_spider_plots(dfaux, title=title, filename=filename)

    title="ROC-AUC, modeling (2)"
    dfaux = df[['2Pauc_mean','2Mauc_mean']]
    dfaux.columns = ['proba','memA,memN']
    arr = get_highest_value_indices(dfaux)
    print(title, dfaux.columns.tolist(), np.unique(arr, return_counts=True))
    filename = "results/sup_auc.pdf"
    plot_overlapped_spider_plots(dfaux, title=title, filename=filename)

    title="AMF1, modeling (2)"
    dfaux = df[['2Pamf1_mean','2Mamf1_mean']]
    dfaux.columns = ['proba','memA,memN']
    arr = get_highest_value_indices(dfaux)
    print(title, dfaux.columns.tolist(), np.unique(arr, return_counts=True))
    filename = "results/sup_amf1.pdf"
    plot_overlapped_spider_plots(dfaux, title=title, filename=filename)

    title="AAP, modeling (2)"
    dfaux = df[['2Paap_mean','2Maap_mean']]
    dfaux.columns = ['proba','memA,memN']
    arr = get_highest_value_indices(dfaux)
    print(title, dfaux.columns.tolist(), np.unique(arr, return_counts=True))
    filename = "results/sup_aap.pdf"
    plot_overlapped_spider_plots(dfaux, title=title, filename=filename)

    # 1: experiment (1) - Anomaly Detection (unsupervised)
    # P: probability normalization
    # mA: Anomaly membership score
    # 2: experiment (2) - Anomaly Modeling (supervised)
    # M: Anomaly and Normmality membership degrees as two independent predictors
    # auc: ROC-AUC score
    # amf1: Adjusted Maximum F1 score
    # aap: Adjusted Average Precision score
    dfaux = df[['dataset', 
            '1Pauc', '1mAauc', '1Pamf1', '1mAamf1', '1Paap', '1mAaap', 
            '2Pauc_mean', '2Pauc_std', '2Mauc_mean', '2Mauc_std', 
            '2Pamf1_mean', '2Pamf1_std', '2Mamf1_mean', '2Mamf1_std', 
            '2Paap_mean', '2Paap_std', '2Maap_mean', '2Maap_std']].copy()

    dfaux['dataset'] = df['dataset'].str.extract(r'_(.*?)\.')
    filename = "results/tables.tex"
    dfaux.to_latex(filename, index=True, formatters={"name": str.upper}, float_format="{:.3f}".format)

