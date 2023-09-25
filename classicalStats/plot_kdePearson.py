import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

df = pd.read_csv("PearsonData2.csv")

ADHD = df[df["DX"] == 1]
Normal = df[df["DX"] == 0]

features_name = list(df.columns[1:-1])

for count, feature in enumerate(features_name):

    # Normality test ---------------------------
    dataDis = np.array(df.loc[:, feature])
    normality_pval = scipy.stats.shapiro(dataDis)[1]
    normality = normality_pval > 0.05

    ADHD_Feature = np.array(ADHD.loc[:, feature])
    Normal_Feature = np.array(Normal.loc[:, feature])

    if ADHD_Feature.mean() > Normal_Feature.mean():
        title_ = "ADHD > Normal"
    else:
        title_ = "ADHD < Normal"

    lim_ = (-500, 1500)
    if "Edge" in feature and count != 0:
        df_feature = list(df.loc[:, feature])
        lim_ = (np.min(df_feature), np.max(df_feature))
    elif "NESP" in feature:
        lim_ = (-1, 5)
    elif "Search" in feature:
        lim_ = (0, 20)
    elif "Pagerank" in feature:
        lim_ = (-0.5, 0.5)
    elif "Eigen" in feature:
        df_feature = list(df.loc[:, feature])
        lim_ = (np.min(df_feature), np.max(df_feature))
    elif "LCAV" in feature:
        df_feature = list(df.loc[:, feature])
        lim_ = (np.min(df_feature), np.max(df_feature))
    elif "MFPT" in feature:
        df_feature = list(df.loc[:, feature])
        lim_ = (np.min(df_feature), np.max(df_feature))

    if "Edge" in feature:
        feature = "EB " + feature[26:]
    if "NESP" in feature:
        feature = feature[: 5] + feature[13:]
    if "Search" in feature:
        feature = "SI " + feature[27:]

    fig, ax = plt.subplots(figsize=(7, 7))
    if normality:
        p_value = scipy.stats.ttest_ind(Normal_Feature, ADHD_Feature)[1]
        test_type = "t-test"
    else:
        p_value = scipy.stats.mannwhitneyu(Normal_Feature, ADHD_Feature)[1]
        test_type = "Mann-Whitney U-test"

    plt.title(f"{title_}, p = {p_value:.5f}", fontsize=26.5)
    plt.xlabel(feature, fontsize=26.5)
    sns.kdeplot(Normal_Feature, color='black', alpha=0.4, ax=ax, linewidth=3)
    sns.kdeplot(ADHD_Feature, color='black', alpha=1, ax=ax, linewidth=3)
    plt.xlim(lim_)
    plt.yticks([])

    if "SI" in feature:
        xtick_labels = [0, 4, 8, 12, 16, 20]
        plt.xticks(xtick_labels, fontsize=23)
    elif "Betweenness centrality" in feature:
        xtick_labels = [-500, 0, 500, 1000, 1500]
        plt.xticks(xtick_labels, fontsize=23)
    elif "Eigen" in feature:
        xtick_labels = [0.02, 0.08, 0.16, 0.25]
        plt.xticks(xtick_labels, fontsize=23)
    elif "MFPT" in feature:
        xtick_labels = [-500, 10000, 25000]
        plt.xticks(xtick_labels, fontsize=23)
    elif "EB" in feature:
        xtick_labels = [0, 500, 1500, 2500]
        plt.xticks(xtick_labels, fontsize=23)
    else:
        xtick_labels = plt.xticks()[1]
        plt.xticks(fontsize=23)

    plt.rcParams['figure.figsize'] = [3, 1]
    sns.despine(left=True)
    ax.set(ylabel=None)
    plt.savefig(f"/home/zaman/Desktop/thesis new documents/AVIN/Pearson/Modeling_Features/features{count}.png",
                dpi=300, pad_inches=10)
