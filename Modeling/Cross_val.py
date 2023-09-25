from utils import SummerizeResult
import matplotlib.pyplot as plt
from plots import ResultsPlot
from ModelPipe import *
import pandas as pd
import numpy as np
import pickle
import os


FeaturesDir = "../Data/ExtractedFeatures/Pearson"
with open("../Data/Test_IDs/TestID.pickle", "rb") as file:
    Test_IDs = pickle.load(file)


# Reading the Feature matrix
pathAVIN = os.path.join(FeaturesDir, "AVI_P_features_Thr0.875.csv")
df = pd.read_csv(pathAVIN)

# Putting the Test IDs
df = df[~df["ScanDir ID"].isin(Test_IDs)]

# preparing the feature set and labels
X, y = df.iloc[:, 1:-5], np.array(df.loc[:, "DX"])

# Modeling
n_split, random_state = 7, 0
ModelResults = ModelVersion3(X, y, n_split, random_state)

# Printing th mean and STD of the results
SummerizeResult(ModelResults)

# Plot the results
ResultsPlot(ModelResults)
