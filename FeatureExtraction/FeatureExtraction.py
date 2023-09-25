# Packages <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
import pickle
from tqdm import tqdm
import brainconn
import os
import glob
import pandas as pd
import numpy as np
from Utils import extract_feature, AddDemographic, FeaturesName
import json
import warnings
warnings.filterwarnings("ignore")



def FeatureExract(thr_list, save_path):
    """This function will do the followings:

    """


    # AVI network
    AVI_info = pd.read_csv("../Data/ROIs/AVI_ROIs.csv")
    ROIs_num = list(AVI_info["Channel Num"])
    ROIs_name = list(AVI_info["ROI_name"])

    # Metadata
    Metadata = pd.read_csv("../Data/MetaData/AllMetaData.csv")
    IDs = list(Metadata["ScanDir ID"])

    # Saving the ConnMatrix
    with open("../Data/ConnectivityMatrix/Pearson_ConnMat.pickle", "rb" ) as file:
        All_ConnMatrix = pickle.load(file)


    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # Tresholding
    for thr in thr_list:
        All_ConnMatrix_copy = All_ConnMatrix.copy()

        # saving the Adjacency matices
        All_AdjacencyMat = {}
        # Looping over Connectivity matrices
        for key, value in All_ConnMatrix_copy.items():
            value = brainconn.utils.matrix.threshold_proportional(value, thr)
            All_AdjacencyMat[key] = value

    #     # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    #     # feature extraction
        FeaturesDict = extract_feature(All_AdjacencyMat)
        
        
        # Finding the Features name
        instance = FeaturesDict[list(FeaturesDict.keys())[10]]
        FeatName = FeaturesName(instance, ROIs_name)
         

        # Adding Demographic features to the features dataframe
        FeaturesDict = AddDemographic(FeaturesDict, Metadata)
        

        # Adding Demograph column names to `FeatName`
        FeatName.extend(["DX", 
                        "Age", 
                        "Handedness", 
                        "Gender", 
                        "Site"])


        ## Dict => pandas Dataframe
        total_arr = [] 
        for key , value1 in FeaturesDict.items():
            
            arr = []
            for value2 in value1.values():
                arr.append(value2)
            arr = np.hstack(arr)
            total_arr.append(arr)
        total_arr = np.vstack(np.array(total_arr))

        

        FeaturesDF = pd.DataFrame(total_arr, columns=FeatName)

        FeaturesDF.to_csv(f"{save_path}/AVI_P_features_Thr{thr}.csv")
        print("Done!")




        
if __name__ == "__main__":

    save_path = "../Data/ExtractedFeatures/Pearson"
    # Tresholding values
    thr_list = [0.99]
    FeatureExract(thr_list, save_path)

        

            
    



