import numpy as np
import brainconn
import pandas as pd
import pickle



def RawAdjacencyMat(thr_list, save_path):
    """
    This function will create Adjacency matirx (with treshold)
    for future modeling


    parameters
    -----------
    thr_list: list
    List of desired tresholds

    save_path: str
    Where to save the data

    return
    ------------
    """
    # AVI network
    AVI_info = pd.read_csv("../Data/ROIs/AVI_ROIs.csv")
    ROIs_num = list(AVI_info["Channel Num"])
    ROIs_name = list(AVI_info["ROI_name"])

    # Metadata
    Metadata = pd.read_csv("../Data/MetaData/AllMetaData.csv")
    IDs = list(Metadata["ScanDir ID"])

    # Saving the ConnMatrix
    with open("../Data/ConnectivityMatrix/MutualInformatio_ConnMat.pickle", "rb" ) as file:
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
            # Upper triangle matrix
            value = value[np.triu_indices_from(value, k=1)]
            All_AdjacencyMat[key] = value


        FeaturesName = []
        nodes = [] # Considering the upper triangle matrix
        for i in range(len(ROIs_name)):
            for j in range(len(ROIs_name)):
                if (i,j) not in nodes and (j,i) not in nodes and i!=j:
                    FeaturesName.append(f"{ROIs_name[i]} and {ROIs_name[j]}")
                    nodes.append((i,j))

        FeaturesName.extend(["DX", 
                            "Age", 
                            "Handedness", 
                            "Gender", 
                            "Site",
                            "ScanDir ID"])

        FinalDictionary = {}

        for key in All_AdjacencyMat.keys():

            # The corresponding row of that id
            df_try = Metadata[Metadata["ScanDir ID"] == int(key)]

            
            DX = int(df_try["DX"])
            Age = float(df_try["Age"])
            
            if int(key) == 4125514: Handness = 0   
            else: Handedness = float(df_try["Handedness"])
            
            Gender = float(df_try["Gender"])
            Site = int(df_try["Site"])
            # print(All_AdjacencyMat[key].shape)
            # print(len(FeaturesName))
            # Adding the metadate
            FinalDictionary[key] = np.hstack([All_AdjacencyMat[key], 
                                                [DX, 
                                                Age, 
                                                Handedness, 
                                                Gender, 
                                                Site,
                                                key]])

        AdjConn = pd.DataFrame(FinalDictionary.values(), columns= FeaturesName)

        AdjConn.to_csv(f"{save_path}/AdjacencyConn_{thr}.csv")







if __name__ == "__main__":

    save_path = "../Data/RawFeatures"
    # Tresholding values
    thr_list = [0.95]
    RawAdjacencyMat(thr_list, save_path)