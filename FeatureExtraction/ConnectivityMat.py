import glob
from Utils import custom_mi_reg
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")



def FeatureExract(data_directory, AVI_info_path, MetaData_path):
    """This function will create the connectivity matrices using MI

    parameters
    -----------
    data_directory: str
    Where the data is

    AVI_info_path: str
    where the AVI information is

    MetaData_path: str
    Where the metadata is


    return
    ----------
    All_ConnMatrix: dict
    dictionary of IDs (key) and connectivity matrices (value)
    
    """


    # Data path
    
    data_path = glob.glob(f"{data_directory}/*.csv")

    # AVI network
    AVI_info = pd.read_csv(AVI_info_path)
    ROIs_num = list(AVI_info["Channel Num"])

    # Metadata
    Metadata = pd.read_csv(MetaData_path)
    IDs = list(Metadata["ScanDir ID"])

    # Saving the ConnMatrix
    All_ConnMatrix = {}

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # Looping over the fMRI record and calculating the connectivity matrices
    for i in range(len(data_path)):

        # Extracting the ID of each data record
        id_ = os.path.split(data_path[i])[1][:-4] 
        # Selecting the ROIs
        data = pd.read_csv(data_path[i]).iloc[:,ROIs_num]

        if not int(id_) in IDs: continue
        else:
            ConnMatrix = data.corr(method=custom_mi_reg)
            All_ConnMatrix[id_] = ConnMatrix.to_numpy()

    return All_ConnMatrix




if __name__ == "__main__":

    data_directory = "/home/zaman/Documents/thesis data (pahse II)/Total-data"
    AVI_info_path = "../Data/ROIs/DMN_ROIs.csv"
    MetaData_path = "../Data/MetaData/AllMetaData.csv"
    save_path = "../Data/ConnectivityMatrix/DMNConnMat.pickle"

    Features = FeatureExract(data_directory, AVI_info_path, MetaData_path)

    with open(save_path, "wb") as file:
        pickle.dump(Features, file)

