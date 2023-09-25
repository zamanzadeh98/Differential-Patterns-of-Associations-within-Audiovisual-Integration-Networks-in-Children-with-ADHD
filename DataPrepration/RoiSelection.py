import pandas as pd

# Creating the ROIs name and selecting their channel
AVI_network = pd.read_csv("../Data/ROIs/AllROI.csv")

AVI_network = AVI_network[~AVI_network["Region name"].isin(["INS", "IPG"])]

AVI_network["ROInum"] = AVI_network["ROInum"].astype(str)
AVI_network["ROI_name"] = (AVI_network["Hemisphere"] + 
        " " + 
        AVI_network["Region name"] + 
        ""+ 
        AVI_network["ROInum"])
        

AVI_network.to_csv("../Data/ROIs/AVI_ROIs.csv")