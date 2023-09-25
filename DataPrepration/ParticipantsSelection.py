# Aim:
# In this py file we will create a file which contain the metadate of the 
# participants who were recorded in site 3 and 6

# Packages
import pandas as pd
import glob
import numpy as np

base_directory = "../Data/MetaData/"

interested_col = ["ScanDir ID", "Site", "Gender", "Age", "Handedness",
                "DX", "Secondary Dx", "ADHD Measure", "ADHD Index",
                "Inattentive", "Hyper/Impulsive", "Verbal IQ", "Performance IQ",
                "QC_Rest_1", "QC_Rest_2", "QC_Rest_3", "QC_Anatomical_1"]

#  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
TrainMetaData = pd.read_csv("../Data/MetaData/subject_status_phenotype.csv").loc[:, interested_col]
TestMetaData = pd.read_csv("../Data/MetaData/allSubs_testSet_phenotypic_dx.csv").loc[:, interested_col]
MetaData = pd.concat([TrainMetaData, TestMetaData])


# Selecting IDs which belong to site 3 and 6 <><><><><><><><><><><><><><><><><><><><>
Sites = [3, 6, "3", "6", 3.0, 6.0]
MetaData = MetaData[MetaData["Site"].isin(Sites)]


# Putting out the left handed subjects <><><><><><><><><><><><><><><><><><><><><><><>
NotRighty = [0, 3, "L", 2]
MetaData = MetaData[~MetaData["Handedness"].isin(NotRighty)]  


# Putting out thoese that didn't pass SQC pipeline <><><><><><><><><><><><><><><><>
SQC_negative = [2292940, 2535204, 2920716,3103809, 1743472,
               1536593, 3560456, 2455205, 1386056, 2288903,
               3286474, 1696588, 2620872, 3684229, 8720244, 
               2920716, 2559559, 2054998, 1536593, 2845989, 
               2292940, 2535204, 2740232, 15034, 15015, 15020,
               15010, 15060, 15059, 15038, 15046, 15023]

MetaData = MetaData[~MetaData["ScanDir ID"].isin(SQC_negative)]



# Correcting the data type <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# MetaData = MetaData.astype("float32")


# Grouping all of the ADHD sbtype <><><><><><><><><><><><><><><><><><><><><><><><>
MetaData = MetaData.replace({"DX" : {3:1 , 2:1, '1':1, '0':0, '3':1, '2':1}})


# Age: 7-14
MetaData = MetaData[np.logical_and(MetaData["Age"]>=7,
                                    MetaData["Age"]<=14)]



print(MetaData["DX"].unique())
print("Number of subject: ", MetaData.shape[0])
print("Number of ADHD group: ", MetaData[MetaData["DX"]==1.0].shape[0])
print("Number of Control group: ", MetaData[MetaData["DX"]==0.0].shape[0])


MetaData.to_csv("../Data/MetaData/AllMetaData2.csv")
print("Done!")