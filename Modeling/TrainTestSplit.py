# For age: 7- 14 we will split data for train and test
# IDs of each set will be saved for future Modeling

import pandas as pd
import pickle

df = pd.read_csv("../Data/MetaData/AllMetaData.csv")



train_set = df.sample(frac=0.8,
                        random_state=0)
test_set = df.drop(train_set.index)

print("train size: ", train_set.shape[0]) # 144
print("test size: ", test_set.shape[0]) # 36
print("Number of ADHD group in train set: ", train_set[train_set["DX"]==1].shape[0]) # 42
print("Number of ADHD group in test set: ", test_set[test_set["DX"]==1].shape[0]) # 11

Test_ID = list(test_set["ScanDir ID"])
with open("../Data/Test_IDs/TestID.pickle", "wb") as file:
    pickle.dump(Test_ID, file)

