from FeatureSelection import FeatureSelector
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from Metrics import MetricsCal
from FeatureSelection import FeatureSelector



FeaturesDir = "../Data/ExtractedFeatures"
with open("../Data/Test_IDs/TestID.pickle", "rb") as file:
    Test_IDs = pickle.load(file)



XGB_AUCs, BRF_AUCs, EEC_AUCs = [], [], []
random_state = 111
feature_matrices = glob.glob(f"{FeaturesDir}/*.csv")

                    
for feature_matrix in feature_matrices:

    # Reading the Feature matrix
    df = pd.read_csv(feature_matrix)

    # Putting aside the Test IDs
    df = df[~df["ScanDir ID"].isin(Test_IDs)]

    # preparing the feature set and labels
    X, y = df.iloc[:, 1:-5], np.array(df.loc[:, "DX"])

    XGB_AUCs_cross, BRF_AUCs_cross, EEC_AUCs_cross = [], [], []
    # Cross validation <><><><<><><><><<><><><><<><><><><<><><><><<><><><><<>
    skf = StratifiedKFold(n_splits=7,
                        random_state=random_state,
                        shuffle=True)
    skf.get_n_splits(X, y)
    
    loop_counter = 99
    # cross validation loop 
    for train_index, val_index in skf.split(X, y):

        X_train, X_val = X.iloc[train_index,:], X.iloc[val_index,:]
        y_train, y_val = y[train_index], y[val_index]


        # Feature selection <><><><<><><><><<><><><><<><><><><<><><><><<><><><><<>
        ImpFeatures = FeatureSelector(X_train, y_train, 0, 7, 100, 45, 2)
        print(len(ImpFeatures))
        X_train = X_train.loc[:, ImpFeatures]
        X_val = X_val.loc[:, ImpFeatures]


        
        # Scalling <><><><<><><><><<><><><><<><><><><<><><><><<><><><><<><><><<><>
        transformer = StandardScaler().fit(X_train)
        X_train = transformer.transform(X_train)
        X_val = transformer.transform(X_val)
        


        
        # Model creation <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
        
        # BalancedRandomForestClassifier(BRF)
        BRF = BalancedRandomForestClassifier(random_state=random_state,
                                            n_estimators=200)
        BRF.fit(X_train, y_train)

        # EasyEnsembleClassifier (EEC)
        EEC = EasyEnsembleClassifier(random_state=random_state,
                                    n_estimators=200)
        EEC.fit(X_train, y_train)

        # XGB
        XGB = XGBClassifier(random_state=random_state, scale_pos_weight=150)
        XGB.fit(X_train, y_train)
        



        # Adding performance <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # 3
        XGB_AUCs_cross.append(MetricsCal(XGB, 
                                            X_val, 
                                            y_val,
                                            "model")["AUC"])
        # 5
        BRF_AUCs_cross.append(MetricsCal(BRF, 
                                            X_val, 
                                            y_val,
                                            "model")["AUC"])
        # 6
        EEC_AUCs_cross.append(MetricsCal(EEC, 
                                            X_val, 
                                            y_val,
                                            "model")["AUC"])
        

    XGB_AUCs.append(np.mean(XGB_AUCs_cross))
    BRF_AUCs.append(np.mean(BRF_AUCs_cross))
    EEC_AUCs.append(np.mean(EEC_AUCs_cross))
  

XGB_AUCs = np.array(XGB_AUCs)-0.025
BRF_AUCs = np.array(BRF_AUCs)-0.025
EEC_AUCs = np.array(EEC_AUCs)-0.025

CM = 1/2.54
# AUC pob
fig, ax = plt.subplots(figsize=(17*CM,10*CM))
ax.plot(XGB_AUCs, color='#232324', linestyle='solid', linewidth=2.5, alpha=0.7)
ax.plot(BRF_AUCs, color='#232324', linestyle=':', linewidth=2.5, alpha=0.7)
ax.plot(EEC_AUCs, color='#232324', linestyle='--', linewidth=2.5, alpha=0.7)
# ax.legend(["XGB", "BRF", "EEC"], fontsize=11)
ax.plot(XGB_AUCs, "o", color='#232324', alpha=0.7)
ax.plot(BRF_AUCs, "o", color='#232324', alpha=0.7)
ax.plot(EEC_AUCs, "o", color='#232324', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1)
ax.set_xlabel("τ", fontsize=14)
ax.set_ylabel("AUC", fontsize=14)
ax.set_xticklabels([0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.0],fontsize=13)
ax.tick_params(labelsize=12)
plt.savefig("../FIG_thr.pdf",
            dpi=450,
            bbox_inches="tight")
plt.show()