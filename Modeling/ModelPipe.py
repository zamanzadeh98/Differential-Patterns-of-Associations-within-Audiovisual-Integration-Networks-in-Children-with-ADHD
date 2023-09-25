# Online packages
import tqdm
import sklearn
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from mrmr import mrmr_classif
from imblearn import pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier


# Custom modules
import HyperParametersValues
from Metrics import MetricsCal
from HyperParametersSearch import hypSearch
from FeatureSelection import FeatureSelector



def ModelVersion3(x, y, n_split, random_state):
    """
    Version 1: Without resampling
    In this function the following steps will be done:
    1. cross validation
    2. Feature selection
    3. Scalling
    4. Modeling
    5. prediction
    6. Performance ex

    parameters
    -------------

    x

    y

    n_split

    random_state


    return
    -------------

    """



    ModelResults = {
                    
                    "BRF": {},
                    "EEC": {},
                    "XGB": {},
                    "Ensemble": {}
                    }

    # Cross validation <><><><<><><><><<><><><><<><><><><<><><><><<><><><><<>
    skf = StratifiedKFold(n_splits=n_split,
                        random_state=random_state,
                        shuffle=True)
    skf.get_n_splits(x, y)

    loop_counter = 0
    # cross validation loop 
    for train_index, val_index in skf.split(x, y):

        X_train, X_val = x.iloc[train_index,:], x.iloc[val_index,:]
        y_train, y_val = y[train_index], y[val_index]


        # Feature selection <><><><<><><><><<><><><><<><><><><<><><><><<><><><><<>
        ImpFeatures = FeatureSelector(X_train, y_train, random_state, n_split, 1000, 45, 2)
        print(len(ImpFeatures))
        X_train = X_train.loc[:, ImpFeatures]
        X_val = X_val.loc[:, ImpFeatures]


        
        # Scalling <><><><<><><><><<><><><><<><><><><<><><><><<><><><><<><><><<><>
        transformer = StandardScaler().fit(X_train)
        X_train = transformer.transform(X_train)
        X_val = transformer.transform(X_val)
        


        
        # Model creation <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
        
        # BalancedRandomForestClassifier(BRF)
        BRF = BalancedRandomForestClassifier(random_state=random_state)

        # EasyEnsembleClassifier (EEC)
        EEC = EasyEnsembleClassifier(random_state=random_state)

        # XGB
        XGB = XGBClassifier(random_state=random_state)
        





        # Modeling <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # <><><><><><>
        BRFtuned = hypSearch(BRF, HyperParametersValues.RF_ParameterSpace, 
                                                            X_train, y_train)
        BRFtuned.fit(X_train, y_train)
        # <><><><><><>
        XGBtuned = hypSearch(XGB, HyperParametersValues.XGB_ParameterSpace, 
                                                            X_train, y_train)
        XGBtuned.fit(X_train, y_train)
        # <><><><><><>
        EECtuned = hypSearch(EEC, HyperParametersValues.EEC_ParameterSpace, 
                                                            X_train, y_train)
        EECtuned.fit(X_train, y_train)




        # Adding performance <><><><><><><><><><><><><><><><><><><><><><><><><><><><>


        # 3
        ModelResults["BRF"][loop_counter] = MetricsCal(BRFtuned, 
                                                    X_val, 
                                                    y_val,
                                                    "BRF")
        # 5
        ModelResults["XGB"][loop_counter] = MetricsCal(XGBtuned, 
                                                    X_val, 
                                                    y_val,
                                                    "XGB")
        # 6
        ModelResults["EEC"][loop_counter] = MetricsCal(EECtuned, 
                                                    X_val, 
                                                    y_val,
                                                    "EEC")
        


    
        # Models Ensemble <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        models =[

                ("BRF", BRFtuned),
                ("XGB", XGBtuned),
                ("EEC", EECtuned)
                ]
        ensemble = VotingClassifier(estimators=models, voting="hard")
        ensemble.fit(X_train, y_train)
        # 7 
        ModelResults["Ensemble"][loop_counter] = MetricsCal(ensemble, 
                                                        X_val, 
                                                        y_val,
                                                        "ensemble")
        loop_counter +=1

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


    return ModelResults



  