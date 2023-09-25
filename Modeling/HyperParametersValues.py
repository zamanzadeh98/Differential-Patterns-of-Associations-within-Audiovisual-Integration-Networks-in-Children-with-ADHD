from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


# 1. XgBoost
XGB_ParameterSpace = {
    'max_depth': [3, 5, 6, 7],
    'min_child_weight': [1, 4, 5],
    'learning_rate': [0.2, 0.3, 0.4, 0.8],
    'n_estimators': [50, 100, 150, 200, 300],
    'scale_pos_weight': [100, 150]
}
    

# 3. Random Forrest
RF_ParameterSpace = {
            'n_estimators': np.arange(150, 400, 50),
              'max_depth': np.arange(10, 50, 10),
              'min_samples_split': [2, 4, 6],
              'criterion': ['gini', 'entropy'],
            #   'min_samples_leaf': [1, 2, 3, 4],
              'max_features': ['sqrt', 'log2']
            }


# 9. EasyEnsembleClassifier(EEC)
EEC_ParameterSpace = {
    "base_estimator": [DecisionTreeClassifier(),
                     AdaBoostClassifier()],
                    "n_estimators": [10, 60, 100, 200, 500]
    
}
        