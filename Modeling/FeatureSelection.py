import sklearn
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif, SelectKBest






def FeatureSelector(x, y, random_state, n_split, anovaK, mrmrK, min_rep):
    """This function select the best k features in a 5 fold cross
    validation using:
    1. selectkbest (ANOVA)
    2.MRMR algorithms
    
    parameters
    ----------
    x: pandas DataFrame
    Feature matrix
    
    y: numpy array
    label array
    
    random_state:float
    Reproducibility
    
    n_split: int
    Number of cross validation folds
    
    anovaK: int


    mrmrK: int
    Expected number of features in each of MRMR algorithms
    
    min_rep:int
    Minimum number of repitition along the folds to be
    counted as important
    
    
    return
    --------
    ImpFeatures:list
    List of best features name
    
    
    """

    # Cross validation <><><><><><><><><><><><><><><><><><><><><><><><>
    skf = StratifiedKFold(n_splits = n_split,
                            random_state = random_state,
                            shuffle=True)
    skf.get_n_splits(x, y)

   
    SelectedFeatures = []
    FeaturesName = np.array(x.columns)
    
    # Cross validation loop **************
    for train_index, _ in skf.split(x, y):

        X_train =  x.iloc[train_index,:] 
        y_train = y[train_index]

        # Handling NaN value with mean
        imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
        X_train = imputer.fit_transform(X_train) 

        # Scaling the inputs
        transformer = StandardScaler().fit(X_train) 
        X_train = pd.DataFrame(transformer.transform(X_train))

        # Step 1 (ANOVA)
        selector = SelectKBest(f_classif, k=anovaK)
        X_train = selector.fit_transform(X_train, y_train)

        # K best features 
        feat_if_imp = selector.get_support()
        feat_selected = np.where(feat_if_imp == True)[0]
        KbestFeatures =  FeaturesName[feat_selected]

        # step2 (MRMR)
        X_train = pd.DataFrame(X_train)
        MRMRfeatures = mrmr_classif(X_train,
                                         y_train, 
                                         K = mrmrK,
                                         show_progress=False
                                         )
        SelectedFeatures.extend(list(KbestFeatures[MRMRfeatures]))

    # Step 3
    FinalFeatures = []
    for i in np.unique(SelectedFeatures):

        if SelectedFeatures.count(i) > min_rep:
            FinalFeatures.append(i)

    return FinalFeatures
# *****************************