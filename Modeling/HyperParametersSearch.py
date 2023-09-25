from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def hypSearch (model, space, X_train, y_train):
    """This fnction will use grid search to 
    find the best value for hyperparameters
    
    params
    -------
    model : class
    Defined model
    
    space: Dictionary
    defined hyperparameter space
    
    x_train : pandas.Dataframe
    feature matrix of the training dataset
    
    y_train: numpy array
    True label of the train data
    
    return
    ---------
    A model, defined using the best hyperparameters values"""

    cv = StratifiedKFold(n_splits=5,
                        random_state=42,
                        shuffle=True)
    GS = GridSearchCV(model,
                 param_grid=space,
                 cv=cv,
                 scoring = "f1",
                 n_jobs=-1,
                 refit=True
                 )

    
    GS_result = GS.fit(X_train, y_train)
    print(GS_result.best_params_)
    best_estimator = GS_result.best_estimator_

    
    return best_estimator