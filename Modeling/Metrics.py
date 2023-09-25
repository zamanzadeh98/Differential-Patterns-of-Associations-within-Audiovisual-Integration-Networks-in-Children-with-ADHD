from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np

def MetricsCal(Model, X_test, y_test, name):
    """This function calculate the following
    metrics:
    1.Accuracy
    2.f1-score
    3.Specificity
    4.Sensitivity
    5.Auc-Score
    
    parameters
    -----------
    Model:object
    fitted model
    
    X_test:ndarray
    Test feature matrix
    
    y_test:array
    Test true label
    
    return
    -----------
    AucScore, specifity_list, sensitivity_list, f1_score_list, accuracy_list
    
    """
    
    if name == "XGB":
        y_pred = Model.predict_proba(X_test)[:, 1]

        new_threshold = 0.4

        # Adjust the classification based on the new threshold
        y_pred = (y_pred > new_threshold).astype(int)
    else: 
        y_pred = Model.predict(X_test)

    #--------------------------------------------Metrics
    cm = confusion_matrix(y_test, y_pred) #confu
    TP = cm[1,1] # true positive 
    TN = cm[0,0] # true negatives
    FP = cm[0,1] # false positives
    FN = cm[1,0] # false negatives
    Specificity = TN/(TN+FP)
    Sensitivity = TP/(FN+TP)

    # Saving each of the metrics
    AUC = roc_auc_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred)
    ACC = accuracy_score(y_test, y_pred)

    try: 
        y_proba = Model.predict_proba(X_test)
        AUC_prob = roc_auc_score(y_test, y_proba[:,1])
    except: AUC_prob = np.nan     
        
    
    
    result = {"AUC": AUC,
            "Specificity": Specificity,
            "Sensitivity": Sensitivity,
            "f1": f1,
            "ACC": ACC,
            "AUC_prob": AUC_prob}

    return result
