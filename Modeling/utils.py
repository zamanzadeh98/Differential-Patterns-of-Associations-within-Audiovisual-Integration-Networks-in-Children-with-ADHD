
import pandas as pd
import pickle 
import numpy as np

def SummerizeResult(ModelResults):  



    Results = {

            "BRF": {},
            "EEC": {},
            "XGB": {},
            "Ensemble": {}
                    }

    for MtericName in ModelResults["XGB"][0].keys():
        
        res = []
        MetricNames = list(ModelResults.keys())

        for ModelName in ModelResults.keys():
            
            
            MetricValues = list(pd.DataFrame(ModelResults[ModelName]).T.loc[:,MtericName])
            res.append(MetricValues)

        df_plot = pd.DataFrame(res).T
        df_plot.columns = MetricNames 

        
        for key in Results.keys():
            Results[key][MtericName] = list(df_plot.loc[:, key])

    for model, model_result in Results.items():
        for  metric in model_result.keys():

            mean_val = np.nanmean(model_result[metric])
            std_val = np.nanstd(model_result[metric])
            print(f"Model : {model} and metric: {metric} ==> Mean : {mean_val:.4f}, STD = {std_val:.4f})")

        print("***************************")