import matplotlib.pyplot as plt
import pandas as pd


def ResultsPlot(ModelResults):

    i, j = 0, 0
    fig, axis = plt.subplots(2,3, figsize = (20,10))
    for MtericName in ModelResults["XGB"][0].keys():
        
        res = []
        MetricNames = list(ModelResults.keys())

        for ModelName in ModelResults.keys():
            
            
            MetricValues = list(pd.DataFrame(ModelResults[ModelName]).T.loc[:,MtericName])
            res.append(MetricValues)
    
        df_plot = pd.DataFrame(res).T
        df_plot.columns = MetricNames 

        
        
        axis[i,j].boxplot(df_plot)
        axis[i,j].set_title(f"{MtericName} score", fontweight="bold")
        axis[i,j].set_xticklabels(MetricNames, rotation = 45)
        axis[i,j].grid()
        axis[i,j].set_ylim(0, 1)


        i += 1
        if i == 2: i = 0; j += 1

    plt.show()
            
            