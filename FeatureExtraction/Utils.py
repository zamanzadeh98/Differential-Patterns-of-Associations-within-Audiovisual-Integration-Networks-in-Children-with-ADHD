from sklearn.feature_selection import mutual_info_regression
import brainconn
import numpy as np

def custom_mi_reg(a, b):
    """This function will extract the mutual information between two regions
    
    parameters
    -----------
    a: numpy array
    fMRI signal of one ROI

    b: numpy array
    fMRI signal of one ROI

    return
    ------------
    mutual_inf: float
    
    """
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return  mutual_info_regression(a, b, random_state=0)[0] # should return a float value





def extract_feature(All_ConnMatrix: dict):

    """This function will extract hand crafted features from 
    connectivity matrices
    
    parameters
    ----------
    All_ConnMatrix: dict
    A dictiontion with IDs as key and connectivity matrices as value
    
    
    return
    ----------
    Features: dict
    """
    
    Features = {}
    
    for key, value in All_ConnMatrix.items():
        
        # If there is nan value in the connectivity matrices => continue
        if np.isnan(np.sum(value)).any():
            print("NAN value detected; ID:" ,key)
            continue
        
        # connection-lenght matrix
        conn_lenght = brainconn.utils.matrix.weight_conversion(value,
                                                                wcm="lengths")

    #     # centrality
        centrality = brainconn.centrality.betweenness_wei(value)


        # eigenvector centrality
        eigen_centrality = brainconn.centrality.eigenvector_centrality_und(value)

        # pagerank_centrality
        page_centrality = brainconn.centrality.pagerank_centrality(value, d=.85)

        clustering
        clustering = brainconn.clustering.clustering_coef_wu(value)

        #Clustering regarding the positivity or negativity of weight
        clustering_pos_neg_partA, clustering_pos_neg_partB = brainconn.clustering.clustering_coef_wu_sign(value)

        #assortativity
        assortativity = brainconn.core.assortativity_wei(value, flag=0)

        # degree strenght
        degree_strenght = brainconn.degree.strengths_und(value)

        # degree number
        degree_number = brainconn.degree.degrees_und(value)

    #   # optimal community structure
        KCAV, KOMM = brainconn.modularity.modularity_finetune_und(value)

        # global efficiency
        conn_normalized_val = brainconn.utils.matrix.weight_conversion(value,
                                                                wcm="normalize")
        g_efficency = brainconn.distance.efficiency_wei(conn_normalized_val)

        # local efficiency
        l_efficiency = brainconn.distance.efficiency_wei(conn_normalized_val, local=True)

        #Modularity
        Modularity_partA, Modularity_partB = brainconn.modularity.modularity_louvain_und(value)

        #density
        density, _,_ = brainconn.physical_connectivity.density_und(value)

        # Edge betweenness 
        edge_between_partA, _ = brainconn.centrality.edge_betweenness_wei(conn_lenght)
        edge_between_partA = edge_between_partA[np.triu_indices_from(edge_between_partA, k=1)]

        # distance_wei_floyd
        distance_floyd_partA, distance_floyd_partB, _ = brainconn.distance.distance_wei_floyd(
                                                                        np.abs(value),
                                                                        transform="inv")
        distance_floyd_partA = distance_floyd_partA[np.triu_indices_from(distance_floyd_partA, k=1)]
        distance_floyd_partB = distance_floyd_partB[np.triu_indices_from(distance_floyd_partB, k=1)]

        # mean_first_passage_time
        mean_first_passage_time = brainconn.distance.mean_first_passage_time(
                                    conn_lenght)
        mean_first_passage_time = mean_first_passage_time[np.triu_indices_from(mean_first_passage_time, k=1)]

        # search_information
        search_information = np.nan_to_num(brainconn.distance.search_information(np.abs(conn_lenght),
                                                                    transform="inv", has_memory=True))
        search_information = search_information[np.triu_indices_from(search_information, k=1)]
        print(np.shape(search_information))

        Features[f"{key}"] = {"ScanDir ID" : key,
                            "Betweenness centrality " : centrality,
                            "Eigenvector centrality" : eigen_centrality,
                            "Pagerank centrality" : page_centrality,
                            "Clustering" : clustering,
                            "Clustering N" : clustering_pos_neg_partA,
                            "Clustering P" : clustering_pos_neg_partB,
                            "Assortativity" : assortativity,
                            "Degree strength" : degree_strenght,
                            "Degree number" : degree_number,
                            "KCAV" : KCAV,
                            "KOMM" : KOMM,
                            "Global efficiency" : g_efficency,
                            "Local efficiency" : l_efficiency,
                            "LCAV" : Modularity_partA,
                            "LOMM" : Modularity_partB,
                            "Density": density,
                            "Edge betweenness ":edge_between_partA,
                            "WSPL" : distance_floyd_partA,
                            "NESP" : distance_floyd_partB,
                            "MFPT" : mean_first_passage_time,
                            "Search_information" : search_information}

               
    return Features




def FeaturesName(Feature:dict, AVI_name:list):
    """This function will extract the name of the extracted features

    parameters
    --------------
    Feature: dict
    an instance of feature matrix

    AVI_name: list
    a list of ROIs name

    return
    -------------
    feature_name: list
    A list of feature name
    """

    feature_name = []
    
    for key, value in Feature.items():
        
        # in the feature_extraction function for the edge based features 
        # we only selected the upper triangle matrix. So we well store
        # the ROIs to only see them once 
        AlreadySeen = []

        if key == "{id}": continue
            
        # Global
        elif np.any(np.shape(value)) == False:
            feature_name.append(key)
            
        # Nodel
        elif np.shape(value)[0] == 68 and value.ndim == 1:
            for i in range(len(AVI_name)):
                feature_name.append(f"{key} of {AVI_name[i]}")

        
        # Edge-based
        elif value.shape[0] == 2278:
            for i in range(len(AVI_name)):
                for j in range(len(AVI_name)):
                    if (i,j) not in AlreadySeen and (j,i) not in AlreadySeen and i!=j:
                        feature_name.append(f"{key} between ({AVI_name[i]} and {AVI_name[j]})")
                        AlreadySeen.append((i,j))
        
        else: print("did not work")
            

    return feature_name



def AddDemographic(Features_dict, metadata):
    """This function will add metadate to the features
    dictionary
    
    paramters
    -----------
    Features_dict:dict
    
    metadata: pandas df
    metadata   
    
    return
    -----------
    Modified_Features_dict: dict
    """
    
    for key in Features_dict.keys():
        

        df = metadata[metadata["ScanDir ID"] == int(key)]

        if int(key) == 4125514: Handedness = 0
        else: Handedness = float(df["Handedness"])
        
        Features_dict[key]["DX"] = int(df["DX"])
        Features_dict[key]["Age"] = float(df["Age"])
        Features_dict[key]["Handedness"] = Handedness
        Features_dict[key]["Gender"] = float(df["Gender"])
        Features_dict[key]["Site"] = int(df["Site"])
    
    return Features_dict