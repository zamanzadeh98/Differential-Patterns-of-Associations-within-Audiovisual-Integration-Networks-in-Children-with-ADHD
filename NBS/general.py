from importlib.resources import path
import os
from tkinter.tix import Tree
import brainconn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nichord.chord import plot_chord
from decimal import Decimal


def edges(Subnetwork):
    """This function records the edges in tuple e.g (x, y)
    where x and y represent nodes number
    
    parameter
    -----------
    Subnetwork: matrix
    
    
    return
    -----------
    edges: list
    list of edges 
    """

    edges = []

    for i in range(len(Subnetwork)):
        for j in range(len(Subnetwork[0])):
            if Subnetwork[i, j] != 0:
                if ((i, j) not in edges and (j, i) not in edges):
                    edges.append((i, j))

    return edges

def NBS(thr_list_right, thr_list_left, mode, path, path_ROI):
    """
    This function plots a radar figure for the NBS.

    parameters
    -----------
    thr_list_right: list
        List of desired thresholds for right-tailed NBS

    thr_list_left: list
        List of desired thresholds for left-tailed NBS

    mode:str
        Pearson based or mutual information based connection

    path: str
    directory of data

    path_ROI: str
    directory of ROI file

    return
    ------------
    """

    # Loading <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    #  network
    network_info = pd.read_csv(path_ROI)
    ROIs_name = list(network_info["ROI_name"])

    # Metadata
    Metadata = pd.read_csv("../Data/MetaData/AllMetaData.csv")


    # loading the ConnMatrix
    with open(path, "rb") as file:
        All_ConnMatrix = pickle.load(file)

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # Grouping
    Normal, ADHD = [], []
    
    tau = Decimal("0." + "9"*10)
    # Looping over Connectivity matrices
    for key, value in All_ConnMatrix.items():
        # in order to all weights on the main diagonal (self-self connections)
        value = brainconn.utils.matrix.threshold_proportional(value, 0.99999999999999999999999999)

        # Apply Fisher r 2 z normalization if mode = pearson
        if mode == "pearson":
            value = np.arctanh(value)
        target = int(Metadata[Metadata["ScanDir ID"] == int(key)]["DX"])
        if target == 0: Normal.append(value)
        elif target == 1: ADHD.append(value)



    ADHD = np.array(ADHD).reshape((68, 68, 53))
    Normal = np.array(Normal).reshape((68, 68, 127))

    # ADHD < Normal => underconnectivity
    for thr in thr_list_right:
        pvals, Subnetwork_right, _ = brainconn.nbs.nbs_bct(x=ADHD, 
                                                        y=Normal,
                                                        thresh=thr, 
                                                        tail="right", 
                                                        verbose=False)


    # ADHD > Normal => overconnectivity
    for thr in thr_list_left:
        pvals, Subnetwork_left, _ = brainconn.nbs.nbs_bct(x=ADHD, 
                                                        y=Normal, 
                                                        thresh=thr, 
                                                        tail="left", 
                                                        verbose=False)
     
  

    # underconnectivity edges <><><><><><><><><><><><><><><><><><><><><><><><>
    edges_right = edges(Subnetwork_right)
    # overconnectivity edges <><><><><><><><><><><><><><><><><><><><><><><><>
    edges_left = edges(Subnetwork_left)


    # Creating labels for radar plot
    network_info["region_names"] = network_info["Region name"] + "(" + network_info['Hemisphere'] +"H)"
    

    idx_to_label = {}
    for i in range(68):
        idx_to_label[list(network_info["num"])[i]] = list(network_info["region_names"])[i]

    network_colors = {}
    for i in range(68):
        network_colors[list(network_info["region_names"])[i]] = "#232324"  # Assign black color

    print(len(edges_right))
    print(len(edges_left))
    
    network_order = ["CAL(RH)", "STG(RH)", "SPS(RH)", "HES(RH)",
                    "CAL(LH)", "STG(LH)", "SPS(LH)", "HES(LH)"]
    plt.figure(figsize=(11, 11))

    plot_chord(
        idx_to_label,
        edges_right,
        network_order=network_order,
        linewidths=7,
        alphas=0.9,
        do_ROI_circles=True,
        do_ROI_circles_specific=True,
        ROI_circle_radius=0.02,
        cmap = (0, 0, 0),
        network_colors=network_colors,  # Pass the updated network_colors dictionary
        mode = "under" # self added
    )

    plot_chord(
        idx_to_label,
        edges_left,
        linewidths=7,
        network_order=network_order,
        alphas=0.9,
        do_ROI_circles=True,
        do_ROI_circles_specific=True,
        ROI_circle_radius=0.02,
        cmap= (0.31, 0.65, 0.77),
        network_colors=network_colors,  # Pass the updated network_colors dictionary,
        mode = "over"
    )



    plt.savefig(
        f"{mode}_over13_under14_5v2subnetwork.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
    )


if __name__ == "__main__":
    # Specify the threshold values
    thr_list_right = [10.5]
    thr_list_left = [7.1]
    path_MI = "../Data/ConnectivityMatrix/MutualInformatio_ConnMat.pickle"
    path_pear = "../Data/ConnectivityMatrix/Pearson_ConnMat.pickle"
    path_ROI = "../Data/ROIs/AVI_ROIs.csv"

    NBS(thr_list_right, 
        thr_list_left, 
        "MI", 
        path_MI, 
        path_ROI)

    # MI, AVIN, Rght : 10.5 ==> 22 (2 subnetwork)
    # MI, AVIN, Left : 7.1 ==> 21 (4 subnetwork)

    # Pearosn, AVIN, right : 14 ==> 21 (2 subnetwork)
    # Pearosn, AVIN, left : 13 ==> 22 (5 subnetwork)