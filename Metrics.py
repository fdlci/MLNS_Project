import networkx as nx
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from mip import *
import re
from utils import community_to_dict


def rearrange_labels(pred_labels,labels):
    """ 
    The algorithms find the best labels by looking at the repartition of the differents labels
    Parameters:
        -----------
        pred_labels: (n, ) np.array
            labels predicted by the GMM model
        
        labels: (n, ) np.array
            true labels
        
        Returns:
        -----
        new_labels: (n, ) np.array    
    """
    
    #Creation of the repartition_labels matrix
    
    num_cluster_pred = np.unique(pred_labels).shape[0]
    num_cluster_true = np.unique(labels).shape[0]
    
    repartition_labels = np.zeros((num_cluster_pred,max(num_cluster_true,num_cluster_pred)))
    
    shape_x,shape_y = repartition_labels.shape

    for i in range(num_cluster_pred):
        
        cluster_i = np.where(pred_labels == i)
        labels_cluster_i = labels[cluster_i]
        values, counts = np.unique(labels_cluster_i,return_counts=True)

        for j,value in enumerate(values):
            repartition_labels[i,value] = counts[j]

    
    #To find the best labels. We have to resolve a problem of combinatorial optimization
    #We use the package mip to resolve the problem
    
    m = Model()

    x = [[m.add_var(var_type=BINARY) for j in range(shape_y)] for i in range(shape_x)]

    m.objective = maximize(xsum(repartition_labels[i,j]*x[i][j] 
                                for i in range(shape_x) for j in range(shape_y)))
    
    #Constraints to get a permutation of the labels
    for i in range(shape_x):
        m += xsum(x[i][j] for j in range(shape_y)) == 1

    for j in range(shape_y):
        m += xsum(x[i][j] for i in range(shape_x)) <= 1

    m.optimize()

    rearrangement = []
    for v in m.vars:
        rearrangement.append(v.x)
    
    rearrangement = np.array(rearrangement).reshape((shape_x,shape_y))
    
    #Optimal permutations of labels
    true_labels = np.array([np.argmax(rearrangement[i]) for i in range(num_cluster_pred)])

    n = pred_labels.shape[0]
    new_labels = np.zeros(n)
    for i in range(n):
        new_labels[i] = true_labels[int(pred_labels[i])]
        
    return new_labels

def get_pred_labels(partition):

    part = community_to_dict(partition)
    part = dict(sorted(part.items()))
    labels_pred = list(part.values())  

    return labels_pred

def get_gt_labels_karate():

    karate = nx.read_gml("karate.gml")
    part = dict(sorted(nx.get_node_attributes(karate, 'community').items()))
    true_labels = list(part.values())

    return true_labels

def get_gt_labels_email():

    labels = []

    with open('email-Eu-core-department-labels.txt') as f:
        for line in f:
            labels.append(int(re.split(r'(\t|\n|\s)\s*', line)[2]))
    
    return labels