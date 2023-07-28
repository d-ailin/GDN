#This code is an extension to the code base of GDN.
#The code visualizes the anomlaies similar to figure 3 (left) in the research paper


import os
import sys

import argparse
from main import Main

import torch
from torch import nn

import numpy as np
import networkx as nx

from scipy.stats import iqr

from matplotlib import pyplot as plt

from test import test
from evaluate import get_full_err_scores

anomaly_node_size = 80
default_node_size = 20

central_node_color = "yellow"
anomaly_node_color = "red"
default_node_color = "black"

anomaly_edge_color = "red"
default_edge_color = (0.35686275, 0.20392157, 0.34901961, 0.1)

train_config = {
    'batch': 128,
    'epoch': 100,
    'slide_win': 5,
    'dim': 64,
    'slide_stride': 5,
    'comment': '',
    'seed': 0,
    'out_layer_num': 1,
    'out_layer_inter_dim': 256,
    'decay': 0,
    'val_ratio': 0.1,
    'topk': 20,
}

env_config = {
    'save_path': '',
    'dataset': 'CTD',
    'report': 'best',
    'device': 'cpu',
    'load_model_path': ''
}


def compute_graph(model: nn.Module,
                  X: torch.tensor):

    n_samples, feature_num, slide_win = X.shape

    # Do a forward pass in the model
    with torch.no_grad():
        model(X, None)

    coeff_weights = model.gnn_layers[0].att_weight_1.cpu().detach().numpy()
    edge_index = model.gnn_layers[0].edge_index_1.cpu().detach().numpy()
    weight_mat = np.zeros((feature_num, feature_num))

    for i in range(len(coeff_weights)):
        edge_i, edge_j = edge_index[:, i]
        edge_i, edge_j = edge_i % feature_num, edge_j % feature_num
        weight_mat[edge_i][edge_j] += coeff_weights[i]

    weight_mat /= n_samples

    return weight_mat


if __name__ == "__main__":

    _, central_node_id, output_file = sys.argv

    main = Main(train_config, env_config, debug=False)
    model = main.model

    checkpoint = torch.load(os.path.join("data", env_config['dataset'], "best.pt"),
                            map_location=torch.device('cpu'))

    main.model.load_state_dict(checkpoint)

    _, train_result = test(model, main.train_dataloader)
    _, test_result = test(model, main.test_dataloader)

    # Compute the anomaly scores for the train and test datasets
    all_scores, all_normals = get_full_err_scores(train_result, test_result)

    X_train = main.train_dataset.x.float()
    n_samples, feature_num, slide_win = X_train.shape

    # Compute the graph structure from train data
    adj_mat = compute_graph(model, X_train)

    # Define the central node as the node with higest mean anomaly score
    if central_node_id == "auto":
        central_node = all_scores.mean(axis=1).argmax()
    else:
        central_node = int(central_node_id)

    # Find the neighboring nodes and selected the edges with highest value
    scores = np.stack([adj_mat[central_node], adj_mat[:, central_node]], axis=1)
    scores = np.max(scores, axis=1)

    # Define red nodes as the nodes with edge weight > 0.1
    red_nodes = list(np.where(scores > 0.1)[0])

    G = nx.from_numpy_matrix(adj_mat)
    G.remove_edges_from(nx.selfloop_edges(G))

    edges = [set(edge) for edge in G.edges()]
    edge_colors = [default_edge_color for edge in edges]

    node_colors = [default_node_color for i in range(feature_num)]
    node_sizes = [default_node_size for i in range(feature_num)]

    node_colors[central_node] = central_node_color
    node_sizes[central_node] = anomaly_node_size

    for node in red_nodes:
    
        if node == central_node:
            continue
    
        node_colors[node] = anomaly_node_color
        node_sizes[node] = anomaly_node_size

        edge_pos = edges.index(set((node, central_node)))
        edge_colors[edge_pos] = anomaly_edge_color

    pos = nx.spring_layout(G)

    x, y = pos[central_node]
    plt.text(x,y + 0.15,
             s=main.feature_map[central_node],
             bbox=dict(facecolor=central_node_color, alpha=0.5), horizontalalignment='center')

    print("Central Node:", main.feature_map[central_node])

    for node in red_nodes:
        x, y = pos[node]
        plt.text(x,y + 0.15,
                 s=main.feature_map[node],
                 bbox=dict(facecolor=anomaly_node_color, alpha=0.5), horizontalalignment='center')

        print("Red Node:", main.feature_map[node])

    nx.draw(G, pos,
            edge_color=edge_colors,
            node_color=node_colors,
            node_size=node_sizes)

    plt.savefig(output_file, format="PNG")

