import random
from typing import List, Union, Optional

import networkx as nx
import numpy as np
import tensorflow as tf
from sacred import Ingredient

from models import gcn, dmon
from src.utilities.converters import SalsaConverter
from src.utilities.graph import normalize_graph

common_ingredient = Ingredient('common')


@common_ingredient.config
def cfg():
    architecture = [4, 4, 2]
    collapse_regularization = 0.2
    dropout_rate = 0.4
    n_clusters = 8
    learning_rate = 0.0012
    frustum_length = 1
    frustum_angle = 1
    edge_cutoff = 1
    features_as_pos = True
    select_frames_random = False
    dataset_path = 'Data/CMU_salsa_full'
    edges_from_gt = False
    seed = 42


@common_ingredient.capture
def get_training_and_validation_graphs(all_graphs: List[nx.Graph],
                                       select_frames_random: bool):
    """Splits the list of graphs into training and validation graphs."""
    if select_frames_random:
        # Randomly select training and validation graphs.
        random.shuffle(all_graphs)

    training_split = int(len(all_graphs) * 0.85)
    training_graphs = all_graphs[:training_split]
    validation_graphs = all_graphs[training_split:]

    return training_graphs, validation_graphs


@common_ingredient.capture
def convert_salsa_to_graphs(dataset_path: str,
                            edge_cutoff: float,
                            frustum_length: float,
                            frustum_angle: float,
                            edges_from_gt: bool):
    sc = SalsaConverter(root_folder=dataset_path, edges_from_gt=edges_from_gt)
    return sc.convert(edge_distance_threshold=edge_cutoff, frustum_length=frustum_length,
                      frustum_angle=frustum_angle)


@common_ingredient.capture
def create_dmon(
                features_as_pos: bool,
                learning_rate: float,
                training_graph: nx.Graph):
    adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(training_graph,
                                                                                  features_as_pos=features_as_pos)
    feature_size = features.shape[1]

    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    model = build_dmon(input_features, input_graph, input_adjacency)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer)
    return model, optimizer


def convert_scipy_sparse_to_sparse_tensor(
        matrix):
    """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

    Args:
      matrix: A scipy sparse matrix.

    Returns:
      A ternsorflow sparse matrix (rank-2 tensor).
    """
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(
        np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
        matrix.shape)


@common_ingredient.capture
def build_dmon(input_features,
               input_graph,
               input_adjacency,
               architecture,
               n_clusters,
               collapse_regularization,
               dropout_rate,
               ):
    """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.

    Args:
      input_features: A dense [n, d] Keras input for the node features.
      input_graph: A sparse [n, n] Keras input for the normalized graph.
      input_adjacency: A sparse [n, n] Keras input for the graph adjacency.

    Returns:
      Built Keras DMoN model.
    """
    output = input_features
    for n_channels in architecture:
        output = gcn.GCN(int(n_channels))([output, input_graph])
    pool, pool_assignment = dmon.DMoN(
        n_clusters,
        collapse_regularization=collapse_regularization,
        dropout_rate=dropout_rate)([output, input_adjacency])
    return tf.keras.Model(
        inputs=[input_features, input_graph, input_adjacency],
        outputs=[pool, pool_assignment])


@common_ingredient.capture
def generate_graph_inputs(graph: nx.Graph, features_as_pos: bool):
    # adjacency, features, labels, label_indices = load_npz(FLAGS.graph_path)
    adjacency = nx.adj_matrix(graph)

    if features_as_pos:
        features_of_nodes = nx.get_node_attributes(graph, 'feats')
        features = np.zeros((adjacency.shape[0], 2))
        feat_index = 0
        for k, v in features_of_nodes.items():
            features[feat_index] = v[:2]
            feat_index += 1
        features_x = 2 * (features[:, 0] - features[:, 0].min()) / (features[:, 0].max() - features[:, 0].min()) - 1
        features_y = 2 * (features[:, 1] - features[:, 1].min()) / (features[:, 1].max() - features[:, 1].min()) - 1
        features = np.vstack([features_x, features_y]).T
    else:
        features = np.identity(n=adjacency.shape[0])

    n_nodes = adjacency.shape[0]
    new_graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(normalize_graph(adjacency.copy()))
    return adjacency, features, new_graph, graph_normalized, n_nodes


def obtain_clusters(features, graph, graph_normalized, model, labels: Optional = None):
    # Obtain the cluster assignments.
    if len(model.input) == 4:
        _, assignments = model([features, graph_normalized, graph, labels], training=False)
    elif len(model.input) == 3:
        _, assignments = model([features, graph_normalized, graph], training=False)
    else:
        _, assignments = model([features, graph_normalized], training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    return clusters
