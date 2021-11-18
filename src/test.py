import os

import networkx as nx
import numpy as np
import tensorflow as tf
from sacred import Experiment

from utilities.common import generate_graph_inputs, build_dmon, common_ingredient, convert_salsa_to_graphs
from utilities.converters import SalsaConverter
from utilities.metrics import grode, pairwise_precision, pairwise_recall
from utilities.visualization import show_results_on_graph

tf.compat.v1.enable_v2_behavior()

ex = Experiment('DMoN Experiments', ingredients=[common_ingredient])


@ex.config
def cfg():
    checkpoint_path = ''
    seed = 42


def obtain_clusters(features, graph, graph_normalized, model):
    # Obtain the cluster assignments.
    _, assignments = model([features, graph_normalized, graph], training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    return clusters, assignments


@ex.automain
def main(checkpoint_path, _run):
    tf.config.experimental.set_visible_devices([], 'GPU')

    print('Starting test with config:')
    print(f'{_run.config}')

    all_graphs = convert_salsa_to_graphs()

    training_graph = all_graphs[0]
    labels = np.array([(m) for m in nx.get_node_attributes(training_graph, 'membership').values()])
    label_indices = np.arange(labels.shape[0])
    adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(training_graph)

    feature_size = features.shape[1]
    experiment_folder = _run.observers[0].dir

    print(f'Experiment folder: {experiment_folder}')

    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    model = build_dmon(input_features, input_graph, input_adjacency)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer, None)
    model.load_weights(checkpoint_path)

    all_card_score = []
    all_full_score = []
    all_pairwise_f1_score = []
    training_scores = []
    prediction_probs = []


    for frame_no, training_graph in enumerate(all_graphs):
        labels = np.array([m for m in nx.get_node_attributes(training_graph, 'membership').values()])
        adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(training_graph)
        clusters, assignments = obtain_clusters(features, graph, graph_normalized, model)
        prediction_probs.append(assignments)

        _, _, _, _, _, card_f1_score = grode(labels, clusters)
        _, _, _, _, _, full_f1_score = grode(labels, clusters, crit='full')
        precision = pairwise_precision(labels, clusters[label_indices])
        recall = pairwise_recall(labels, clusters[label_indices])
        pairwise_f1_score = 2 * precision * recall / (precision + recall)

        _run.log_scalar("card_f1_of_graph", card_f1_score, frame_no)
        _run.log_scalar("full_f1_of_graph", full_f1_score, frame_no)
        all_pairwise_f1_score.append(pairwise_f1_score)
        all_card_score.append(card_f1_score)
        all_full_score.append(full_f1_score)

        graphs_folder = os.path.join(experiment_folder, 'graphs')
        os.makedirs(graphs_folder, exist_ok=True)
        show_results_on_graph(training_graph, frame_no, graphs_folder, predictions=clusters, draw_frustum=False)
        training_scores.append([card_f1_score, full_f1_score])


    with open(os.path.join(experiment_folder, 'results.txt'), 'w') as f:
        print(f'Training F1 scores: {np.mean(training_scores, axis=0)}', file=f)
        print(f'All Card F1 score: {np.mean(all_card_score, axis=0)}', file=f)
        print(f'All Full F1 score: {np.mean(all_full_score, axis=0)}', file=f)

    np.save(os.path.join(experiment_folder, 'prediction_probs'), prediction_probs)
