import os

import networkx as nx
import numpy as np
import tensorflow as tf
from sacred import Experiment

from utilities.common import generate_graph_inputs, build_dmon
from utilities.converters import SalsaConverter
from utilities.metrics import grode, pairwise_precision, pairwise_recall

tf.compat.v1.enable_v2_behavior()

ex = Experiment('DMoN Experiments')


# ex.observers.append(FileStorageObserver('experiments'))


@ex.config
def cfg():
    architecture = [4]
    collapse_regularization = 0.2
    dropout_rate = 0.5
    n_clusters = 8
    n_epochs = 500
    learning_rate = 0.0008
    frustum_length = 1
    frustum_angle = 1
    edge_cutoff = 1
    features_as_pos = True
    total_frames = 'max'
    dataset_path = 'data/salsa_ps'
    checkpoint_path = 'experiments/Salsa Original - Full frames/checkpoints/cp-best_full_acc.ckpt'
    seed = 42


def obtain_clusters(features, graph, graph_normalized, model):
    # Obtain the cluster assignments.
    _, assignments = model([features, graph_normalized, graph], training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    return clusters, assignments


@ex.automain
def main(edge_cutoff, frustum_length, frustum_angle, features_as_pos, architecture, n_clusters, collapse_regularization,
         dropout_rate, total_frames, checkpoint_path, learning_rate, dataset_path, _run):
    tf.config.experimental.set_visible_devices([], 'GPU')

    print('Starting test with config:')
    print(f'{_run.config}')

    sc = SalsaConverter(root_folder=dataset_path, edges_from_gt=False)
    all_graphs = sc.convert(edge_distance_threshold=edge_cutoff, frustum_length=frustum_length,
                            frustum_angle=frustum_angle)

    training_graph = all_graphs[0]
    labels = np.array([(m) for m in nx.get_node_attributes(training_graph, 'membership').values()])
    label_indices = np.arange(labels.shape[0])
    adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(training_graph,
                                                                                  features_as_pos=features_as_pos)

    feature_size = features.shape[1]
    experiment_folder = _run.observers[0].dir

    print(f'Experiment folder: {experiment_folder}')

    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

    model = build_dmon(input_features, input_graph, input_adjacency, architecture,
                       n_clusters,
                       collapse_regularization,
                       dropout_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer, None)
    model.load_weights(checkpoint_path)

    all_card_score = []
    all_full_score = []
    all_pairwise_f1_score = []
    training_scores = []
    prediction_probs = []

    if isinstance(total_frames, str) and total_frames == 'max':
        total_frames = len(all_graphs)

    random_indexes = range(total_frames)

    for frame_no, training_graph in enumerate(all_graphs):
        labels = np.array([m for m in nx.get_node_attributes(training_graph, 'membership').values()])
        adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(training_graph,
                                                                                      features_as_pos=features_as_pos)
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

        if frame_no in random_indexes:
            training_scores.append([card_f1_score, full_f1_score])

    with open(os.path.join(experiment_folder, 'results.txt'), 'w') as f:
        print(f'Training F1 scores: {np.mean(training_scores, axis=0)}', file=f)
        print(f'All Card F1 score: {np.mean(all_card_score, axis=0)}', file=f)
        print(f'All Full F1 score: {np.mean(all_full_score, axis=0)}', file=f)

    np.save(os.path.join(experiment_folder, 'prediction_probs'), prediction_probs)
