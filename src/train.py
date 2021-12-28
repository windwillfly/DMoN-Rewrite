# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
r"""Graph Clustering with Graph Neural Networks.

===============================
This is the implementation of our paper,
[Graph Clustering with Graph Neural Networks]
(https://arxiv.org/abs/2006.16904).

The included code creates a DMoN (Deep Modularity Network) as introduced in the
paper.

Example execution to reproduce the results from the paper.
------
# From google-research/
python3 -m graph_embedding.dmon.train \
--graph_path=graph_embedding/dmon/data/cora.npz --dropout_rate=0.5
"""
import os
import random

import networkx as nx
import numpy as np
import sacred.run
import tensorflow as tf
from sacred import Experiment
from tqdm import tqdm

import utilities.metrics as metrics
from utilities.common import get_training_and_validation_graphs, convert_salsa_to_graphs, create_dmon, \
    generate_graph_inputs, obtain_clusters, common_ingredient
from utilities.metrics import grode
from utilities.timer import Timer
from utilities.visualization import show_results_on_graph, plot_single_experiment

tf.compat.v1.enable_v2_behavior()

ex = Experiment('DMoN Experiments', ingredients=[common_ingredient])


@ex.config
def cfg():
    n_epochs = 250
    seed = 42


@ex.capture
def train_graph(_run, n_epochs, test_graphs, model, optimizer,
                training_graphs, validation_graphs):
    t = Timer()
    t.start('Preparation')
    experiment_folder = os.path.abspath(_run.observers[0].dir)
    os.makedirs(experiment_folder, exist_ok=True)
    print(f'Experiment folder: {experiment_folder}')
    dmon_training_inputs = []
    for g in training_graphs:
        adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(g)
        labels = np.array([m for m in nx.get_node_attributes(g, 'membership').values()])
        dmon_training_inputs.append(
            {'graph': graph, 'feat': features, 'graph_norm': graph_normalized, 'labels': labels})

    dmon_validation_inputs = []
    for g in validation_graphs:
        adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(g)
        labels = np.array([m for m in nx.get_node_attributes(g, 'membership').values()])
        dmon_validation_inputs.append(
            {'graph': graph, 'feat': features, 'graph_norm': graph_normalized, 'labels': labels})

    print(f'There are {len(training_graphs)} training, {len(validation_graphs)} validation graphs.')

    checkpoint_path = "{experiment_folder}/checkpoints/cp-{epoch}.ckpt"
    predictions = []
    best_card_acc = 0
    best_full_acc = 0
    t.stop()
    for epoch in tqdm(range(n_epochs)):
        metrics_array = []
        random.shuffle(dmon_training_inputs)

        t.start('Training loops')
        for inputs in dmon_training_inputs:
            features = inputs['feat']
            graph = inputs['graph']
            graph_normalized = inputs['graph_norm']
            labels = inputs['labels']
            # TO-DO add labels to input
            if len(model.input) == 4:
                loss_values, grads = grad(model, [features, graph_normalized, graph, labels])
            elif len(model.input) == 3:
                loss_values, grads = grad(model, [features, graph_normalized, graph])
            else:
                loss_values, grads = grad(model, [features, graph_normalized])

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        t.stop()

        t.start('Validation loops')
        for inputs in dmon_validation_inputs:
            features = inputs['feat']
            graph = inputs['graph']
            graph_normalized = inputs['graph_norm']
            labels = inputs['labels']

            clusters = obtain_clusters(features, graph, graph_normalized, model)
            _, _, _, _, _, card_f1_score = grode(labels, clusters)
            _, _, _, _, _, full_f1_score = grode(labels, clusters, crit='full')

            all_metrics = [loss_val.numpy() for loss_val in model.losses]
            all_metrics.extend([card_f1_score, full_f1_score])
            metrics_array.append(all_metrics)
            predictions.append(clusters.tolist())

        t.stop()
        avg_metrics = np.mean(metrics_array, axis=0)

        _run.log_scalar("Spectral Loss", avg_metrics[0], epoch)
        _run.log_scalar("Collapse Loss", avg_metrics[1], epoch)
        _run.log_scalar("T=2/3 F1 Score", avg_metrics[2], epoch)
        _run.log_scalar("T=1 F1 Score", avg_metrics[3], epoch)
        if avg_metrics[2] > best_card_acc:
            print(f'Best cardinality acc reached: {avg_metrics[2]}')
            ckpt_path = checkpoint_path.format(epoch='best_card_acc', experiment_folder=experiment_folder)
            print(f'Saving checkpoint to: {ckpt_path}')
            best_card_acc = avg_metrics[2]
            model.save_weights(ckpt_path)
        if avg_metrics[3] > best_full_acc:
            print(f'Best full acc reached: {avg_metrics[3]}')
            ckpt_path = checkpoint_path.format(epoch='best_full_acc', experiment_folder=experiment_folder)
            print(f'Saving checkpoint to: {ckpt_path}')
            best_full_acc = avg_metrics[3]
            model.save_weights(ckpt_path)

    model.save_weights(checkpoint_path.format(epoch=n_epochs, experiment_folder=experiment_folder))

    # copyfile(os.path.join(experiment_folder, 'metrics.json'), os.path.join(experiment_folder, 'metrics.json'))

    # Need to make sure we are saving 'metrics.json'
    _run._emit_heartbeat()

    plot_single_experiment(experiment_folder)

    model.load_weights(
        checkpoint_path.format(epoch='best_full_acc', experiment_folder=experiment_folder))
    all_card_score = []
    all_full_score = []
    all_pairwise_f1_score = []
    frustum_length = _run.config['common']['frustum_length']
    frustum_angle = _run.config['common']['frustum_angle']
    for frame_no, training_graph in enumerate(test_graphs):
        labels = np.array([(m) for m in nx.get_node_attributes(training_graph, 'membership').values()])
        label_indices = np.arange(labels.shape[0])

        adjacency, features, graph, graph_normalized, n_nodes = generate_graph_inputs(training_graph)
        clusters = obtain_clusters(features, graph, graph_normalized, model)

        _, _, _, _, _, card_f1_score = grode(labels, clusters)
        _, _, _, _, _, full_f1_score = grode(labels, clusters, crit='full')
        precision = metrics.pairwise_precision(labels, clusters[label_indices])
        recall = metrics.pairwise_recall(labels, clusters[label_indices])
        pairwise_f1_score = 2 * precision * recall / (precision + recall)

        _run.log_scalar("card_f1_of_graph", card_f1_score, frame_no)
        _run.log_scalar("full_f1_of_graph", full_f1_score, frame_no)
        all_pairwise_f1_score.append(pairwise_f1_score)
        all_card_score.append(card_f1_score)
        all_full_score.append(full_f1_score)

        # if training_graph in training_graphs:
        #     if full_f1_score < 0.25:
        #         show_results_on_graph(training_graph, predictions=clusters, frame_no=str(frame_no),
        #                               save_path=os.path.join(experiment_folder, 'valid_very_very_bad'),
        #                               frustum_length=frustum_length,
        #                               frustum_angle=frustum_angle)
        #     if 0.25 <= full_f1_score < 0.5:
        #         show_results_on_graph(training_graph, predictions=clusters, frame_no=str(frame_no),
        #                               save_path=os.path.join(experiment_folder, 'valid_very_bad'),
        #                               frustum_length=frustum_length,
        #                               frustum_angle=frustum_angle)
        #     if 0.5 <= full_f1_score < 0.75:
        #         show_results_on_graph(training_graph, predictions=clusters, frame_no=str(frame_no),
        #                               save_path=os.path.join(experiment_folder, 'valid_bad'),
        #                               frustum_length=frustum_length,
        #                               frustum_angle=frustum_angle)
        # else:
        #     if full_f1_score < 1:
        #         show_results_on_graph(training_graph, predictions=clusters, frame_no=str(frame_no),
        #                               save_path=os.path.join(experiment_folder, 'train_bad'),
        #                               frustum_length=frustum_length,
        #                               frustum_angle=frustum_angle)
    print('Overall Accuracies')
    print(f'Pairwise F1: {np.mean(all_pairwise_f1_score)}')
    print(f'Card F1 (T=2/3): {np.mean(all_card_score)}')
    print(f'Full F1 (T=1): {np.mean(all_full_score)}')


# Computes the gradients wrt. the sum of losses, returns a list of them.
def grad(model, inputs):
    with tf.GradientTape() as tape:
        pred = model(inputs, training=True)
        loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)


@ex.automain
def main(_run: sacred.run.Run):
    tf.config.experimental.set_visible_devices([], 'GPU')

    all_graphs = convert_salsa_to_graphs()

    model, optimizer = create_dmon(training_graph=all_graphs[0])

    training_graphs, validation_graphs = get_training_and_validation_graphs(all_graphs)

    train_graph(test_graphs=all_graphs,
                model=model,
                optimizer=optimizer,
                training_graphs=training_graphs,
                validation_graphs=validation_graphs)
