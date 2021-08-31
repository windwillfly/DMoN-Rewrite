import math
import operator
import os
import json
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx

from utilities.math_utils import calc_frustum, Point


def read_metrics(metrics_json):
    with open(metrics_json, 'r') as f:
        result = json.load(f)
    card_f1 = result['T=2/3 F1 Score']['values']
    card_f1_steps = result['T=2/3 F1 Score']['steps']
    full_f1 = result['T=1 F1 Score']['values']
    full_f1_steps = result['T=1 F1 Score']['steps']
    spectral_loss = result['Spectral Loss']['values']
    spectral_loss_steps = result['Spectral Loss']['steps']
    collapse_loss = result['Collapse Loss']['values']
    collapse_loss_steps = result['Collapse Loss']['steps']

    return card_f1, card_f1_steps, full_f1, full_f1_steps, spectral_loss, spectral_loss_steps, collapse_loss, collapse_loss_steps


def plot_single_experiment(experiment_name):

    card_f1, card_f1_steps, full_f1, full_f1_steps, spectral_loss, spectral_loss_steps, collapse_loss, collapse_loss_steps = read_metrics(
        os.path.join(experiment_name, f'metrics.json'))

    fig = plt.figure(figsize=(19.2, 10.8))
    plt.title('Grouping Accuracy over Training Epochs')
    plt.plot(card_f1_steps, card_f1, label=f'T=2/3')
    plt.plot(full_f1_steps, full_f1, label=f'T=1')

    plt.legend()
    acc_figure_save_path = os.path.join(experiment_name, 'results.png')
    plt.savefig(acc_figure_save_path)
    plt.close(fig)

    fig2 = plt.figure(figsize=(19.2, 10.8))

    plt.title('Network Losses over Training Epochs')

    plt.plot(spectral_loss_steps, spectral_loss, label=f'Spectral Loss')
    plt.plot(collapse_loss_steps, collapse_loss, label=f'Collapse Loss')
    plt.legend()
    loss_figure_save_path = os.path.join(experiment_name, 'losses.png')
    plt.savefig(loss_figure_save_path)
    plt.close(fig2)
    return acc_figure_save_path, loss_figure_save_path

def show_results_on_graph(graph: nx.Graph, frame_no: str, save_path: str, predictions: Optional[List] = None):
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))

    ax.set_title('DMON Training Results on Salsa Cocktail Party')

    ax.axis('equal')

    draw_gt_graph(ax, graph)
    draw_predictions(graph, predictions)

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    filename = f'dmon_{frame_no}.png'
    save_filepath = os.path.join(save_path, filename)
    plt.savefig(save_filepath)
    plt.close(fig)


def show_gt_graph(graph: nx.Graph, frame_no: str, draw_frustum=True):
    fig, ax = plt.subplots(figsize=(10.8, 7.2))

    ax.set_title(f'Salsa Cocktail Party - Frame {frame_no}')

    ax.axis('equal')

    draw_gt_graph(ax, graph, draw_frustum)

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.show()
    # os.makedirs('graphs', exist_ok=True)
    # plt.savefig(f'graphs/dmon_{frame_no}.png')
    return


def draw_predictions(graph: nx.Graph, predictions: Optional[List] = None):
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
                       '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    predicted_node_colors = [distinct_colors[i] for i in predictions]
    for color, feature in zip(predicted_node_colors, nx.get_node_attributes(graph, 'feats').values()):
        pos = feature[:2]
        new_pos = list(map(operator.add, pos, [0.05, 0.2]))
        plt.scatter(*new_pos, 200, color=color, edgecolors='black', linewidths=1, alpha=0.4)


def draw_gt_graph(ax, graph: nx.Graph, draw_frustum=True):
    node_pos = {node_n: feat[:2] for feat, node_n in
                zip(nx.get_node_attributes(graph, 'feats').values(),
                    nx.get_node_attributes(graph, 'person_no'))}
    node_edgecolors = ['black'] * graph.number_of_nodes()
    linewidths = [1 if c == 'black' else 5 for c in node_edgecolors]
    nx.draw(
        graph,
        node_color=list(nx.get_node_attributes(graph, 'color').values()),
        pos=node_pos,
        linewidths=linewidths,
        width=.3, ax=ax, node_size=200, edgecolors=node_edgecolors)

    if draw_frustum:
        # Draw view frustum
        for feat, color in zip(nx.get_node_attributes(graph, 'feats').values(),
                               nx.get_node_attributes(graph, 'color').values()):
            frustum = calc_frustum(feat)
            t1 = plt.Polygon(frustum, alpha=0.1, edgecolor='black', facecolor=color, linewidth=2)
            plt.gca().add_patch(t1)

    for person_feat, person_no in zip(nx.get_node_attributes(graph, 'feats').values(),
                                      nx.get_node_attributes(graph, 'person_no').values()):
        text_pos = Point(*person_feat[:2]) + Point(0.15, 0.15)
        ax.text(*text_pos, person_no)


def toy_frustum_example() -> nx.Graph:
    g = nx.Graph()

    nodes = []
    edges = []
    nodes.append((0,
                  {'membership': 0, 'color': '#27c7bd', 'feats': [0, 0, 0, 0],
                   'person_no': 0, 'ts': 0}))
    nodes.append((1,
                  {'membership': 0, 'color': '#27c7bd', 'feats': [1, 0, math.pi, 0],
                   'person_no': 1, 'ts': 0}))

    nodes.append((2,
                  {'membership': 1, 'color': '#01579b', 'feats': [3, 3, math.pi / 4, 0],
                   'person_no': 2, 'ts': 0}))
    nodes.append((3,
                  {'membership': 1, 'color': '#01579b', 'feats': [4, 3, math.pi + math.pi / 4, 0],
                   'person_no': 3, 'ts': 0}))

    nodes.append((4,
                  {'membership': 2, 'color': '#01579b', 'feats': [6, 6, math.pi / 2, 0],
                   'person_no': 4, 'ts': 0}))
    nodes.append((5,
                  {'membership': 2, 'color': '#01579b', 'feats': [7, 6, math.pi + math.pi / 2, 0],
                   'person_no': 5, 'ts': 0}))

    nodes.append((6,
                  {'membership': 3, 'color': '#01579b', 'feats': [9, 9, - math.pi / 3, 0],
                   'person_no': 6, 'ts': 0}))
    nodes.append((7,
                  {'membership': 3, 'color': '#01579b', 'feats': [10, 9, - (math.pi + math.pi / 3), 0],
                   'person_no': 7, 'ts': 0}))

    nodes.append((8,
                  {'membership': 4, 'color': '#01579b', 'feats': [12, 12, - 3 * math.pi / 4, 0],
                   'person_no': 8, 'ts': 0}))
    nodes.append((9,
                  {'membership': 4, 'color': '#01579b', 'feats': [13, 12, - (math.pi + 3 * math.pi / 4), 0],
                   'person_no': 9, 'ts': 0}))

    edges.append((0, 1))
    edges.append((2, 3))
    edges.append((4, 5))
    edges.append((6, 7))
    edges.append((8, 9))

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    return g


if __name__ == '__main__':
    graph = toy_frustum_example()
    show_gt_graph(graph, "0")
