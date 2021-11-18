import json
import networkx as nx
import numpy as np
import os.path
import pandas as pd
from utilities.converters import SalsaConverter
from utilities.metrics import grode


def moving_average(arr, n):
    ret = np.cumsum(arr, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1:] /= n
    return ret


# Recursively calculates exponential moving average of a given array
# with a smoothing factor alpha
def exponential_moving_average(arr, alpha):
    smoothed = np.zeros(arr.shape)
    smoothed[0] = arr[0]
    for i in range(1, arr.shape[0]):
        smoothed[i] = (1 - alpha) * smoothed[i - 1] + alpha * arr[i]

    return smoothed


def read_results(results_txt):
    with open(results_txt) as f:
        lines = f.read().splitlines()
    full_f1 = float(lines[2].split('All Full F1 score:')[-1])
    card_f1 = float(lines[1].split('All Card F1 score:')[-1])
    return card_f1, full_f1


def get_best_config(experiment_folder):
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    return config


def post_process(data_path):
    best_full_index = []
    best_card_index = []

    all_fold_full_f1_scores = []
    all_fold_card_f1_scores = []
    #window_ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    window_ranges = np.arange(0.1, 0.3, 0.01)
    #window_ranges = np.arange(0.1, 1, 0.1)
    #window_ranges = np.arange(2, 10, 1)

    pd_rows_full_f1 = [[data_path] * 2 + ['Window Length'] * len(window_ranges),
                       [data_path] * 2 + list(window_ranges)]
    pd_rows_card_f1 = [[data_path] * 2 + ['Window Length'] * len(window_ranges),
                       [data_path] * 2 + list(window_ranges)]
    for fold in range(1, 6):
        folder = os.path.join(data_path, str(fold))
        previous_card, previous_full = read_results(os.path.join(folder, 'results.txt'))
        best_full = 0
        best_card = 0
        config = get_best_config(folder)
        if 'common' in config:
            config = config['common']
        sc = SalsaConverter(root_folder=config['dataset_path'], edges_from_gt=False)
        all_graphs = sc.convert(edge_distance_threshold=config['edge_cutoff'],
                                frustum_length=config['frustum_length'],
                                frustum_angle=config['frustum_angle'])
        assignments_file = os.path.join(folder, 'prediction_probs.npy')

        full_f1_scores = []
        card_f1_scores = []

        #window_ranges = [2, 3, 4, 5, 6, 7, 8, 9]
        for window_len in window_ranges:

            assignments = np.load(assignments_file)
            # new_preds = moving_average(assignments, window_len)
            new_preds = exponential_moving_average(assignments, window_len)
            #assignments[window_len - 1:] = new_preds
            clusters = new_preds.argmax(axis=-1)

            all_card_score = []
            all_full_score = []

            for frame_no, graph in enumerate(all_graphs):
                labels = np.array([m for m in nx.get_node_attributes(graph, 'membership').values()])
                _, _, _, _, _, card_f1_score = grode(labels, clusters[frame_no])
                _, _, _, _, _, full_f1_score = grode(labels, clusters[frame_no], crit='full')
                all_card_score.append(card_f1_score)
                all_full_score.append(full_f1_score)

            final_card_f1_score = np.mean(all_card_score)
            final_full_f1_score = np.mean(all_full_score)

            full_f1_scores.append(final_full_f1_score)
            card_f1_scores.append(final_card_f1_score)
            print(f'Window length: {window_len}')
            print(f'Cardinal F1 score: {final_card_f1_score}')
            print(f'Full F1 score: {final_full_f1_score}')
            if final_full_f1_score > previous_full and final_full_f1_score > best_full:
                print(f'New best full F1 score! New: {final_full_f1_score} - Old: {previous_full}')
                best_full = final_full_f1_score
            if final_card_f1_score > previous_card and final_card_f1_score > best_card:
                print(f'New best full F1 score! New: {final_card_f1_score} - Old: {previous_card}')
                best_card = final_card_f1_score

        pd_rows_full_f1.append(['Folds', fold] + full_f1_scores)
        pd_rows_card_f1.append(['Folds', fold] + card_f1_scores)
        all_fold_full_f1_scores.append(full_f1_scores)
        all_fold_card_f1_scores.append(card_f1_scores)

    #with pd.ExcelWriter(os.path.join(data_path, f'post_process_results.xlsx')) as writer:
    with pd.ExcelWriter(os.path.join(data_path, f'post_process_results_ema_precise.xlsx')) as writer:
        pd.DataFrame(pd_rows_full_f1).to_excel(writer, sheet_name='Full F1 Scores')
        pd.DataFrame(pd_rows_card_f1).to_excel(writer, sheet_name='Card F1 Scores')
    for best_full, best_card in zip(best_full_index, best_card_index):
        print(best_full)
        print(best_card)


if __name__ == '__main__':
    data_path = 'experiments_salsa_cpp_folds_test'
    post_process(data_path)
