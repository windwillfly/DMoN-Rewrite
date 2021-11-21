import math

import json
import numpy as np
import os
import subprocess


def experiment_network_hyperparameters(dataset_path):
    architectures = [[4, 4, 2], [4], [16], [32]]
    # architectures = [list(map(str, a)) for a in architectures]

    for collapse_reg in range(1, 5):
        for dropout_rate in range(2, 6):
            for architecture in architectures:
                cr = collapse_reg / 10
                dr = dropout_rate / 10
                lr = 7 * 0.0001
                arch = str(architecture).replace(' ', '')
                dataset_name = os.path.basename(dataset_path)

                call_string = f'python src/train.py -F Experiments/{dataset_name}_hyperparams with ' \
                              f'common.architecture={arch} ' \
                              f'common.collapse_regularization={cr} ' \
                              f'common.dropout_rate={dr} ' \
                              f'common.n_clusters=6 ' \
                              f'n_epochs=200 ' \
                              f'common.learning_rate={lr} ' \
                              f'common.frustum_length=1 ' \
                              f'common.frustum_angle=1 ' \
                              f'common.edge_cutoff=1 ' \
                              f'common.features_as_pos=True ' \
                              f'common.total_frames=max ' \
                              f'common.select_frames_random=True ' \
                              f'common.dataset_path=data/{dataset_path} '
                subprocess.run(call_string)


def experiment_frustum(dataset_path, collapse_regularization=0.5, arch=None, drop_out=0.5, learning_rate=0.0008):
    if arch is None:
        arch = [4]

    for frustum_length in np.arange(0.5, 1.5, 0.25):
        for frustum_angle in range(30, 90, 15):
            for edge_cutoff in np.arange(0.5, 1.5, 0.25):
                frustum_angle_rad = math.radians(frustum_angle)
                dataset_name = os.path.basename(dataset_path)

                call_string = f'python src/train.py -F Experiments/{dataset_name}_frustum with  ' \
                    f'"common.architecture={arch}" ' \
                    f'"common.collapse_regularization={collapse_regularization}" ' \
                    f'"common.dropout_rate={drop_out}" ' \
                    f'"common.n_clusters=8" ' \
                    f'"n_epochs=150" ' \
                    f'"common.learning_rate={learning_rate}" ' \
                    f'"common.frustum_length={frustum_length}" ' \
                    f'"common.frustum_angle={frustum_angle_rad}" ' \
                    f'"common.edge_cutoff={edge_cutoff}" ' \
                    f'"common.features_as_pos=True" ' \
                    f'"common.total_frames=max" ' \
                    f'"common.select_frames_random=True" ' \
                    f'"common.dataset_path=data/{dataset_path}" '
                subprocess.run(call_string)


def get_best_metrics(experiments_folder):
    def read_result(result_json):
        with open(result_json, 'r') as f:
            result = json.load(f)
        card_f1 = result['T=2/3 F1 Score']['values']
        full_f1 = result['T=1 F1 Score']['values']
        return card_f1, full_f1

    max_full_f1 = 0
    max_card_f1 = 0

    max_card_f1_path = None
    max_full_f1_path = None

    with os.scandir(experiments_folder) as it:
        for experiment_run_id in it:
            if experiment_run_id.is_dir():
                result_json = os.path.join(experiment_run_id.path, 'metrics.json')
                if not os.path.exists(result_json):
                    continue

                card_f1, full_f1 = read_result(result_json)
                max_card = np.mean(card_f1)
                max_full = np.mean(full_f1)

                if max_full > max_full_f1:
                    max_full_f1 = max_full
                    max_full_f1_path = experiment_run_id.path

                if max_card > max_card_f1:
                    max_card_f1 = max_card
                    max_card_f1_path = experiment_run_id.path

    return {'max_full_f1': {'score': max_full_f1, 'path': max_full_f1_path},
            'max_card_f1': {'score': max_card_f1, 'path': max_card_f1_path}}


def get_experiment_config(experiment_path) -> dict:
    config_json_path = os.path.join(experiment_path, 'config.json')
    with open(config_json_path) as f:
        config = json.load(f)

    return config


if __name__ == '__main__':
    dataset = 'salsa_cpp'
    #experiment_network_hyperparameters(dataset)
    # data_path = ['salsa_cpp', 'salsa_ps', 'CMU_salsa_full']
    # for dataset in data_path:
    metrics = get_best_metrics(fr'Experiments\{dataset}_hyperparams')
    config = get_experiment_config(metrics['max_full_f1']['path'])
    experiment_frustum(dataset,
                       arch=config['common']['architecture'],
                       drop_out=config['common']['dropout_rate'],
                       learning_rate=config['common']['learning_rate'],
                       collapse_regularization=config['common']['collapse_regularization']
                       )

    metrics = get_best_metrics(fr'Experiments\{dataset}_frustum')
    with open(fr'Experiments\{dataset}_frustum\best_accs.txt', 'w') as f:
        print(f'Best FULL f1: {metrics["max_full_f1"]["path"]} - {metrics["max_full_f1"]["score"]}', file=f)
        print(f'Best CARD f1: {metrics["max_card_f1"]["path"]} - {metrics["max_card_f1"]["score"]}', file=f)
