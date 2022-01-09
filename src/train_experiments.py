import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import numpy as np


def experiment_network_hyperparameters(dataset_path):
    architectures = [[4], [16], [32]]
    # architectures = [list(map(str, a)) for a in architectures]
    clusters = {'salsa_combined': 16,
                'salsa_cpp': 16,
                'salsa_ps': 8,
                'cmu_salsa': 5,
                'cocktail_party': 5}

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        features_as_pos = [False]
        for feats in features_as_pos:
            for collapse_reg in range(1, 5):
                for dropout_rate in range(1, 5):
                    for architecture in architectures:
                        cr = collapse_reg / 10
                        dr = dropout_rate / 10
                        lr = 0.001
                        arch = str(architecture).replace(' ', '')
                        dataset_name = os.path.basename(dataset_path)
                        n_clusters = clusters[dataset_name]

                        call_string = f'python src/train.py -F Experiments/{dataset_name}_hyperparams with ' \
                                      f'common.architecture={arch} ' \
                                      f'common.collapse_regularization={cr} ' \
                                      f'common.dropout_rate={dr} ' \
                                      f'common.n_clusters={n_clusters} ' \
                                      f'n_epochs=75 ' \
                                      f'common.learning_rate={lr} ' \
                                      f'common.frustum_length=1.25 ' \
                                      f'common.frustum_angle=0.7853981633974483 ' \
                                      f'common.edge_cutoff=0 ' \
                                      f'common.features_as_pos={feats} ' \
                                      f'common.select_frames_random=True ' \
                                      f'common.dataset_path=data/{dataset_path} '

                        futures.append(executor.submit(subprocess.run, call_string))

        for future in concurrent.futures.as_completed(futures):
            print(future.result())


def experiment_frustum(dataset_path, collapse_regularization=0.2, arch=None, drop_out=0.5, learning_rate=0.001):
    if arch is None:
        arch = [4]

    for frustum_length in np.arange(0.75, 1.75, 0.25):
        for frustum_angle in range(2, 5):
            for edge_cutoff in np.arange(0., 0.6, 0.1):
                frustum_angle_rad = frustum_angle * math.pi / 12
                dataset_name = os.path.basename(dataset_path)

                call_string = f'python src/train.py -F Experiments/{dataset_name}_frustum with  ' \
                              f'"common.architecture={arch}" ' \
                              f'"common.collapse_regularization={collapse_regularization}" ' \
                              f'"common.dropout_rate={drop_out}" ' \
                              f'"common.n_clusters={16}" ' \
                              f'"n_epochs=100" ' \
                              f'"common.learning_rate={learning_rate}" ' \
                              f'"common.frustum_length={frustum_length}" ' \
                              f'"common.frustum_angle={frustum_angle_rad}" ' \
                              f'"common.edge_cutoff={edge_cutoff}" ' \
                              f'"common.features_as_pos=True" ' \
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
    dataset = 'cmu_salsa'
    experiment_network_hyperparameters(dataset)
    metrics = get_best_metrics(fr'Experiments\{dataset}_hyperparams')
    print(metrics)
    # with open(fr'Experiments\{dataset}_frustum\best_accs.txt', 'w') as f:
    #     print(f'Best FULL f1: {metrics["max_full_f1"]["path"]} - {metrics["max_full_f1"]["score"]}', file=f)
    #     print(f'Best CARD f1: {metrics["max_card_f1"]["path"]} - {metrics["max_card_f1"]["score"]}', file=f)
    # experiment_network_hyperparameters(dataset)
    # data_path = ['salsa_cpp', 'salsa_ps', 'CMU_salsa_full']
    # for dataset in data_path:
    # metrics = get_best_metrics(fr'Experiments\{dataset}_hyperparams')
    # config = get_experiment_config(metrics['max_full_f1']['path'])
    # experiment_frustum(dataset,
    #                    arch=config['common']['architecture'],
    #                    drop_out=config['common']['dropout_rate'],
    #                    learning_rate=config['common']['learning_rate'],
    #                    collapse_regularization=config['common']['collapse_regularization']
    #                    )
    #
    # metrics = get_best_metrics(fr'Experiments\{dataset}_frustum')
    # with open(fr'Experiments\{dataset}_frustum\best_accs.txt', 'w') as f:
    #     print(f'Best FULL f1: {metrics["max_full_f1"]["path"]} - {metrics["max_full_f1"]["score"]}', file=f)
    #     print(f'Best CARD f1: {metrics["max_card_f1"]["path"]} - {metrics["max_card_f1"]["score"]}', file=f)
