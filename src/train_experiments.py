import json
import math
import os
import subprocess

import numpy as np


def experiment_network_hyperparameters(dataset_path):
    architectures = [[4, 2], [4, 4], [4], [16], [32]]
    # architectures = [list(map(str, a)) for a in architectures]

    for dropout_rate in range(4, 7):
        for learning_rate in range(8, 13):
            for architecture in architectures:
                dr = dropout_rate / 10
                lr = learning_rate * 0.0001
                arch = str(architecture).replace(' ', '')

                call_string = f'python src/train.py -F Experiments/CMU_hyperparams with ' \
                              f'architecture={arch} ' \
                              f'collapse_regularization=0.5 ' \
                              f'dropout_rate={dr} ' \
                              f'n_clusters=8 ' \
                              f'n_epochs=150 ' \
                              f'learning_rate={lr} ' \
                              f'frustum_length=1 ' \
                              f'frustum_angle=1 ' \
                              f'edge_cutoff=1 ' \
                              f'features_as_pos=True ' \
                              f'eval_mode="normal" ' \
                              f'total_frames=max ' \
                              f'select_frames_random=True ' \
                              f'dataset_path=data/{dataset_path} '
                subprocess.run(call_string)


def experiment_frustum(dataset_path, arch=None, drop_out=0.5, learning_rate=0.0008):
    if arch is None:
        arch = [4]

    for frustum_length in np.arange(0.8, 1.4, 0.1):
        for frustum_angle in range(30, 70, 10):
            for edge_cutoff in np.arange(0.75, 1.5, 0.15):
                frustum_angle_rad = math.radians(frustum_angle)

                subprocess.run(
                    f'python src/train.py -F Experiments/CMU_frustum with  '
                    f'architecture={arch} '
                    f'collapse_regularization=0.5 '
                    f'dropout_rate={drop_out} '
                    f'n_clusters=8 '
                    f'n_epochs=150 '
                    f'learning_rate={learning_rate} '
                    f'frustum_length={frustum_length} '
                    f'frustum_angle={frustum_angle_rad} '
                    f'edge_cutoff={edge_cutoff} '
                    f'features_as_pos=True '
                    f'eval_mode="normal" '
                    f'total_frames=max '
                    f'select_frames_random=True '
                    f'dataset_path=data/{dataset_path} '
                )


def get_best_metrics(experiments_folder, return_full_f1_score=True):
    def read_result(result_json):
        with open(result_json, 'r') as f:
            result = json.load(f)
        card_f1 = result['T=2/3 F1 Score']['values']
        full_f1 = result['T=1 F1 Score']['values']
        # card_f1 = list(result['card_f1'].values())
        # full_f1 = list(result['full_f1'].values())
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

    if return_full_f1_score:
        return max_full_f1_path
    else:
        return max_card_f1_path


def get_experiment_config(experiment_path) -> dict:
    config_json_path = os.path.join(experiment_path, 'config.json')
    with open(config_json_path) as f:
        config = json.load(f)

    return config


if __name__ == '__main__':
    data_path = 'CMU_salsa_full'

    experiment_frustum(dataset_path=data_path)
    # experiment_network_hyperparameters(data_path)
    # best_full_f1_exp_path = get_best_metrics(r'Experiments\CMU_hyperparams')
    # config = get_experiment_config(best_full_f1_exp_path)
