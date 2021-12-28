import math

import json
import os
import subprocess
from find_best_experiment import get_best_metrics


def train_folds(best_config, dataset_path, best_experiment_path):
    dataset_folder = os.path.basename(dataset_path)
    experiment_name = '_'.join(best_experiment_path.split(os.path.sep)[1:])
    if 'common' in best_config:
        best_config = best_config['common']
    architecture = best_config['architecture']
    dropout_rate = best_config['dropout_rate']
    collapse_regularization = best_config['collapse_regularization']
    n_clusters = best_config['n_clusters']
    n_epochs = 100
    learning_rate = best_config['learning_rate']
    frustum_length = best_config['frustum_length']
    frustum_angle = best_config['frustum_angle']
    edge_cutoff = best_config['edge_cutoff']
    features_as_pos = best_config['features_as_pos']
    if 'edges_from_gt' not in best_config:
        edges_from_gt = False
    else:
        edges_from_gt = best_config['edges_from_gt']
    for fold in range(1, 6):
        dataset_fold_path = os.path.join(dataset_path + f'_fold{fold}', 'train')

        experiment_folder = os.path.join(f'Experiments_tests', f'{experiment_name}_folds')
        subprocess.run(
            f'python src/train.py -F {experiment_folder} with '
            f'common.architecture="{architecture}" '
            f'common.collapse_regularization={collapse_regularization} '
            f'common.dropout_rate={dropout_rate} '
            f'common.n_clusters={n_clusters} '
            f'n_epochs={n_epochs} '
            f'common.learning_rate={learning_rate} '
            f'common.frustum_length={frustum_length} '
            f'common.frustum_angle={frustum_angle} '
            f'common.edge_cutoff={edge_cutoff} '
            f'common.features_as_pos={features_as_pos} '
            f'common.dataset_path={dataset_fold_path} '
            f'common.edges_from_gt={edges_from_gt} '  # Graph edges initialized from ground truth
            f'common.select_frames_random=True '
            f'common.use_body_orientation=True '
            f'-d'
        )

def get_best_config(experiment_folder):
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    dataset_name = 'salsa_cpp'
    dataset_path = os.path.join('data', dataset_name)
    #best_metrics = get_best_metrics(os.path.join('Experiments', f'salsa_ps_frustum'))
    #best_experiment_path = best_metrics['max_full_f1']['path']
    best_experiment_path = os.path.join('Experiments', f'salsa_cpp_edge_prune', '0.2')
    best_config = get_best_config(best_experiment_path)
    train_folds(best_config, dataset_path, best_experiment_path)