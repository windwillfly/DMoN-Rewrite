import json
import os
import subprocess


def test_folds(dataset_path):

    for fold in range(1, 6):
        dataset_name = os.path.basename(dataset_path)
        best_data_path = os.path.join(f'experiments_{dataset_name}_folds_ps_params', f'{fold}')
        best_config = get_best_config(best_data_path)
        architecture = best_config['common']['architecture']
        dropout_rate = best_config['common']['dropout_rate']
        collapse_regularization = best_config['common']['collapse_regularization']
        n_clusters = best_config['common']['n_clusters']
        frustum_length = best_config['common']['frustum_length']
        frustum_angle = best_config['common']['frustum_angle']
        edge_cutoff = best_config['common']['edge_cutoff']
        dataset_fold_path = os.path.join(dataset_path + f'_fold{fold}', 'test')
        ckpt_path = os.path.join(best_data_path, 'checkpoints',
                                 'cp-best_full_acc.ckpt')

        experiment_folder = f'experiments_{dataset_name}_folds_ps_params_test'
        subprocess.run(
            f'python src/test.py -F {experiment_folder} with '
            f'common.architecture={architecture} '
            f'common.collapse_regularization={collapse_regularization} '
            f'common.dropout_rate={dropout_rate} '
            f'common.n_clusters={n_clusters} '
            f'common.frustum_length={frustum_length} '
            f'common.frustum_angle={frustum_angle} '
            f'common.edge_cutoff={edge_cutoff} '
            f'common.features_as_pos=True '
            f'common.total_frames="max" '
            f'common.dataset_path={dataset_fold_path} '
            f'checkpoint_path={ckpt_path}'
        )


def get_best_config(experiment_folder):
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    dataset_path = os.path.join('data', 'salsa_cpp')
    test_folds(dataset_path)
