import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures


def test_folds(dataset_path, experiment_name):

    dataset_name = os.path.basename(dataset_path)
    dataset_video_mapping = {'salsa_cpp': 'salsa_cpp_cam4.avi',
                             'salsa_ps': 'salsa_ps_cam3.avi',
                             'salsa_combined': 'salsa_combined_cam3.avi',
                             'cocktail_party': r'D:\datasets\CocktailParty\cam4\helium_3.0',
                             'cmu_salsa': 'hd_00_07.mp4'}

    video_name = dataset_video_mapping[dataset_name]
    video_path = os.path.join('videos', video_name)

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for fold in range(1, 6):
            best_data_path = os.path.join(experiment_name, f'{fold}')
            best_config = get_best_config(best_data_path)
            if 'common' in best_config:
                best_config = best_config['common']
            architecture = best_config['architecture']
            dropout_rate = best_config['dropout_rate']
            collapse_regularization = best_config['collapse_regularization']
            n_clusters = best_config['n_clusters']
            frustum_length = best_config['frustum_length']
            frustum_angle = best_config['frustum_angle']
            edge_cutoff = best_config['edge_cutoff']
            features_as_pos = best_config['features_as_pos']
            edges_from_gt = best_config['edges_from_gt']
            dataset_fold_path = os.path.join(dataset_path + f'_fold{fold}', 'test')
            ckpt_path = os.path.join(best_data_path, 'checkpoints',
                                     'cp-best_full_acc.ckpt')

            experiment_folder = f'{experiment_name}_test'
            call_string = f'python src/test.py -F {experiment_folder} with '\
                f'visualization_video_path={video_path} '\
                f'common.architecture="{architecture}" '\
                f'common.collapse_regularization={collapse_regularization} '\
                f'common.dropout_rate={dropout_rate} '\
                f'common.n_clusters={n_clusters} '\
                f'common.frustum_length={frustum_length} '\
                f'common.frustum_angle={frustum_angle} '\
                f'common.edge_cutoff={edge_cutoff} '\
                f'common.features_as_pos={features_as_pos} '\
                f'common.dataset_path={dataset_fold_path} '\
                f'checkpoint_path={ckpt_path} '\
                f'common.edges_from_gt={edges_from_gt} '\
                f'common.use_body_orientation=True '\
                f'-d'

            futures.append(executor.submit(subprocess.run, call_string))

        for future in concurrent.futures.as_completed(futures):
            print(future.result())


def get_best_config(experiment_folder):
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    dataset_name = 'cocktail_party'
    dataset_path = os.path.join('data', dataset_name)
    experiment_name = os.path.join(f'Experiments_tests', f'{dataset_name}_hyperparams_4_folds')
    test_folds(dataset_path, experiment_name)
