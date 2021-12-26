import math

import json
import numpy as np
import os
import subprocess

if __name__ == '__main__':
    dataset_path = 'salsa_combined'
    experiments_folder = os.path.join('Experiments', dataset_path)
    arch = [32]
    collapse_regularization = 0.2
    drop_out = 0.5
    learning_rate = 0.0012
    frustum_length = 1.25
    frustum_angle_rad = math.pi / 4
    edge_cutoff = 0.
    n_clusters = 16
    n_epochs = 100
    features_as_pos = False

    call_string = f'python src/train.py -F {experiments_folder} with  ' \
                  f'"common.architecture={arch}" ' \
                  f'"common.collapse_regularization={collapse_regularization}" ' \
                  f'"common.dropout_rate={drop_out}" ' \
                  f'"common.n_clusters={n_clusters}" ' \
                  f'"n_epochs={n_epochs}" ' \
                  f'"common.learning_rate={learning_rate}" ' \
                  f'"common.frustum_length={frustum_length}" ' \
                  f'"common.frustum_angle={frustum_angle_rad}" ' \
                  f'"common.edge_cutoff={edge_cutoff}" ' \
                  f'"common.features_as_pos={features_as_pos}" ' \
                  f'"common.select_frames_random=True" ' \
                  f'"common.dataset_path=data/{dataset_path}" '
    subprocess.run(call_string)

    #with open(fr'Experiments\{dataset_name}\best_accs.txt', 'w') as f:
    #    print(f'Best FULL f1: {metrics["max_full_f1"]["path"]} - {metrics["max_full_f1"]["score"]}', file=f)
    #    print(f'Best CARD f1: {metrics["max_card_f1"]["path"]} - {metrics["max_card_f1"]["score"]}', file=f)
