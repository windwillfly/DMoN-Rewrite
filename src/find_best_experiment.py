import json
import numpy as np
import os


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_folder', type=str, required=True)
    args = parser.parse_args()

    best_metrics = get_best_metrics(args.experiments_folder)
    print(best_metrics)
