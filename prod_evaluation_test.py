import numpy as np
import pandas as pd
import logging
import re
from autogluon.tabular import TabularPredictor
from evaluate import evaluate_tournament


def frame_text(text_lines):
    max_length = max(len(line) for line in text_lines)
    width = max_length + 4
    border = '+' + '-' * (width - 2) + '+'
    framed_text = [border]
    for line in text_lines:
        framed_line = '| ' + line.ljust(max_length) + ' |'
        framed_text.append(framed_line)
    framed_text.append(border)
    return '\n'.join(framed_text)


def print_stat(tournament, total_bank, method='', bet_amount=100, logger=None):

    lines = [f"Prediction Method: {method}", f"---", f"Number of Matches: {len(tournament)}",
             f"Bet Amount: {bet_amount}", f"Total Profit: {total_bank:.2f}",
             f"Accuracy: {np.mean(tournament['is_correct']):.2%}",
             f"Correct Predictions: {sum(tournament['is_correct'])}",
             f"Incorrect Predictions: {len(tournament) - sum(tournament['is_correct'])}", f"---"]

    framed_text = frame_text(lines)
    logger.info(framed_text)


if __name__ == "__main__":

    models = [
        (TabularPredictor.load('models/attributes_dire_170_10_2019_2024', require_version_match=False),
         'ExtraTreesEntr_FULL', 'attributes', False),

        (TabularPredictor.load('models/attributes_radiant_170_10_2019_2024', require_version_match=False),
         'LightGBMXT', 'attributes', True),

        (TabularPredictor.load('models/attributes_radiant_170_10_2021_2024', require_version_match=False),
         'LightGBM_r143', 'attributes', True),

        (TabularPredictor.load('models/heroes_only_dire_170_10_2019_2024', require_version_match=False),
         'CatBoost_r69', 'heroes', False),

        (TabularPredictor.load('models/heroes_only_dire_170_10_2021_2024', require_version_match=False),
         'CatBoost_r167', 'heroes', False),

        (TabularPredictor.load('models/heroes_only_radiant_170_10_2019_2024', require_version_match=False),
         'CatBoost_r50_FULL', 'heroes', True),

        (TabularPredictor.load('models/heroes_only_radiant_170_10_2021_2024', require_version_match=False),
         'LightGBM_r131', 'heroes', True),

        (TabularPredictor.load('models/onehot_dire_170_10_2019_2024', require_version_match=False),
         'NeuralNetTorch_r185_FULL', 'onehot', False),

        (TabularPredictor.load('models/onehot_radiant_170_10_2021_2024', require_version_match=False),
         'XGBoost_r98_FULL', 'onehot', True)
    ]


    datasets = {
        'TI13': pd.read_pickle('data/datasets/evaluation_datasets/ti13.pkl'),
        'PGL Valhalla': pd.read_pickle('data/datasets/evaluation_datasets/pgl_valhalla.pkl'),
        'Validation Dataset': pd.read_pickle('data/datasets/evaluation_datasets/validation_dataset.pkl'),
        'BB Dacha': pd.read_pickle('data/datasets/evaluation_datasets/bb_dacha.pkl'),
        'Dreamleague': pd.read_pickle('data/datasets/evaluation_datasets/dreamleague_group_stage.pkl')
    }

    thresholds = list(range(5, 10))

    for dataset_name, tournament in datasets.items():
        safe_dataset_name = re.sub(r'[^\w\-_\. ]', '_', dataset_name)
        log_filename = f"{safe_dataset_name}_assemble_evaluation_log.txt"

        logger = logging.getLogger(safe_dataset_name)
        logger.setLevel(logging.INFO)


        if not logger.handlers:

            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.INFO)

            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)

            logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)


        header_lines = [f"Evaluating dataset: {dataset_name}"]
        header_frame = frame_text(header_lines)
        logger.info(header_frame)

        teams = set(tournament.dire_team) | set(tournament.radiant_team)

        teams_to_exclude = []
        teams_to_include = list(teams)

        for threshold in thresholds:
            threshold_lines = [f"Evaluating threshold: {threshold}"]
            threshold_frame = frame_text(threshold_lines)
            logger.info(threshold_frame)

            df, bank = evaluate_tournament(
                tournament,
                models=models,
                method='votes',
                threshold=threshold,
                teams_to_exclude=teams_to_exclude,
                teams_to_include=teams_to_include
            )
            method_name = f"{dataset_name}_threshold_{threshold}"
            print_stat(df, bank, method=method_name, logger=logger)

        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
