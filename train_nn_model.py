import pandas as pd
import os
from utils import features_dataset_encoded
from autogluon.tabular import TabularPredictor


def train_model(train_dataset_path, label_column):
    train_data = pd.read_pickle(train_dataset_path)

    features_train = features_dataset_encoded(train_data, radiant_first=True)

    model_output_dir = os.path.join("models_v2", f"{os.path.basename(train_dataset_path)[:-4]}")

    predictor = (TabularPredictor(label=label_column, eval_metric='accuracy', path=model_output_dir)
    .fit(
        train_data=features_train,
        num_gpus=1,
        excluded_model_types={'FASTAI', 'NN_TORCH'},
        presets=['high_quality'],
        num_stack_levels=0,
        num_bag_folds=0
    ))
    return predictor


train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 10)
train_model('data/datasets/major_matches_170_10_2019_2024.pkl', 10)

# train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 10)
# train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 10)
#
# train_model('data/datasets/major_matches_185_2_2019_2024.pkl', 100)
# train_model('data/datasets/major_matches_185_2_2021_2024.pkl', 100)
#
# train_model('data/datasets/all_matches_160_20_2023_2024.pkl', 100)
# train_model('data/datasets/all_matches_170_10_2023_2024.pkl', 100)
# train_model('data/datasets/all_matches_185_2_2023_2024.pkl.pkl', 100)
