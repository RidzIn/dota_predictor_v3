import pandas as pd
import os
from utils import features_dataset_encoded
from autogluon.tabular import TabularPredictor


def train_model(train_path, train_dataset_path, label_column, radiant_first, method):

    # data/datasets/attributes_dire_160_20_2021_2024.pkl
    train_data = pd.read_pickle(train_path)

    # data/datasets/major_matches_170_10_2021_2024.pkl


    features_train = features_dataset_encoded(train_data, radiant_first=radiant_first, method=method)

    model_output_dir = os.path.join("models_v3", f"{os.path.basename(train_dataset_path)[:-4]}")

    if method == 'onehot':
        predictor = (TabularPredictor(label=label_column, eval_metric='accuracy', path=model_output_dir)
        .fit(
            train_data=features_train,
            num_gpus=1,
            presets=['high_quality'],
            num_stack_levels=0,
            num_bag_folds=0
        ))
    else:
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


# # Dire Heroes Only
# train_model('data/datasets/major_matches_170_10_2019_2024.pkl', 'data/datasets/heroes_only_dire_170_10_2021_2024.pkl', label_column=10, radiant_first=False, method='heroes')
# train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 'data/datasets/heroes_only_dire_170_10_2019_2024.pkl', label_column=10, radiant_first=False, method='heroes')
#
# train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 'data/datasets/heroes_only_dire_160_20_2019_2024.pkl', label_column=10, radiant_first=False, method='heroes')
# train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 'data/datasets/heroes_only_dire_160_20_2021_2024.pkl', label_column=10, radiant_first=False, method='heroes')
#
#
# # Radiant Heroes Only
# # Dire Heroes Only
# train_model('data/datasets/major_matches_170_10_2019_2024.pkl', 'data/datasets/heroes_only_radiant_170_10_2021_2024.pkl', label_column=10, radiant_first=True, method='heroes')
# train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 'data/datasets/heroes_only_radiant_170_10_2019_2024.pkl', label_column=10, radiant_first=True, method='heroes')
#
# train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 'data/datasets/heroes_only_radiant_160_20_2019_2024.pkl', label_column=10, radiant_first=True, method='heroes')
# train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 'data/datasets/heroes_only_radiant_160_20_2021_2024.pkl', label_column=10, radiant_first=True, method='heroes')
#
#
#
# # Dire Attributes
# train_model('data/datasets/major_matches_170_10_2019_2024.pkl', 'data/datasets/attributes_dire_170_10_2021_2024.pkl', label_column=100, radiant_first=False, method='attributes')
# train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 'data/datasets/attributes_dire_170_10_2019_2024.pkl', label_column=100, radiant_first=False, method='attributes')
#
# train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 'data/datasets/attributes_dire_160_20_2019_2024.pkl', label_column=100, radiant_first=False, method='attributes')
# train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 'data/datasets/attributes_dire_160_20_2021_2024.pkl', label_column=100, radiant_first=False, method='attributes')
#
#
# # Radiant Attributes
# train_model('data/datasets/major_matches_170_10_2019_2024.pkl',  'data/datasets/attributes_radiant_170_10_2021_2024.pkl', label_column=100, radiant_first=True, method='attributes')
# train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 'data/datasets/attributes_radiant_170_10_2019_2024.pkl', label_column=100, radiant_first=True, method='attributes')
#
# train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 'data/datasets/attributes_radiant_160_20_2019_2024.pkl', label_column=100, radiant_first=True, method='attributes')
# train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 'data/datasets/attributes_radiant_160_20_2021_2024.pkl', label_column=100, radiant_first=True, method='attributes')
#


# Dire OneHot
train_model('data/datasets/major_matches_170_10_2019_2024.pkl', 'data/datasets/onehot_dire_170_10_2021_2024.pkl', label_column=278, radiant_first=False, method='onehot')
train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 'data/datasets/onehot_dire_170_10_2019_2024.pkl', label_column=278, radiant_first=False, method='onehot')

train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 'data/datasets/onehot_dire_160_20_2019_2024.pkl', label_column=278, radiant_first=False, method='onehot')
train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 'data/datasets/onehot_dire_160_20_2021_2024.pkl', label_column=278, radiant_first=False, method='onehot')


# Radiant OneHot
train_model('data/datasets/major_matches_170_10_2019_2024.pkl', 'data/datasets/onehot_radiant_170_10_2021_2024.pkl', label_column=278, radiant_first=True, method='onehot')
train_model('data/datasets/major_matches_170_10_2021_2024.pkl', 'data/datasets/onehot_radiant_170_10_2019_2024.pkl', label_column=278, radiant_first=True, method='onehot')

train_model('data/datasets/major_matches_160_20_2019_2024.pkl', 'data/datasets/onehot_radiant_160_20_2019_2024.pkl', label_column=278, radiant_first=True, method='onehot')
train_model('data/datasets/major_matches_160_20_2021_2024.pkl', 'data/datasets/onehot_radiant_160_20_2021_2024.pkl', label_column=278, radiant_first=True, method='onehot')

