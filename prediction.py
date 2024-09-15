from utils import get_hero_matchups, features_winrates, features_onehot

from autogluon.tabular import TabularPredictor

predictor_onehot = TabularPredictor.load('AutogluonModels/nn_models', require_version_match=False)


# WE PREDICT DIRE TEAM PROBABILITY TO WIN


def get_meta_prediction(dire_pick, radiant_pick):
    """Parse data from OpenDota API and calculate win probability based on recent matches played on this
    heroes by non-professional players"""
    avg_winrates = {}
    for hero in dire_pick:
        temp_df = get_hero_matchups(hero, radiant_pick)
        avg_winrates[hero] = temp_df["winrate"].sum() / 5
    dire_win_prob = round(sum(avg_winrates.values()) / 5, 3)
    return {"dire": dire_win_prob, "radiant": 1 - dire_win_prob}


def get_onehot_prediction(dire_pick, radiant_pick):
    features_df = features_onehot(dire_pick, radiant_pick)
    return predictor_onehot.predict_proba(features_df, model='NeuralNetTorch_BAG_L2/17a72_00004')


def get_hero_stats(dire_pick, radiant_pick):
    return features_winrates(dire_pick, radiant_pick)
