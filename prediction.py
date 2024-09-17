import urllib

from bs4 import BeautifulSoup

from utils import get_hero_matchups, features_winrates, features_onehot

from autogluon.tabular import TabularPredictor

predictor_onehot = TabularPredictor.load('AutogluonModels/nn_models', require_version_match=False)

# predictor_onehot = TabularPredictor.load('AutogluonModels/nn_models_v2', require_version_match=False)

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
    return predictor_onehot.predict_proba(features_df, model='NeuralNetTorch_BAG_L2\\17a72_00004')
    # return predictor_onehot.predict_proba(features_df, model='NeuralNetTorch_BAG_L1\\5a9b23de')


def get_hero_stats(dire_pick, radiant_pick):
    return features_winrates(dire_pick, radiant_pick)


def parse_match_info_for_gui(link, map_number):
    result = []

    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(link, headers=headers)


    with urllib.request.urlopen(req) as response:
        data = response.read()
        soup = BeautifulSoup(data, 'html.parser')

    maps = soup.find_all('div', class_='map__finished-v2')

    map_div = maps[map_number - 1]

    map_info = {'map_number': map_div.find('span').text.strip()}

    teams = map_div.find_all('div', class_='team')
    heroes_sections = map_div.find_all('div', class_='heroes')

    for i, team in enumerate(teams):
        side = team.find('span', class_='side').text.strip().lower()
        team_name = team.find('span', class_='name').text.strip()

        heroes = []
        hero_picks = heroes_sections[i].find_all('div', class_='pick')
        for hero in hero_picks:
            heroes.append(hero['data-tippy-content'])

        if side == 'dire':
            map_info['dire_team'] = team_name
            map_info['dire_heroes'] = heroes
        elif side == 'radiant':
            map_info['radiant_team'] = team_name
            map_info['radiant_heroes'] = heroes

    if len(map_info.get('dire_heroes', [])) == 5 and len(map_info.get('radiant_heroes', [])) == 5:
        result.append(map_info)

    return result[0]


