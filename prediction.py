import urllib
from bs4 import BeautifulSoup
from utils import features_encoded


def get_votes_prediction(dire_pick, radiant_pick, models, dire_team='Dire', radiant_team='Radiant'):
    predictions = []

    for predictor, model_name, method, is_radiant in models:
        pred = get_prediction(dire_pick, radiant_pick, predictor, model_name, is_proba=False, method=method, radiant_first=is_radiant)
        # print(get_prediction(dire_pick, radiant_pick, predictor, model_name, is_proba=True, method=method, radiant_first=is_radiant))
        predictions.append(pred[0])

    dire_votes = sum(predictions)
    radiant_votes = len(predictions) - dire_votes

    if dire_votes >= radiant_votes:
        return 1, dire_pick, dire_team, dire_votes
    else:
        return 0, radiant_pick, radiant_team, radiant_votes


def get_prediction(dire_pick, radiant_pick, predictor, model, method, radiant_first=False, is_proba=True):
    features_df = features_encoded(dire_pick, radiant_pick, radiant_first=radiant_first, method=method)
    pred = predictor.predict_proba(features_df, model=model) if is_proba else predictor.predict(features_df, model=model)
    return pred


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


