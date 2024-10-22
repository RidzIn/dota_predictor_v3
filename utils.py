import requests
import json
import pandas as pd
import numpy as np


with open("data/util/heroes_decoder.json") as file:
    heroes_id_names = json.load(file)

with open('data/winrates.json') as file:
    winrates_dict = json.load(file)

with open('data/util/heroes_encoder.json') as file:
    id_heroes_names = json.load(file)


with open('data/util/heroes_attributes.json') as file:
    heroes_attributes = json.load(file)


def read_heroes(file_name="data/util/heroes.txt"):
    """
    Take txt file of heroes and return set object
    """
    hero_set = set()
    with open(file_name, "r") as file:
        for line in file:
            hero_set.add(line.strip())
    return hero_set


def get_match_picks(match_id):
    response = requests.get(f'https://api.opendota.com/api/matches/{match_id}')
    if 'Internal Server Error' in response.text:
        raise ValueError("Open Dota crushed, refresh the page and try Test Yourself page")
    radiant_picks = [pick["hero_id"] for pick in response.json()['picks_bans'] if pick["is_pick"] and pick["team"] == 0]
    dire_picks = [pick["hero_id"] for pick in response.json()['picks_bans'] if pick["is_pick"] and pick["team"] == 1]

    radiant_picks_decode = []
    dire_picks_decode = []
    for id in radiant_picks:
        for key, value in heroes_id_names.items():
            if int(key) == int(id):
                radiant_picks_decode.append(value)
                break

    for id in dire_picks:
        for key, value in heroes_id_names.items():
            if int(key) == int(id):
                dire_picks_decode.append(value)
                break

    return {"dire": dire_picks_decode, 'radiant': radiant_picks_decode,
            "dire_team": response.json()['dire_team']['name'], 'radiant_team': response.json()['radiant_team']['name']}


def get_hero_matchups(hero_name, pick):
    for key, value in heroes_id_names.items():
        if value == hero_name:
            hero_key = key
            break

    response = requests.get(f"https://api.opendota.com/api/heroes/{hero_key}/matchups")

    data = json.loads(response.text)
    temp_df = pd.DataFrame(data)
    temp_df["winrate"] = round(temp_df["wins"] / temp_df["games_played"], 2)

    temp_df["name"] = [heroes_id_names[str(i)] for i in temp_df["hero_id"]]
    temp_df = temp_df[temp_df["name"].isin(pick)]
    return temp_df


def get_feature_vec(winrates: dict, dire_pick: list, radiant_pick: list) -> list:
    dire_pick_synergy_features = get_synergy_features(winrates, dire_pick)
    radiant_pick_synergy_features = get_synergy_features(winrates, radiant_pick)
    dire_pick_duel_features, radiant_pick_duel_features = get_duel_features(
        winrates, dire_pick, radiant_pick
    )

    return (
        dire_pick_synergy_features
        + dire_pick_duel_features
        + radiant_pick_synergy_features
    )


def get_synergy_features(winrates: dict, pick: list) -> list:
    pick_copy = pick[::]
    synergy_features = []
    for pos, h1 in enumerate(pick):
        temp_hero_synergy = 0
        for h2 in pick_copy:
            temp_hero_synergy += winrates[h1][h2]["with_winrate"]
        synergy_features.append(pos+1)
        synergy_features.append(temp_hero_synergy / 5)
    return synergy_features


def get_duel_features(winrates: dict, dire_pick: list, radiant_pick: list) -> tuple:
    dire_duel_features, radiant_duel_features = [], []
    for pos, h1 in enumerate(dire_pick):
        temp_dire_feature = 0
        temp_radiant_feature = 0
        for h2 in radiant_pick:
            against_winrate = winrates[h1][h2].get("against_winrate", 0)
            temp_dire_feature += against_winrate
            temp_radiant_feature += (1 - against_winrate)

        dire_duel_features.append(pos+1)
        dire_duel_features.append(temp_dire_feature/5)
        radiant_duel_features.append(pos + 1)
        radiant_duel_features.append(temp_radiant_feature / 5)

    return dire_duel_features, radiant_duel_features


def features_winrates(dire_pick, radiant_pick):
    X = []

    feature_vec = get_feature_vec(winrates_dict, dire_pick, radiant_pick)
    feature_vec_with_label = feature_vec + [0]
    X.append(feature_vec_with_label)

    df = pd.DataFrame(X)
    df.columns = df.columns.map(str)
    return df


def pick_feature(pick):
    combined_array = []
    for h in pick:
        if 'Outworld Devourer' == h:
            h = 'Outworld Destroyer'
        combined_array.append(id_heroes_names[h])

        for w in list(heroes_attributes[h].values())[:-1]:
            combined_array.append(w)
    return combined_array


def pick_feature_v2(pick):
    combined_array = []
    for h in pick:
        for w in list(heroes_attributes[h].values())[:-1]:
            combined_array.append(w)

    return combined_array


def pick_feature_v3(pick):
    combined_array = []
    for h in pick:
        if 'Outworld Devourer' == h:
            h = 'Outworld Destroyer'
        combined_array.append(id_heroes_names[h])

    return combined_array


def features_dataset_encoded(df, radiant_first=False):
    X = []
    for i in range(len(df)):

        dire = pick_feature_v3(df.iloc[i]['dire_heroes'])
        radiant = pick_feature_v3(df.iloc[i]['radiant_heroes'])
        if radiant_first:
            combined_array = radiant + dire
        else:
            combined_array = dire + radiant

        combined_array.append(df.iloc[i]['dire_win'])
        X.append(combined_array)

    return pd.DataFrame(X)


def features_encoded(dire_pick, radiant_pick, radiant_first=False):
    dire = pick_feature_v3(dire_pick)
    radiant = pick_feature_v3(radiant_pick)

    if radiant_first:
        return pd.DataFrame([radiant + dire])
    else:
        return pd.DataFrame([dire + radiant])


# print(pick_feature(['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink']))
# print(features_encoded(['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink'], ['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink']))


