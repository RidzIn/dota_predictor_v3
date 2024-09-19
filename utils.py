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


def get_onehot(pick):
    one_hot_encoded = np.zeros(139, dtype=int)

    for pos, hero in enumerate(pick):
        if 'Outworld Devourer' == hero:
            hero = 'Outworld Destroyer'
        encode_hero = id_heroes_names[hero]

        one_hot_encoded[int(encode_hero) - 1] = pos+1
    return one_hot_encoded


def features_dataset_onehot(df):
    X = []
    for i in range(len(df)):

        combined_array = np.concatenate((get_onehot(df.iloc[i]['dire_heroes']), get_onehot(df.iloc[i]['radiant_heroes']), [int(df.iloc[i]['dire_win'])]))
        X.append(combined_array)

    return pd.DataFrame(X)


def features_dataset_onehot_radiant_first(df):
    X = []
    for i in range(len(df)):

        combined_array = np.concatenate((get_onehot(df.iloc[i]['radiant_heroes']), get_onehot(df.iloc[i]['dire_heroes']), [int(df.iloc[i]['radiant_win'])]))
        X.append(combined_array)

    return pd.DataFrame(X)


def features_onehot(dire_pick, radiant_pick):
    X = []

    combined_array = np.concatenate((get_onehot(dire_pick), get_onehot(radiant_pick), [0]))
    X.append(combined_array)

    df = pd.DataFrame(X)
    df.columns = df.columns.map(str)
    return df


def features_onehot_radiant_first(dire_pick, radiant_pick):
    X = []

    combined_array = np.concatenate((get_onehot(radiant_pick), get_onehot(dire_pick), [0]))
    X.append(combined_array)

    df = pd.DataFrame(X)
    df.columns = df.columns.map(str)
    return df



