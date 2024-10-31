import json
import pandas as pd

with open("data/util/heroes_decoder.json") as file:
    heroes_id_names = json.load(file)


with open('data/util/heroes_encoder.json') as file:
    id_heroes_names = json.load(file)


with open('data/util/heroes_attributes.json') as file:
    heroes_attributes = json.load(file)


with open('data/winrates.json') as file:
    winrates_dict = json.load(file)


def read_heroes(file_name="data/util/heroes.txt"):
    """
    Take txt file of heroes and return set object
    """
    hero_set = set()
    with open(file_name, "r") as file:
        for line in file:
            hero_set.add(line.strip())
    return hero_set


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
        for h2 in pick_copy:
            synergy_features.append(winrates[h1][h2]["with_winrate"])
    return synergy_features


def get_duel_features(winrates: dict, dire_pick: list, radiant_pick: list) -> tuple:
    dire_duel_features, radiant_duel_features = [], []
    for pos, h1 in enumerate(dire_pick):

        for h2 in radiant_pick:
            against_winrate = winrates[h1][h2].get("against_winrate", 0)
            dire_duel_features.append(against_winrate)
            radiant_duel_features.append(1 - against_winrate)

        # dire_duel_features.append(temp_dire_feature/5)
        # radiant_duel_features.append(temp_radiant_feature / 5)

    return dire_duel_features, radiant_duel_features


def pick_attribute_heroes_feature(pick):
    combined_array = []
    for h in pick:
        if 'Outworld Devourer' == h:
            h = 'Outworld Destroyer'
        combined_array.append(id_heroes_names[h])

        for w in list(heroes_attributes[h].values())[:-1]:
            combined_array.append(w)
    return combined_array


def pick_only_attributes_feature(pick):
    combined_array = []
    for h in pick:
        for w in list(heroes_attributes[h].values())[:-1]:
            combined_array.append(w)

    return combined_array


def pick_only_heroes_feature(pick):
    combined_array = []
    for h in pick:
        if 'Outworld Devourer' == h:
            h = 'Outworld Destroyer'
        combined_array.append(id_heroes_names[h])

    return combined_array


def pick_onehot_feature(pick):
    one_hot_array = [0] * 139

    for h in pick:
        if h == 'Outworld Devourer':
            h = 'Outworld Destroyer'

        hero_index = id_heroes_names[h]

        one_hot_array[int(hero_index)] = 1

    return one_hot_array


def features_encoded(dire_pick, radiant_pick, method, radiant_first=False):
    if method == 'heroes':
        dire = pick_only_heroes_feature(dire_pick)
        radiant = pick_only_heroes_feature(radiant_pick)
    if method == 'attributes':
        dire = pick_attribute_heroes_feature(dire_pick)
        radiant = pick_attribute_heroes_feature(radiant_pick)
    if method == 'onehot':
        dire = pick_onehot_feature(dire_pick)
        radiant = pick_onehot_feature(radiant_pick)

    if radiant_first:
        return pd.DataFrame([radiant + dire])
    else:
        return pd.DataFrame([dire + radiant])



def features_dataset_encoded(df, method, radiant_first=False):
    X = []
    for i in range(len(df)):
        if method == 'heroes':
            dire = pick_only_heroes_feature(df.iloc[i]['dire_heroes'])
            radiant = pick_only_heroes_feature(df.iloc[i]['radiant_heroes'])
        if method == 'attributes':
            dire = pick_attribute_heroes_feature(df.iloc[i]['dire_heroes'])
            radiant = pick_attribute_heroes_feature(df.iloc[i]['radiant_heroes'])

        if method == 'onehot':
            dire = pick_onehot_feature(df.iloc[i]['dire_heroes'])
            radiant = pick_onehot_feature(df.iloc[i]['radiant_heroes'])

        if radiant_first:
            combined_array = radiant + dire
        else:
            combined_array = dire + radiant

        combined_array.append(df.iloc[i]['dire_win'])
        X.append(combined_array)

    return pd.DataFrame(X)


df = pd.read_pickle('data/datasets/evaluation_datasets/ti13.pkl')

# print(features_dataset_encoded(df, 'onehot', radiant_first=True))
# print(len(features_winrates(['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink'])))
# print(features_encoded(['Riki', 'Anti-Mage', 'Slark', 'Mirana', 'Hoodwink'], ['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink'], 'onehot'))


