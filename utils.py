import json
import pandas as pd


with open("data/util/heroes_decoder.json") as file:
    heroes_id_names = json.load(file)


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


def features_dataset_encoded(df, method, radiant_first=False):
    X = []
    for i in range(len(df)):
        if method == 'heroes':
            dire = pick_feature_v3(df.iloc[i]['dire_heroes'])
            radiant = pick_feature_v3(df.iloc[i]['radiant_heroes'])
        if method == 'attributes':
            dire = pick_feature(df.iloc[i]['dire_heroes'])
            radiant = pick_feature(df.iloc[i]['radiant_heroes'])

        if radiant_first:
            combined_array = radiant + dire
        else:
            combined_array = dire + radiant

        combined_array.append(df.iloc[i]['dire_win'])
        X.append(combined_array)

    return pd.DataFrame(X)


def features_encoded(dire_pick, radiant_pick, method, radiant_first=False):
    if method == 'heroes':
        dire = pick_feature_v3(dire_pick)
        radiant = pick_feature_v3(radiant_pick)
    if method == 'attributes':
        dire = pick_feature(dire_pick)
        radiant = pick_feature(radiant_pick)

    if radiant_first:
        return pd.DataFrame([radiant + dire])
    else:
        return pd.DataFrame([dire + radiant])


# print(pick_feature(['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink']))
# print(features_encoded(['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink'], ['Riki', 'Mars', 'Slark', 'Mirana', 'Hoodwink']))


