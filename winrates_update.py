import json
import pandas as pd
from tqdm import tqdm
from utils import read_heroes

MIN_MATCHUPS = 5


def get_matches_hero_appears(df, hero):
    """Возвращает матчи, в которых участвует указанный герой."""
    condition = df["dire_heroes"].apply(lambda x: hero in x) | df["radiant_heroes"].apply(lambda x: hero in x)
    return df[condition]


def get_matches_hero_with_hero(df, hero_1, hero_2):
    """Возвращает матчи, где два героя играют в одной команде."""
    condition = df["dire_heroes"].apply(lambda x: hero_1 in x and hero_2 in x) | df["radiant_heroes"].apply(lambda x: hero_1 in x and hero_2 in x)
    return df[condition]


def get_matches_hero_against_hero(df, hero_1, hero_2):
    """Возвращает матчи, где два героя играют в противоположных командах."""
    condition = (
        (df["dire_heroes"].apply(lambda x: hero_1 in x and hero_2 not in x) & df["radiant_heroes"].apply(lambda x: hero_2 in x and hero_1 not in x)) |
        (df["dire_heroes"].apply(lambda x: hero_2 in x and hero_1 not in x) & df["radiant_heroes"].apply(lambda x: hero_1 in x and hero_2 not in x))
    )
    return df[condition]


def calculate_winrate(wins, total, k=1):
    """Расчёт сглаженного винрейта с использованием сглаживания Лапласа."""
    return (wins + k) / (total + 2 * k)


def weighted_winrate(pair_winrate, total_matches, m=10, global_winrate=0.5):
    """Расчёт взвешенного винрейта, комбинируя винрейт пары и общий винрейт."""
    return (total_matches * pair_winrate + m * global_winrate) / (total_matches + m)


def get_hero_overall_winrates(df):
    """Возвращает общий винрейт для каждого героя."""
    heroes = read_heroes()
    overall_winrates = {}
    for hero in heroes:
        hero_matches = get_matches_hero_appears(df, hero)
        hero_wins = 0
        hero_total = len(hero_matches)

        if hero_total > 0:
            # Победы героя в команде Dire
            condition_dire = hero_matches["dire_heroes"].apply(lambda x: hero in x)
            hero_wins += hero_matches["dire_win"][condition_dire].sum()

            # Победы героя в команде Radiant
            condition_radiant = hero_matches["radiant_heroes"].apply(lambda x: hero in x)
            hero_wins += hero_matches["radiant_win"][condition_radiant].sum()

            overall_winrate = hero_wins / hero_total
        else:
            overall_winrate = 0.5  # Если герой не участвовал в матчах

        overall_winrates[hero] = overall_winrate
    return overall_winrates


def get_hero_stat(df, hero_to_calculate, hero_to_filter=None, against=True, k=1, m=10, overall_winrates=None):
    """Возвращает статистику для конкретного героя с учётом сглаживания и взвешивания."""
    if overall_winrates is None:
        overall_winrates = {}

    hero_wins = 0
    hero_total = 0

    # Фильтрация матчей
    if hero_to_filter is not None:
        if against:
            df_filtered = get_matches_hero_against_hero(df, hero_to_calculate, hero_to_filter)
        else:
            df_filtered = get_matches_hero_with_hero(df, hero_to_calculate, hero_to_filter)
    else:
        df_filtered = get_matches_hero_appears(df, hero_to_calculate)

    # Подсчёт побед и общего количества матчей
    condition_dire = df_filtered["dire_heroes"].apply(lambda x: hero_to_calculate in x)
    condition_radiant = df_filtered["radiant_heroes"].apply(lambda x: hero_to_calculate in x)

    hero_wins += df_filtered["dire_win"][condition_dire].sum()
    hero_wins += df_filtered["radiant_win"][condition_radiant].sum()

    hero_total += condition_dire.sum() + condition_radiant.sum()

    # Расчёт сглаженного винрейта пары
    pair_winrate = calculate_winrate(hero_wins, hero_total, k)

    # Получение общего винрейта героя
    hero_overall_winrate = overall_winrates.get(hero_to_calculate, 0.5)

    # Расчёт взвешенного винрейта
    winrate = weighted_winrate(pair_winrate, hero_total, m, hero_overall_winrate)

    return {
        "total": hero_total,
        "wins": hero_wins,
        "loses": hero_total - hero_wins,
        "winrate": round(winrate, 2),
    }


def get_full_hero_stat(df, hero_to_calculate, overall_winrates, k=1, m=10):
    """Возвращает полную статистику героя против всех других героев."""
    result = {}
    heroes = read_heroes()
    for hero in heroes:
        if hero == hero_to_calculate:
            # Для самого себя используем общий винрейт
            result[hero] = {
                "against_winrate": round(1 - overall_winrates.get(hero_to_calculate, 0.5), 2),
                "with_winrate": round(overall_winrates.get(hero_to_calculate, 0.5), 2)
            }
        else:
            # Статистика против другого героя
            stat_against = get_hero_stat(
                df, hero_to_calculate, hero_to_filter=hero, against=True, k=k, m=m, overall_winrates=overall_winrates
            )
            # Статистика с другим героем в команде
            stat_with = get_hero_stat(
                df, hero_to_calculate, hero_to_filter=hero, against=False, k=k, m=m, overall_winrates=overall_winrates
            )
            result[hero] = {
                "against_winrate": stat_against["winrate"],
                "with_winrate": stat_with["winrate"]
            }
    return result


def get_updated_winrates_dict(df):
    """Обновляет словарь винрейтов для всех героев."""
    result = {}
    heroes = read_heroes()
    overall_winrates = get_hero_overall_winrates(df)
    for hero in tqdm(heroes):
        result[hero] = get_full_hero_stat(df, hero, overall_winrates)
    return result


def update_winrates(data_file_path, winrates_file_name):
    matches = pd.read_pickle(data_file_path)
    winrates_dict = get_updated_winrates_dict(matches)

    with open(winrates_file_name, "w") as outfile:
        json.dump(winrates_dict, outfile)

# update_winrates("C:\\Users\\Ridz\\Desktop\\Matches\\major_matches.pkl", "data/winrates_major_v2.json")


update_winrates("data/matches_v4.pkl", "data/winrates.json")
