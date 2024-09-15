from bs4 import BeautifulSoup
import os
import pandas as pd
from tqdm import tqdm


def filter_match(soup):
    teams = soup.find_all('a', class_='team__stats-name')
    if len(teams) < 2:
        return False, None

    team1_name = teams[0].get_text().strip()
    team2_name = teams[1].get_text().strip()

    scores = soup.find('div', class_='score__scores')
    if not scores or len(scores.find_all('span')) < 2:
        return False, None

    team1_score = int(scores.find_all('span')[0].get_text().strip())
    team2_score = int(scores.find_all('span')[1].get_text().strip())

    percentages = soup.find_all('div', class_='percent')
    if len(percentages) < 2:
        return False, None

    team1_percentage = float(percentages[0].get_text().strip()[:-1])
    team2_percentage = float(percentages[1].get_text().strip()[:-1])

    result = {
        "team_1": team1_name,
        "team_2": team2_name,
        "team_1_score": team1_score,
        "team_2_score": team2_score,
        "team_1_percentage": team1_percentage,
        "team_2_percentage": team2_percentage,
        'percentage_diff': round(team1_percentage - team2_percentage, 2)
    }

    if abs(result['team_1_percentage'] - result['team_2_percentage']) < 20:
        return True, None

    outsider = 'team_1' if result['team_1_percentage'] < result['team_2_percentage'] else 'team_2'
    if result[f'{outsider}_score'] > 0:
        print(f"outsider won: {outsider} maps")
        return True, result[outsider]

    return False, None


def parse_match_info(soup, outsider_name=None):
    maps = soup.find_all('div', class_='map__finished-v2')
    result = []

    for map_div in maps:
        map_info = {'map_number': map_div.find('span').text.strip()}

        match_id = map_div.find('small').text.strip().replace('Match ID: ', '')
        map_info['match_id'] = match_id

        teams = map_div.find_all('div', class_='team')

        heroes_sections = map_div.find_all('div', class_='heroes')

        for i, team in enumerate(teams):
            side = team.find('span', class_='side').text.strip().lower()
            team_name = team.find('span', class_='name').text.strip()

            heroes = []
            hero_picks = heroes_sections[i].find_all('div', class_='pick')
            for hero in hero_picks:
                heroes.append(hero['data-tippy-content'])

            win_status = bool(team.find('div', class_='winner'))

            if side == 'dire':
                map_info['dire_team'] = team_name
                map_info['dire_heroes'] = heroes
                map_info['dire_win'] = win_status
            elif side == 'radiant':
                map_info['radiant_team'] = team_name
                map_info['radiant_heroes'] = heroes
                map_info['radiant_win'] = win_status

        if map_info.get('dire_win', False):
            map_info['winner'] = map_info['dire_team']
        elif map_info.get('radiant_win', False):
            map_info['winner'] = map_info['radiant_team']
        else:
            map_info['winner'] = 'Unknown'

        if len(map_info['dire_heroes']) == 5 and len(map_info['radiant_heroes']) == 5:
            if outsider_name is None:
                result.append(map_info)
            if outsider_name and map_info['winner'] == outsider_name:
                result.append(map_info)
    return result


def parse_matches(match_dir, file_name="matches", filtered=True):
    if not os.path.exists(match_dir):
        raise FileNotFoundError(f"Match directory not found: {match_dir}")

    match_file_list = os.listdir(match_dir)
    matches_info = []

    skips = 0
    for i, match in tqdm(enumerate(match_file_list)):
        try:
            with open(os.path.join(match_dir, match), 'r', encoding='utf-8') as file:
                content = file.read()
            soup = BeautifulSoup(content, 'html.parser')

            match_passed_the_filter, outsider_team = filter_match(soup)

            if match_passed_the_filter:
                match_info = parse_match_info(soup, outsider_team)
                for map in match_info:
                    matches_info.append(map)
            elif not filtered:
                match_info = parse_match_info(soup, None)
                for map in match_info:
                    matches_info.append(map)
            else:
                skips += 1
                print(f"{skips}/{i}")
        except UnicodeDecodeError:
            print(f"Couldn't read the file: {match.upper()}")
        except Exception as e:
            print(f"Error occurred while processing file: {match}")
            print(f"Error message: {str(e)}")
    print("Total skips:", skips)
    return pd.DataFrame(matches_info).to_pickle(f"{file_name}.pkl")


parse_matches('matches', 'ti13_with_filter', filtered=True)
