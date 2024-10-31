from bs4 import BeautifulSoup
import os
import pandas as pd
from tqdm import tqdm


def filter_match(soup, odds_threshold=1.65, prob_threshold=10):
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

    # If no users votes on the match we find odds
    if len(percentages) < 2:
        odds = parse_odds(soup)

        # Odds were not parsed
        if odds['team1_odds'] == 0:
            return False, None

        team1_odds = odds['team1_odds']
        team2_odds = odds['team2_odds']

        # Check if the teams are equal
        if odds_threshold < team1_odds and odds_threshold < team2_odds:
            return True, None

        # Find the outsider team
        if odds_threshold >= team1_odds:
            outsider = 'team_2'
        if odds_threshold >= team2_odds:
            outsider = 'team_1'

        result = {
            "team_1": team1_name,
            "team_2": team2_name,
            "team_1_score": team1_score,
            "team_2_score": team2_score,
            "team_1_percentage": team1_odds,
            "team_2_percentage": team2_odds,
            'percentage_diff': round(team1_odds - team2_odds, 2)
        }

        if result[f'{outsider}_score'] > 0:
            return True, result[outsider]

        return False, None

    # If user votes are available on the page
    else:
        team1_percentage = float(percentages[0].get_text().strip()[:-1])
        team2_percentage = float(percentages[1].get_text().strip()[:-1])

        # In BO2 matches there are 3 posible outcomes, so I need to find outcome for 2:0 for second team
        if team2_percentage + team1_percentage < 99:
            team2_percentage = 100 - (team1_percentage + team2_percentage)

        result = {
            "team_1": team1_name,
            "team_2": team2_name,
            "team_1_score": team1_score,
            "team_2_score": team2_score,
            "team_1_percentage": team1_percentage,
            "team_2_percentage": team2_percentage,
            'percentage_diff': round(team1_percentage - team2_percentage, 2)
        }

        if abs(result['team_1_percentage'] - result['team_2_percentage']) < prob_threshold:
            return True, None

        outsider = 'team_1' if result['team_1_percentage'] < result['team_2_percentage'] else 'team_2'
        if result[f'{outsider}_score'] > 0:
            return True, result[outsider]

        return False, None



def parse_odds(soup):
    result = []

    rows = soup.find_all('a', class_='table__body-row')
    for row in rows:
        map_info = {}

        odds_cell = row.find('div', class_='table__body-row__cell width-50 align-right')
        if odds_cell:
            cell_div = odds_cell.find('div', class_='cell')
            if cell_div:
                bookmakers_items = cell_div.find_all('div', class_='bookmakers__item')
                for item in bookmakers_items:
                    title_div = item.find('div', class_='bookmakers__item-title')
                    if title_div and title_div.text.strip() == 'Map 1':
                        bets = item.find_all('span', class_='bookmakers__item-bet')
                        if len(bets) == 2:
                            map_info['team1_odds'] = float(bets[0].text.strip())
                            map_info['team2_odds'] = float(bets[1].text.strip())
                            result.append(map_info)
                            break
    if len(result) == 0:
        return {'team1_odds': 0.0, 'team2_odds': 0.0}
    return result[0]


def parse_match_info(soup, outsider_name=None):
    result = []

    odds_dict = parse_odds(soup)

    maps = soup.find_all('div', class_='map__finished-v2')
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
                if i == 0:
                    map_info['dire_odds'] = odds_dict['team1_odds']
                else:
                    map_info['dire_odds'] = odds_dict['team2_odds']
            elif side == 'radiant':
                map_info['radiant_team'] = team_name
                map_info['radiant_heroes'] = heroes
                map_info['radiant_win'] = win_status
                if i == 0:
                    map_info['radiant_odds'] = odds_dict['team1_odds']
                else:
                    map_info['radiant_odds'] = odds_dict['team2_odds']

        if map_info.get('dire_win', False):
            map_info['winner'] = map_info['dire_team']
        elif map_info.get('radiant_win', False):
            map_info['winner'] = map_info['radiant_team']
        else:
            map_info['winner'] = 'Unknown'


        if len(map_info.get('dire_heroes', [])) == 5 and len(map_info.get('radiant_heroes', [])) == 5:
            if outsider_name is None:
                result.append(map_info)

            # sometimes odds are messed up, I switch them
            elif outsider_name and map_info['winner'] == outsider_name:
                if outsider_name == map_info['dire_team'] and map_info['dire_odds'] < map_info['radiant_odds']:
                    temp = map_info['dire_odds']
                    map_info['dire_odds'] = map_info['radiant_odds']
                    map_info['radiant_odds'] = temp
                if outsider_name == map_info['radiant_team'] and map_info['radiant_odds'] < map_info['dire_odds']:
                    temp = map_info['dire_odds']
                    map_info['dire_odds'] = map_info['radiant_odds']
                    map_info['radiant_odds'] = temp
                result.append(map_info)
    return result


def parse_matches(match_dir, file_name="matches", filtered=True, odds_threshold=1.7, prob_threshold=10):
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

            match_passed_the_filter, outsider_team = filter_match(soup, odds_threshold=odds_threshold, prob_threshold=prob_threshold)
            if match_passed_the_filter and filtered:
                # We take only outsides won maps
                match_info = parse_match_info(soup, outsider_team)
                for map in match_info:
                    matches_info.append(map)
            elif not filtered:
                # If not filter takes every map from the match
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


parse_matches('matches', 'dreamleague_group_stage', filtered=False, odds_threshold=1.7, prob_threshold=10)

# parse_matches('C:\\Users\\Ridz\\Desktop\\major_2021_2024', 'major_matches_185_2_2021_2024', filtered=True, odds_threshold=1.85, prob_threshold=2)
# parse_matches('C:\\Users\\Ridz\\Desktop\\major_2021_2024', 'bad_major_matches_170_10_2021_2024', filtered=True, odds_threshold=1.70, prob_threshold=10)
# parse_matches('C:\\Users\\Ridz\\Desktop\\major_2021_2024', 'major_matches_160_20_2021_2024', filtered=True, odds_threshold=1.6, prob_threshold=20)

# parse_matches('C:\\Users\\Ridz\\Desktop\\matches_2023_2024', 'all_matches_185_2_2023_2024', filtered=True, odds_threshold=1.85, prob_threshold=2)
# parse_matches('C:\\Users\\Ridz\\Desktop\\matches_2023_2024', 'all_matches_170_10_2023_2024', filtered=True, odds_threshold=1.70, prob_threshold=10)
# parse_matches('C:\\Users\\Ridz\\Desktop\\matches_2023_2024', 'all_matches_160_20_2023_2024', filtered=True, odds_threshold=1.6, prob_threshold=20)

# parse_matches('C:\\Users\\Ridz\\Desktop\\major_all', 'major_matches_185_2_2019_2024', filtered=True, odds_threshold=1.85, prob_threshold=2)
# parse_matches('C:\\Users\\Ridz\\Desktop\\major_all', 'temp_2', filtered=True, odds_threshold=1.70, prob_threshold=10)
# parse_matches('C:\\Users\\Ridz\\Desktop\\major_all', 'major_matches_160_20_2019_2024', filtered=True, odds_threshold=1.6, prob_threshold=20)
