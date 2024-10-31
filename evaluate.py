from copy import copy

from matplotlib import pyplot as plt
from tqdm import tqdm
from prediction import get_prediction, get_votes_prediction
import pandas as pd
import numpy as np


def print_stat(df, method, bet_amount, total_bank):
    right_pred_q_40 = round(df[df['is_correct'] == True]['pred_odd'].quantile(0.4), 2)
    right_pred_q_60 = round(df[df['is_correct'] == True]['pred_odd'].quantile(0.6), 2)

    wrong_pred_q_40 = round(df[df['is_correct'] == False]['pred_odd'].quantile(0.4), 2)
    wrong_pred_q_60 = round(df[df['is_correct'] == False]['pred_odd'].quantile(0.6), 2)


    print(f"Prediction_method: {method}")
    print("-------------------------------")
    print(f"DataFrame length: {len(df)}")
    print(
        f"Right prediction odd: [ {right_pred_q_40} : {right_pred_q_60} ] | Wrong prediction odd: [ {wrong_pred_q_40} : {wrong_pred_q_60} ]")
    print(f"Bet amount: {bet_amount} | Total earning: {total_bank:.2f}")
    print(
        f"Accuracy: {np.mean(df['is_correct']):.2%} | Right: {sum(df['is_correct'])} | Wrong: {len(df) - sum(df['is_correct'])}")
    print("-------------------------------")

    bin_edges = [0, 1.4, 1.7, 2.0, 3.0, 6.0]
    df['odd_range'] = pd.cut(df['pred_odd'], bins=bin_edges, include_lowest=True)

    # Group data by odd ranges and count correct and incorrect predictions
    grouped = df.groupby('odd_range', observed=True)['is_correct'].value_counts().unstack().fillna(0)

    # Create the bar plot
    grouped.plot(kind='bar', stacked=False, color=['red', 'green'], figsize=(12, 6))
    plt.title('Number of Correct and Incorrect Predictions by Quantile Odd Ranges')
    plt.xlabel('Odd Range')
    plt.ylabel('Number of Predictions')
    plt.legend(['Incorrect', 'Correct'])
    plt.show()


def evaluate_combination(df_1, df_2, bet_amount=100):
    combined_df = copy(df_1)
    combined_df['y_pred'] = df_2['y_pred']

    for i in range(len(df_1)):
        if df_1.iloc[i]['y_pred'] != df_2.iloc[i]['y_pred']:
            combined_df.drop(i, inplace=True)

    total_bank = 0
    for i in range(len(combined_df)):
        if combined_df.iloc[i]['y_pred'] == combined_df.iloc[i]['y_true']:
            total_bank += (bet_amount * combined_df.iloc[i]['pred_odd'] - bet_amount)
        else:
            total_bank -= bet_amount

    print_stat(combined_df, 'Combination', bet_amount, total_bank)

    return combined_df


def evaluate_tournament(
    tournament,
    predictor=None,
    model=None,
    models=None,
    threshold=None,
    bet_amount=100,
    only_odds_included=False,
    radiant_first=False,
    method='Unknown',
    teams_to_exclude=None,
    teams_to_include=None,
):
    """
    Evaluate the tournament predictions.

    2 options:
    1. evaluate tournaments on assemble 'votes' method
        evaluate_tournament_v2(dreamleague, models=models, threshold=5, method='votes')
    2. evaluate tournaments on one particular model
        evaluate_tournament_v2(dreamleague, predictor=models[0][0], model=models[0][1], method=models[0][2], radiant_first=models[0][3])


    Parameters:
    - tournament (pd.DataFrame): DataFrame containing tournament data.
    - predictor: The predictor function or model to use (for method 'prediction').
    - model: The model to use for predictions (if applicable).
    - models: List of models to use for voting (if applicable).
    - threshold (float): The threshold for accepting predictions.
    - bet_amount (float): The amount to bet on each match.
    - only_odds_included (bool): Whether to include only matches with odds.
    - radiant_first (bool): Whether the radiant team is first in the prediction output.
    - method (str): The prediction method to use ('votes' 'onehot', 'heroes', 'attributes').

    Returns:
    - pd.DataFrame: DataFrame with match information and prediction results.
    """
    y_true = []
    y_pred = []
    pred_proba = []
    teams = []
    is_correct = []
    match_infos = []
    passed_matches = 0
    predicted_odds = []
    total_bank = 0

    for row in tqdm(tournament.itertuples()):

        if row.dire_team not in teams_to_include:
            if row.radiant_team not in teams_to_include:
                continue

        if row.dire_team in teams_to_exclude or row.radiant_team in teams_to_exclude:
            continue

        # Get prediction based on method
        if method == 'votes':
            # Ensure models are provided
            if models is None:
                raise ValueError("Models must be provided for 'votes' method.")
            predicted_dire_win, _, predicted_team, scores = get_votes_prediction(
                row.dire_heroes, row.radiant_heroes, models=models
            )
            prediction_confidence = scores
            if prediction_confidence < threshold:
                continue
        else:
            # Ensure predictor is provided
            if predictor is None:
                raise ValueError("Predictor must be provided for 'prediction' method.")
            prediction_df = get_prediction(
                row.dire_heroes,
                row.radiant_heroes,
                predictor=predictor,
                model=model,
                radiant_first=radiant_first,
                method=method
            )
            max_prob = prediction_df.values.max()
            prediction_confidence = max_prob
            if prediction_confidence < threshold:
                continue
            predicted_dire_win = bool(prediction_df.values.argmax())
            predicted_team = row.dire_team if predicted_dire_win else row.radiant_team

        # Actual result
        actual_dire_win = row.dire_win
        winning_team = row.dire_team if actual_dire_win else row.radiant_team

        # Betting calculations
        if float(row.dire_odds) > 1:
            if predicted_dire_win == actual_dire_win:
                win_amount = (
                    float(row.dire_odds) if actual_dire_win else float(row.radiant_odds)
                ) * bet_amount - bet_amount
                total_bank += win_amount
            else:
                total_bank -= bet_amount

        if only_odds_included and row.dire_odds == 0.0:
            continue

        predicted_odds.append(
            float(row.dire_odds) if predicted_dire_win else float(row.radiant_odds)
        )

        passed_matches += 1
        pred_proba.append(prediction_confidence)
        y_true.append(winning_team)
        y_pred.append(predicted_team)

        # Match and team info
        match_info = f"|Match ID: {row.match_id} | Map {row.map_number}|"
        match_infos.append(match_info)

        team_info = f"{row.radiant_team} | {row.dire_team}"
        teams.append(team_info)

        # Correct prediction
        is_correct.append(actual_dire_win == predicted_dire_win)

    # Create DataFrame
    df = pd.DataFrame({
        'match_info': match_infos,
        'teams': teams,
        'y_true': y_true,
        'y_pred': y_pred,
        'pred_odd': predicted_odds,
        'is_correct': is_correct,
        'pred_proba': pred_proba
    })

    print_stat(df, method, bet_amount, total_bank)
    return df, total_bank

