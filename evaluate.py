from copy import copy

from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt
from tqdm import tqdm
from prediction import get_prediction
import pandas as pd
import numpy as np


def evaluate_tournament(tournament, predictor, model, threshold=0.50, bet_amount=100, only_odds_included=False, radiant_first=False):
    """
    Evaluate the tournament predictions and calculate accuracy.

    Parameters:
    - tournament (pd.DataFrame): DataFrame containing tournament data.
    - prediction_method (str): The prediction method to use ('NN' or other).

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
        # Get prediction
        if radiant_first:
            prediction_df = get_prediction(row.dire_heroes, row.radiant_heroes, predictor=predictor, model=model, radiant_first=True)
        else:
            prediction_df = get_prediction(row.dire_heroes, row.radiant_heroes, predictor=predictor, model=model)

        # Prediction probability
        max_prob = prediction_df.values.max()
        if max_prob < threshold:
            print('Does not match the threshold')
            continue

        # Actual result
        actual_dire_win = row.dire_win
        winning_team = row.dire_team if actual_dire_win else row.radiant_team

        # Predicted result
        if radiant_first:
            predicted_dire_win = bool(prediction_df.values.argmin())
        else:
            predicted_dire_win = bool(prediction_df.values.argmax())

        if float(row.dire_odds) > 1:
            if predicted_dire_win == actual_dire_win:
                if actual_dire_win == 0:
                    total_bank += (float(row.radiant_odds) * bet_amount - bet_amount)
                else:
                    total_bank += (float(row.dire_odds) * bet_amount - bet_amount)
            else:
                total_bank -= bet_amount

        if only_odds_included:
            if row.dire_odds == 0.0:
                continue
        predicted_odds.append(float(row.dire_odds) if predicted_dire_win else float(row.radiant_odds))

        predicted_team = row.dire_team if predicted_dire_win else row.radiant_team

        passed_matches += 1

        pred_proba.append(max_prob)
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

    temp_pred_odds = []
    for i in predicted_odds:
        if i > 1:
            temp_pred_odds.append(i)

    print_stat(df, 'Radiant first', bet_amount, total_bank)
    return df


def print_stat(df, method, bet_amount, total_bank):
    right_pred_q_40 = round(df[df['is_correct'] == True]['pred_odd'].quantile(0.4), 2)
    right_pred_q_60 = round(df[df['is_correct'] == True]['pred_odd'].quantile(0.6), 2)

    wrong_pred_q_40 = round(df[df['is_correct'] == False]['pred_odd'].quantile(0.4), 2)
    wrong_pred_q_60 = round(df[df['is_correct'] == False]['pred_odd'].quantile(0.6), 2)


    print(f"Prediction_method: {method}")
    print("-------------------------------")
    print(f"DataFrame length: {len(df)}")
    print(
        f"Right prediction odd: [ {right_pred_q_40} : {right_pred_q_60} ] | Right prediction odd: [ {wrong_pred_q_40} : {wrong_pred_q_60} ]")
    print(f"Bet amount: {bet_amount} | Total earning: {total_bank:.2f}")
    print(
        f"Accuracy: {np.mean(df['is_correct']):.2%} | Right: {sum(df['is_correct'])} | Wrong: {len(df) - sum(df['is_correct'])}")
    print("-------------------------------")

    num_bins = 5  # Number of groups to divide the data into
    df['odd_range'] = pd.qcut(df['pred_odd'], q=num_bins, precision=2)

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
