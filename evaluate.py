from prediction import get_onehot_prediction
import pandas as pd
import numpy as np


def evaluate_tournament(tournament, threshold=0.55):
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
    y_pred_raw_list = []
    y_unpred_raw_list = []
    passed_matches = 0
    for row in tournament.itertuples():
        # Get prediction

        prediction_df = get_onehot_prediction(row.dire_heroes, row.radiant_heroes)
        # Prediction probability
        max_prob = prediction_df.values.max()
        if max_prob < threshold:
            print('Does not match the threshold')
            continue

        # Actual result
        actual_dire_win = row.dire_win
        winning_team = row.dire_team if actual_dire_win else row.radiant_team

        # Predicted result
        predicted_dire_win = bool(prediction_df.values.argmax())
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
        'is_correct': is_correct,
        'pred_proba': pred_proba,
        'y_pred_raw': y_pred_raw_list,
        'y_unpred_raw': y_unpred_raw_list
    })

    # Calculate and print accuracy
    accuracy = np.mean(is_correct)
    print(f"Prediction_method: NN")
    print(f"Accuracy: {accuracy:.2%} | Right: {sum(is_correct)} | Wrong: {passed_matches - sum(is_correct)}")
    print("-------------------------------")
    return df


# def evaluate_combination(tournament=None, nn_df=None, ml_df=None):
#     if tournament is not None:
#         ml_df = evaluate_tournament(tournament, prediction_method='ML')
#         nn_df = evaluate_tournament(tournament, prediction_method='NN')
#
#     combined_df = copy(nn_df)
#     combined_df['ml_pred'] = ml_df['pred_proba']
#
#     for i in range(len(nn_df)):
#         if nn_df.iloc[i]['y_pred'] != ml_df.iloc[i]['y_pred']:
#             combined_df.drop(i, inplace=True)
#
#     accuracy = np.mean(combined_df['is_correct'])
#     print(f"Prediction_method: Combination")
#     print(f"Matches length: {len(nn_df)} | Combination length: {len(combined_df)}")
#     print(f"Accuracy: {accuracy:.2%} | Right: {sum(combined_df['is_correct'])} | Wrong: {len(combined_df) - sum(combined_df['is_correct'])}")
#     print("-------------------------------")
#
#     return {'combined_df': combined_df, 'ml_df': ml_df, 'nn_df': nn_df}
