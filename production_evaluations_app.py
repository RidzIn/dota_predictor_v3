import numpy as np
import pandas as pd
import streamlit as st
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt

from evaluate import evaluate_tournament


def print_stat(tournament, total_bank, method='', bet_amount=100):
    st.header(f"Prediction Method: {method}")
    st.write("---")

    st.markdown(f"**Number of Matches:** {len(tournament)}")
    st.markdown(f"**Bet Amount:** {bet_amount}")
    st.markdown(f"**Total Profit:** {total_bank:.2f}")
    st.markdown(f"**Accuracy:** {np.mean(tournament['is_correct']):.2%}")
    st.markdown(f"**Correct Predictions:** {sum(tournament['is_correct'])}")
    st.markdown(f"**Incorrect Predictions:** {len(tournament) - sum(tournament['is_correct'])}")
    st.write("---")

    bin_edges = [0, 1.4, 1.7, 2.0, 3.0, 6.0]
    tournament['odd_range'] = pd.cut(tournament['pred_odd'], bins=bin_edges, include_lowest=True)

    grouped = tournament.groupby('odd_range', observed=True)['is_correct'].value_counts().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.plot(kind='bar', stacked=False, color=['red', 'green'], ax=ax)
    plt.title('Number of Correct and Incorrect Predictions by Odds Range')
    plt.xlabel('Odds Range')
    plt.ylabel('Number of Predictions')
    plt.legend(['Incorrect', 'Correct'])

    st.pyplot(fig)

    st.write('------')


models = [
    (TabularPredictor.load('models/attributes_dire_170_10_2019_2024', require_version_match=False),
     'ExtraTreesEntr_FULL', 'attributes', False),

    (TabularPredictor.load('models/attributes_radiant_170_10_2019_2024', require_version_match=False),
     'LightGBMXT', 'attributes', True),

    (TabularPredictor.load('models/attributes_radiant_170_10_2021_2024', require_version_match=False),
     'LightGBM_r143', 'attributes', True),

    (TabularPredictor.load('models/heroes_only_dire_170_10_2019_2024', require_version_match=False),
     'CatBoost_r69', 'heroes', False),

    (TabularPredictor.load('models/heroes_only_dire_170_10_2021_2024', require_version_match=False),
     'CatBoost_r167', 'heroes', False),

    (TabularPredictor.load('models/heroes_only_radiant_170_10_2019_2024', require_version_match=False),
     'CatBoost_r50_FULL', 'heroes', True),

    (TabularPredictor.load('models/heroes_only_radiant_170_10_2021_2024', require_version_match=False),
     'LightGBM_r131', 'heroes', True),

    (TabularPredictor.load('models/onehot_dire_170_10_2019_2024', require_version_match=False),
     'NeuralNetTorch_r185_FULL', 'onehot', False),

    (TabularPredictor.load('models/onehot_radiant_170_10_2021_2024', require_version_match=False),
     'XGBoost_r98_FULL', 'onehot', True)
]


datasets = {
    'TI13': pd.read_pickle('data/datasets/evaluation_datasets/ti13.pkl'),
    'PGL Valhalla': pd.read_pickle('data/datasets/evaluation_datasets/pgl_valhalla.pkl'),
    'Validation Dataset': pd.read_pickle('data/datasets/evaluation_datasets/validation_dataset.pkl'),
    'BB Dacha': pd.read_pickle('data/datasets/evaluation_datasets/bb_dacha.pkl'),
    'Dreamleague': pd.read_pickle('data/datasets/evaluation_datasets/dreamleague_group_stage.pkl')
}

tournament = datasets[st.selectbox('Tournament', datasets.keys(), key=1)]

teams = set(tournament.dire_team) | set(tournament.radiant_team)

teams_to_exclude = st.multiselect('Teams to avoid prediction', teams, default=[])

teams_to_include = st.multiselect('Teams to make prediction on', teams, default=teams)


page1, page2 = st.tabs(['Evaluate on scores', 'Evaluate on Model'])

with page1:
    threshold = st.selectbox('Score threshold', list(range(5, 10)))

    if st.button('evaluate'):
        df, bank = evaluate_tournament(tournament, models=models, method='votes', threshold=threshold,
                                       teams_to_exclude=teams_to_exclude, teams_to_include=teams_to_include)
        print_stat(df, bank, method='Votes')

        st.dataframe(df)

with page2:
    target_model = st.selectbox('Models', [model[1] for model in models])
    model_index = next((i for i, model in enumerate(models) if model[1] == target_model), None)
    predictor = models[model_index][0]
    model_name = models[model_index][1]
    model_method = models[model_index][2]
    radiant_first = models[model_index][3]

    if st.button("Predict"):
        df, bank = evaluate_tournament(tournament, threshold=0.5, predictor=predictor, model=model_name, method=model_method, radiant_first=radiant_first,
                                       teams_to_exclude=teams_to_exclude, teams_to_include=teams_to_include)
        print_stat(df, bank, method='Model')

        st.dataframe(df)


