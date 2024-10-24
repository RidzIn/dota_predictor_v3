import streamlit as st
from autogluon.tabular import TabularPredictor

from prediction import get_prediction, parse_match_info_for_gui, get_votes_prediction
from utils import read_heroes


# models_v2/test_2
# CatBoost_r137_FULL - 5100
# CatBoost_r86_FULL - 4848


# models_v2/test_3
# CatBoost_r49 - 6100

# major_matches_170_10_2021_2024
# RandomForest_r16_FULL - 5000

# models_v2/major_matches_170_10_2019_2024
# CatBoost_r13_FUL - 5000


predictor_1 = TabularPredictor.load('models_v2/test_2', require_version_match=False)


predictor_2 = TabularPredictor.load('models_v2/test_3', require_version_match=False)


predictor_3 = TabularPredictor.load('models_v2/major_matches_170_10_2021_2024', require_version_match=False)


predictor_4 = TabularPredictor.load('models_v2/major_matches_170_10_2019_2024', require_version_match=False)

models = [
    (predictor_1, 'CatBoost_r137_FULL', 'attributes'),
    (predictor_1, 'CatBoost_r86_FULL', 'attributes'),
    (predictor_2, 'CatBoost_r49', 'attributes'),
    (predictor_3, 'RandomForest_r16_FULL', 'heroes'),
    (predictor_4, 'CatBoost_r13_FULL', 'heroes')
]


def display_full_prediction(dire_pick, radiant_pick, dire_team='Dire', radiant_team='Radiant'):

    _, pred_pick, pred_team, score = get_votes_prediction(dire_pick, radiant_pick, models, dire_team, radiant_team)

    st.write('-----')

    st.header(f"{pred_team}")

    col1, col2 = st.columns(2)

    with col1:
        st.json(pred_pick)
    with col2:
        st.metric('', f"{score}")

    st.write('-----')


tab0, tab2 = st.tabs(["Link", 'Test Yourself'])

with tab0:
    st.header("Insert Link")
    link = st.text_input(label="Enter")
    map_number = st.selectbox("Map number:", list(range(1, 5)))
    predict_button = st.button('Predict')
    if predict_button:
        match_info = parse_match_info_for_gui(link, map_number)
        display_full_prediction(match_info['dire_heroes'], match_info['radiant_heroes'], match_info['dire_team'], match_info['radiant_team'])


with tab2:
    heroes = read_heroes()
    """
    ## \tSELECT HEROES FOR DIRE TEAM
    """

    dire_1, dire_2, dire_3, dire_4, dire_5 = st.columns(5)

    with dire_1:
        d1 = st.selectbox("Dire Position 1", heroes, index=None)

    with dire_2:
        d2 = st.selectbox("Dire Position 2", heroes, index=None)

    with dire_3:
        d3 = st.selectbox("Dire Position 3", heroes, index=None)

    with dire_4:
        d4 = st.selectbox("Dire Position 4", heroes, index=None)

    with dire_5:
        d5 = st.selectbox("Dire Position 5", heroes, index=None)

    """
    ## \tSELECT HEROES FOR RADIANT TEAM
    """

    radiant_1, radiant_2, radiant_3, radiant_4, radiant_5 = st.columns(5)

    with radiant_1:
        r1 = st.selectbox("Radiant Position 1", heroes, index=None)

    with radiant_2:
        r2 = st.selectbox("Radiant Position 2", heroes, index=None)

    with radiant_3:
        r3 = st.selectbox("Radiant Position 3", heroes, index=None)

    with radiant_4:
        r4 = st.selectbox("Radiant Position 4", heroes, index=None)

    with radiant_5:
        r5 = st.selectbox("Radiant Position 5", heroes, index=None)

    if st.button("Predict", key=1):
        dire_pick = [d1, d2, d3, d4, d5]
        radiant_pick = [r1, r2, r3, r4, r5]
        display_full_prediction(dire_pick, radiant_pick)

