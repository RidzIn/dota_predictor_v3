import streamlit as st
from autogluon.tabular import TabularPredictor

from prediction import get_prediction, get_hero_stats, parse_match_info_for_gui, get_meta_prediction
from utils import get_match_picks, read_heroes


# # CatBoost_r50_FULL
# predictor_dire = TabularPredictor.load('models_v2/major_matches_170_10_2021_2024', require_version_match=False)
# dire_model = 'XGBoost_r34'



# CatBoost_r13_FULL
# models_v2/major_matches_170_10_2021_2024
predictor_1 = TabularPredictor.load('models_v2/major_matches_170_10_2021_2024', require_version_match=False)


# LightGBM_r15, CatBoost_r86_FULL
# models_v2/major_matches_170_10_2019_2024
predictor_2 = TabularPredictor.load('models_v2/major_matches_170_10_2019_2024', require_version_match=False)


def get_pred(dire_pick, radiant_pick, dire_team, radiant_team):
    pred1 = get_prediction(dire_pick, radiant_pick, predictor_1, 'CatBoost_r13_FULL', is_proba=False)
    pred2 = get_prediction(dire_pick, radiant_pick, predictor_2, 'CatBoost_r86_FULL', is_proba=False)
    pred3 = get_prediction(dire_pick, radiant_pick, predictor_2, 'LightGBM_r15', is_proba=False)
    # st.write(pred1[0])
    # st.write(pred2[0])
    # st.write(pred3[0])

    if sum([pred1[0], pred2[0], pred3[0]]) >= 2:
        return dire_pick, dire_team, sum([pred1[0], pred2[0], pred3[0]])
    else:
        return radiant_pick, radiant_team, 3 - sum([pred1[0], pred2[0], pred3[0]])


def display_full_prediction(dire_pick, radiant_pick, dire_team='Dire', radiant_team='Radiant'):

    print(dire_pick)
    print(radiant_pick)

    # dire_pred = get_prediction(dire_pick, radiant_pick, predictor_dire, dire_model)

    pred_pick, pred_team, score = get_pred(dire_pick, radiant_pick, dire_team, radiant_team)

    st.write('-----')

    st.header(f"{pred_team}")

    col1, col2 = st.columns(2)

    with col1:
        st.json(pred_pick)
    with col2:
        st.metric('', f"{score}")

    st.write('-----')

    # st.write(pred_team)
    # st.write(pred_team)
    # print(dire_pred)
    # if float(dire_pred[False]) > 0.5:
    #     st.header(f"{radiant_team}")
    #     st.write('-----')
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.json(radiant_pick)
    #     with col2:
    #         st.metric('', f"{float(dire_pred[False]) * 100:.2f}%")
    #     st.write('-----')
    #
    # if float(dire_pred[True] > 0.5):
    #     st.header(f"{dire_team}")
    #     st.write('-----')
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.json(dire_pick)
    #     with col2:
    #         st.metric('', f"{float(dire_pred[True]) * 100:.2f}%")
    #     st.write('-----')


tab0, tab1, tab2 = st.tabs(["Link", 'Match id', 'Test Yourself'])

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

