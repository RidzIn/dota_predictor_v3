import streamlit as st
from autogluon.tabular import TabularPredictor

from prediction import get_prediction, get_hero_stats, parse_match_info_for_gui, get_meta_prediction
from utils import get_match_picks, read_heroes


predictor_dire = TabularPredictor.load('AutogluonModels/dire_first', require_version_match=False)
dire_model = 'NeuralNetTorch_r36_BAG_L2\\e1e10a75'


predictor_radiant = TabularPredictor.load('AutogluonModels/radiant_first', require_version_match=False)
radiant_model = 'LightGBM_BAG_L1\\T10'


def display_hero_stats(dire_pick, radiant_pick, pred_pick=None):
    df = get_hero_stats(dire_pick, radiant_pick)
    if pred_pick == 'Dire':
        st.header("Predicted Hero Stats")
        hero, synergy, against = st.columns(3)
        for i in range(5):
            with hero:
                st.write(f"{dire_pick[i]}")
            with synergy:
                st.write(f"Synergy: {df.iloc[0][2 * i + 1] * 100:.2f}%")
            with against:
                st.write(f"Against: {df.iloc[0][2 * i + 11] * 100:.2f}%")
        st.write('--------')

        st.header("Unpredicted Hero Stats")
        hero, synergy, against = st.columns(3)
        for i in range(5):
            with hero:
                st.write(f"{radiant_pick[i]}")
            with synergy:
                st.write(f"Synergy: {df.iloc[0][2 * i + 21] * 100:.2f}%")
            with against:
                st.write(f"Against: {(1 - df.iloc[0][2 * i + 11]) * 100:.2f}%")

    elif pred_pick == 'Radiant':
        st.header("PREDICTED Hero Stats")

        hero, synergy, against = st.columns(3)
        for i in range(5):
            with hero:
                st.write(f"{radiant_pick[i]}")
            with synergy:
                st.write(f"Synergy: {df.iloc[0][2 * i + 21] * 100:.2f}%")
            with against:
                st.write(f"Against: {(1 - df.iloc[0][2 * i + 11]) * 100:.2f}%")
        st.write('---------')

        st.header("UNPREDICTED Hero Stats")
        hero, synergy, against = st.columns(3)
        for i in range(5):
            with hero:
                st.write(f"{dire_pick[i]}")
            with synergy:
                st.write(f"Synergy: {df.iloc[0][2 * i + 1] * 100:.2f}%")
            with against:
                st.write(f"Against: {df.iloc[0][2 * i + 11] * 100:.2f}%")
        st.write('--------')


def display_full_prediction(dire_pick, radiant_pick, dire_team='Dire', radiant_team='Radiant'):

    dire_pred = get_prediction(dire_pick, radiant_pick, predictor_dire, dire_model)
    radiant_pred = get_prediction(dire_pick, radiant_pick, predictor_radiant, radiant_model, radiant_first=True)
    if float(dire_pred[0]) > 0.5:
        st.header(f"{radiant_team}")
        st.write('-----')
        col1, col2 = st.columns(2)
        with col1:
            st.json(radiant_pick)
        with col2:
            st.metric('', f"{float(dire_pred[0]) * 100:.2f}%")
            st.metric('', f"{float(radiant_pred[1]) * 100:.2f}%")
        st.write('-----')
        display_hero_stats(dire_pick, radiant_pick, 'Radiant')

    if float(dire_pred[1] > 0.5):
        st.header(f"{dire_team}")
        st.write('-----')
        col1, col2 = st.columns(2)
        with col1:
            st.json(dire_pick)
        with col2:
            st.metric('', f"{float(dire_pred[1]) * 100:.2f}%")
            st.metric('', f"{float(radiant_pred[0]) * 100:.2f}%")
        st.write('-----')
        display_hero_stats(dire_pick, radiant_pick, 'Dire')

    meta_pred = get_meta_prediction(dire_pick, radiant_pick)

    st.write("-----")
    col1, col2 = st.columns(2)
    with col1:
        st.header(f"**{dire_team}**")
        st.metric("", f"{meta_pred['dire']*100:.2f}%")
    with col2:
        st.header(f"**{radiant_team}**")
        st.metric("", f"{meta_pred['radiant'] * 100:.2f}%")


tab0, tab1, tab2 = st.tabs(["Link", 'Match id', 'Test Yourself'])

with tab0:
    st.header("Insert Link")
    link = st.text_input(label="Enter")
    map_number = st.selectbox("Map number:", list(range(1, 5)))
    predict_button = st.button('Predict')
    if predict_button:
        match_info = parse_match_info_for_gui(link, map_number)
        display_full_prediction(match_info['dire_heroes'], match_info['radiant_heroes'], match_info['dire_team'], match_info['radiant_team'])

with tab1:
    match_id = st.number_input(label="Put match id")

    if st.button("Predict", key=2):
        temp_dict = get_match_picks(int(match_id))
        display_full_prediction(temp_dict['dire'], temp_dict['radiant'], temp_dict['dire_team'], temp_dict['radiant_team'])

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

