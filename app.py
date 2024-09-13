import streamlit as st

from prediction import get_meta_prediction, get_winrates_prediction, get_onehot_prediction, get_hero_stats
from utils import get_match_picks, read_heroes


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
    else:
        st.header("Radiant Hero Stats")

        hero, synergy, against = st.columns(3)
        for i in range(5):
            with hero:
                st.write(f"{radiant_pick[i]}")
            with synergy:
                st.write(f"Synergy: {df.iloc[0][2 * i + 21] * 100:.2f}%")
            with against:
                st.write(f"Against: {(1 - df.iloc[0][2 * i + 11]) * 100:.2f}%")
        st.write('---------')

        st.header("Dire Hero Stats")
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

    ml_pred = get_winrates_prediction(dire_pick, radiant_pick)
    nn_pred = get_onehot_prediction(dire_pick, radiant_pick)

    if float(ml_pred[0]) > 0.5 and float(nn_pred[0]) > 0.5:
        st.header(f"{radiant_team}")
        st.write('-----')
        st.json(radiant_pick)
        st.write('-----')
        display_hero_stats(dire_pick, radiant_pick, 'Radiant')
    else:
        display_hero_stats(dire_pick, radiant_pick)

    if float(ml_pred[0]) > 0.5 and float(nn_pred[0]) > 0.5:
        meta = get_meta_prediction(dire_pick, radiant_pick)
        st.write('-----')
        meta_col, nn_col, ml_col = st.columns(3)
        with meta_col:
            st.header("Meta")
            st.metric('', f"{meta['radiant'] * 100:.2f}%")
        with nn_col:
            st.header("NN")
            st.metric('', f"{float(nn_pred[0]) * 100:.2f}%")
        with ml_col:
            st.header("NL")
            st.metric('', f"{float(ml_pred[0]) * 100:.2f}%")

    if float(ml_pred[1]) > 0.5 and float(nn_pred[1] > 0.5):
        st.header(f"{dire_team}")
        st.write('-----')
        st.json(dire_pick)
        st.write('-----')
        display_hero_stats(dire_pick, radiant_pick, 'Dire')
    else:
        display_hero_stats(dire_pick, radiant_pick)


    if float(ml_pred[1]) > 0.5 and float(nn_pred[1] > 0.5):
        meta = get_meta_prediction(dire_pick, radiant_pick)
        st.write('-----')

        meta_col, nn_col, ml_col = st.columns(3)
        with meta_col:
            st.header("Meta")
            st.metric('', f"{meta['dire'] * 100:.2f}%")
        with nn_col:
            st.header("NN")
            st.metric('', f"{float(nn_pred[1]) * 100:.2f}%")
        with ml_col:
            st.header("ML")
            st.metric('', f"{float(ml_pred[1]) * 100:.2f}%")
    st.write('-----')


tab1, tab2 = st.tabs(["Link", 'Test Yourself'])

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
        display_full_prediction(dire_pick, radiant_pick, )

