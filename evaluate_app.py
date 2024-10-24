import numpy as np
import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt

from evaluate import evaluate_tournament
from utils import features_dataset_encoded


st.title("Models Evaluation")



# models_v2/test_2
# CatBoost_r137_FULL - 5100
# CatBoost_r86_FULL - 4848


# models_v2/test_3
# CatBoost_r49 - 6100

# major_matches_170_10_2021_2024
# RandomForest_r16_FULL - 5000

# models_v2/major_matches_170_10_2019_2024
# CatBoost_r13_FUL - 5000

MODEL_PATH = 'models_v2/major_matches_170_10_2019_2024'



TI13_PATH = 'data/datasets/ti13.pkl'
PGL_VALHALLA_PATH = 'data/datasets/pgl_valhalla.pkl'
VALIDATION_DATASET_PATH = 'data/datasets/validation_dataset.pkl'
BB_DACHA_PATH = 'parser/bb_dacha.pkl'


@st.cache_resource
def load_model(model_path):
    return TabularPredictor.load(model_path)


predictor = load_model(MODEL_PATH)
st.sidebar.success("Model loaded successfully!")


@st.cache_resource
def load_data(file_path):
    return pd.read_pickle(file_path)


def print_stat(df, total_bank, method='', bet_amount=100):


    st.header(f"Метод предсказания: {method}")
    st.write("---")

    st.markdown(f"**Количество матчей:** {len(df)}")
    st.markdown(f"**Сумма ставки:** {bet_amount}")
    st.markdown(f"**Общий выигрыш:** {total_bank:.2f}")
    st.markdown(f"**Точность:** {np.mean(df['is_correct']):.2%}")
    st.markdown(f"**Правильных предсказаний:** {sum(df['is_correct'])}")
    st.markdown(f"**Неправильных предсказаний:** {len(df) - sum(df['is_correct'])}")
    st.write("---")

    bin_edges = [0, 1.4, 1.7, 2.0, 3.0, 6.0]
    df['odd_range'] = pd.cut(df['pred_odd'], bins=bin_edges, include_lowest=True)

    grouped = df.groupby('odd_range', observed=True)['is_correct'].value_counts().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.plot(kind='bar', stacked=False, color=['red', 'green'], ax=ax)
    plt.title('Количество правильных и неправильных предсказаний по диапазонам коэффициентов')
    plt.xlabel('Диапазон коэффициентов')
    plt.ylabel('Количество предсказаний')
    plt.legend(['Неправильные', 'Правильные'])


    st.pyplot(fig)

    st.write('------')


def evaluate_models(predictor, ti13, pgl_valhalla, valid_df, bb_dacha):
    if ti13 is not None:
        st.subheader("TI13 Valhalla")
        ti13_features = features_dataset_encoded(ti13, radiant_first=False)
        st.write(predictor.leaderboard(ti13_features))

    if pgl_valhalla is not None:
        st.subheader("PGL Valhalla")
        pgl_valhalla_features = features_dataset_encoded(pgl_valhalla, radiant_first=False)
        st.write(predictor.leaderboard(pgl_valhalla_features))

    if valid_df is not None:
        st.subheader("Valid")
        valid_df_features = features_dataset_encoded(valid_df, radiant_first=False)
        st.write(predictor.leaderboard(valid_df_features))

    if bb_dacha is not None:
        st.subheader("BB Dacha")
        bb_dacha_features = features_dataset_encoded(bb_dacha, radiant_first=False)
        st.write(predictor.leaderboard(bb_dacha_features))

    if ti13 is not None and pgl_valhalla is not None and valid_df is not None and bb_dacha is not None:
        st.subheader("Combination")
        concat = pd.concat([ti13_features, valid_df_features, pgl_valhalla_features, bb_dacha_features]).reset_index(drop=True)
        st.write(predictor.leaderboard(concat))


def financial_metrics_for_model(model):
    df1, bank1 = evaluate_tournament(ti13, predictor, model, only_odds_included=False, radiant_first=False)
    print_stat(df1, total_bank=bank1, method='TI13')
    df2, bank2 = evaluate_tournament(valid_df, predictor, model, only_odds_included=False, radiant_first=False)
    print_stat(df2, total_bank=bank2, method='Valid')

    df4, bank4 = evaluate_tournament(bb_dacha, predictor, model, only_odds_included=False, radiant_first=False)
    print_stat(df4, total_bank=bank4, method='BB Dacha')

    df3, bank3 = evaluate_tournament(pgl_valhalla, predictor, model, only_odds_included=False, radiant_first=False)
    print_stat(df3, total_bank=bank3, method='PGL Valhalla')

    df_cat = pd.concat([df1, df2, df3]).reset_index(drop=True)
    bank_cat = bank1 + bank2 + bank3
    print_stat(df_cat, total_bank=bank_cat, method='Combination')



ti13 = load_data(TI13_PATH)
pgl_valhalla = load_data(PGL_VALHALLA_PATH)
valid_df = load_data(VALIDATION_DATASET_PATH)
bb_dacha = load_data(BB_DACHA_PATH)


with st.container():
    if st.button("Evaluate Models"):
        evaluate_models(predictor, ti13, pgl_valhalla, valid_df, bb_dacha)


with st.container():
    model = st.text_input(label='Model')
    if st.button("Financial Metrics"):
        financial_metrics_for_model(model)



