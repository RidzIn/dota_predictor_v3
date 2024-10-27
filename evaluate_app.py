import numpy as np
import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt

from evaluate import evaluate_tournament
from utils import features_dataset_encoded


st.title("Models Evaluation")

models = [
    # XGBoost_r98_FULL	0.5859872611464968	0.6176470588235294	0.5555555555555556	0.6274509803921569	0.5492957746478874
    # onehot_radiant_170_10_2021_2024
    ('models_v3/onehot_radiant_170_10_2021_2024', 'XGBoost_r98_FULL', 'onehot', True),

    # LightGBMXT	0.5700636942675159	0.5882352941176471	0.5666666666666667	0.5490196078431373	0.5633802816901409
    # models_v3/attributes_radiant_170_10_2019_2024
    ('models_v3/attributes_radiant_170_10_2019_2024', 'LightGBMXT', 'attributes', True),

    # LightGBM_r131	0.5828025477707006	0.6372549019607843	0.6333333333333333	0.5098039215686274	0.49295774647887325
    # models_v3/heroes_only_radiant_170_10_2021_2024
    ('models_v3/heroes_only_radiant_170_10_2021_2024', 'LightGBM_r131', 'heroes', True),

    # NeuralNetTorch_r185_FULL	0.5573248407643312	0.49019607843137253	0.5777777777777777	0.6078431372549019	0.5915492957746479
    # models_v3/onehot_dire_170_10_2019_2024
    ('models_v3/onehot_dire_170_10_2019_2024', 'NeuralNetTorch_r185_FULL', 'onehot', False),

    # LightGBMLarge	0.5573248407643312	0.5980392156862745	0.6111111111111112	0.5098039215686274	0.4647887323943662
    # models_v3/onehot_dire_170_10_2019_2024
    ('models_v3/onehot_dire_170_10_2019_2024', 'LightGBMLarge', 'onehot', False),

    # CatBoost_r167	0.5732484076433121	0.49019607843137253	0.5888888888888889	0.6470588235294118	0.6197183098591549
    # models_v3/heroes_only_dire_170_10_2021_2024
    ('models_v3/heroes_only_dire_170_10_2021_2024', 'CatBoost_r167', 'heroes', False),

    # LightGBM_r143	0.5732484076433121	0.6274509803921569	0.6222222222222222	0.5098039215686274	0.4788732394366197
    # models_v3/attributes_radiant_170_10_2021_2024
    ('models_v3/attributes_radiant_170_10_2021_2024', 'LightGBM_r143', 'attributes', True)
]

# ExtraTreesEntr_FULL	0.554140127388535	0.4803921568627451	0.6	0.49019607843137253	0.647887323943662
# models_v3/attributes_dire_170_10_2019_2024

# LightGBMXT	0.5700636942675159	0.5882352941176471	0.5666666666666667	0.5490196078431373	0.5633802816901409
# models_v3/attributes_radiant_170_10_2019_2024

# LightGBM_r143	0.5732484076433121	0.6274509803921569	0.6222222222222222	0.5098039215686274	0.4788732394366197
# models_v3/attributes_radiant_170_10_2021_2024

# CatBoost_r69	0.5764331210191083	0.5784313725490197	0.6	0.6666666666666666	0.4788732394366197
# models_v3/heroes_only_dire_170_10_2019_2024

# CatBoost_r167	0.5732484076433121	0.49019607843137253	0.5888888888888889	0.6470588235294118	0.6197183098591549
# models_v3/heroes_only_dire_170_10_2021_2024

# CatBoost_r50_FULL	0.5668789808917197	0.5784313725490197	0.5666666666666667	0.5882352941176471	0.5352112676056338
# models_v3/heroes_only_radiant_170_10_2019_2024

# LightGBM_r131	0.5828025477707006	0.6372549019607843	0.6333333333333333	0.5098039215686274	0.49295774647887325
# models_v3/heroes_only_radiant_170_10_2021_2024

# NeuralNetTorch_r185_FULL	0.5573248407643312	0.49019607843137253	0.5777777777777777	0.6078431372549019	0.5915492957746479
# models_v3/onehot_dire_170_10_2019_2024

# XGBoost_r98_FULL	0.5859872611464968	0.6176470588235294	0.5555555555555556	0.6274509803921569	0.5492957746478874
# models_v3/onehot_radiant_170_10_2021_2024


MODEL_PATH = 'models_v3/onehot_radiant_170_10_2021_2024'
method = 'onehot'
is_radiant = True



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
        ti13_features = features_dataset_encoded(ti13, radiant_first=is_radiant, method=method)
        ti13_leaderboard = predictor.leaderboard(ti13_features)
        st.write(ti13_leaderboard)

    if pgl_valhalla is not None:
        st.subheader("PGL Valhalla")
        pgl_valhalla_features = features_dataset_encoded(pgl_valhalla, radiant_first=is_radiant, method=method)
        pgl_valhalla_leaderboard = predictor.leaderboard(pgl_valhalla_features)
        st.write(pgl_valhalla_leaderboard)

    if valid_df is not None:
        st.subheader("Valid")
        valid_df_features = features_dataset_encoded(valid_df, radiant_first=is_radiant, method=method)
        valid_leaderboard = predictor.leaderboard(valid_df_features)
        st.write(valid_leaderboard)

    if bb_dacha is not None:
        st.subheader("BB Dacha")
        bb_dacha_features = features_dataset_encoded(bb_dacha, radiant_first=is_radiant, method=method)
        bb_dacha_leaderboard = predictor.leaderboard(bb_dacha_features)
        st.write(bb_dacha_leaderboard)


    if ti13 is not None and pgl_valhalla is not None and valid_df is not None and bb_dacha is not None:
        st.subheader("Combination")
        concat_features = pd.concat([ti13_features, valid_df_features, pgl_valhalla_features, bb_dacha_features]).reset_index(drop=True)
        concat_leaderboard = predictor.leaderboard(concat_features)
        st.write(concat_leaderboard)

    top_models = concat_leaderboard.head(5)['model'].tolist()

    rows = []

    for model in top_models:
        row = {'model': model}
        row['concat_score'] = concat_leaderboard[concat_leaderboard['model'] == model]['score_test'].values[0]

        if ti13 is not None:
            row['ti13_score'] = ti13_leaderboard[ti13_leaderboard['model'] == model]['score_test'].values[0]
        if pgl_valhalla is not None:
            row['pgl_valhalla_score'] = \
            pgl_valhalla_leaderboard[pgl_valhalla_leaderboard['model'] == model]['score_test'].values[0]
        if valid_df is not None:
            row['valid_score'] = valid_leaderboard[valid_leaderboard['model'] == model]['score_test'].values[0]
        if bb_dacha is not None:
            row['bb_dacha_score'] = bb_dacha_leaderboard[bb_dacha_leaderboard['model'] == model]['score_test'].values[0]

        rows.append(row)

    final_df = pd.DataFrame(rows)

    st.write(final_df)



def financial_metrics_for_model(model):
    df1, bank1 = evaluate_tournament(ti13, predictor, model, only_odds_included=False, radiant_first=is_radiant, method=method)
    print_stat(df1, total_bank=bank1, method='TI13')
    df2, bank2 = evaluate_tournament(valid_df, predictor, model, only_odds_included=False, radiant_first=is_radiant,method=method)
    print_stat(df2, total_bank=bank2, method='Valid')

    df4, bank4 = evaluate_tournament(bb_dacha, predictor, model, only_odds_included=False, radiant_first=is_radiant,method=method)
    print_stat(df4, total_bank=bank4, method='BB Dacha')

    df3, bank3 = evaluate_tournament(pgl_valhalla, predictor, model, only_odds_included=False, radiant_first=is_radiant,method=method)
    print_stat(df3, total_bank=bank3, method='PGL Valhalla')

    df_cat = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)
    bank_cat = bank1 + bank2 + bank3 + bank4
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



