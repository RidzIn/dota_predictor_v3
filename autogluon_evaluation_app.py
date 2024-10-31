import numpy as np
import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt

from evaluate import evaluate_tournament
from utils import features_dataset_encoded

st.title("Models Evaluation")

# Allow the user to input the model path
MODEL_PATH = st.text_input('Enter the model path', 'models/onehot_radiant_170_10_2021_2024')
method = st.selectbox('Select method', ['onehot', 'attributes', 'heroes'])  # Replace 'other_method' with actual methods
is_radiant = st.checkbox('Is radiant first?', value=True)


# Define available datasets
datasets = {
    'TI13 Valhalla': 'data/datasets/evaluation_datasets/ti13.pkl',
    'PGL Valhalla': 'data/datasets/evaluation_datasets/pgl_valhalla.pkl',
    'Validation Dataset': 'data/datasets/evaluation_datasets/validation_dataset.pkl',
    'BB Dacha': 'data/datasets/evaluation_datasets/bb_dacha.pkl',
    'Dreamleague': 'parser/dreamleague.pkl'
}

# Allow user to select datasets
selected_datasets = st.multiselect('Select datasets to evaluate', list(datasets.keys()), default=list(datasets.keys()))


@st.cache_resource
def load_model(model_path):
    return TabularPredictor.load(model_path)


# Load the model
try:
    predictor = load_model(MODEL_PATH)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    predictor = None


@st.cache_resource
def load_data(file_path):
    return pd.read_pickle(file_path)


def print_stat(df, total_bank, method='', bet_amount=100):
    st.header(f"Prediction Method: {method}")
    st.write("---")

    st.markdown(f"**Number of Matches:** {len(df)}")
    st.markdown(f"**Bet Amount:** {bet_amount}")
    st.markdown(f"**Total Profit:** {total_bank:.2f}")
    st.markdown(f"**Accuracy:** {np.mean(df['is_correct']):.2%}")
    st.markdown(f"**Correct Predictions:** {sum(df['is_correct'])}")
    st.markdown(f"**Incorrect Predictions:** {len(df) - sum(df['is_correct'])}")
    st.write("---")

    bin_edges = [0, 1.4, 1.7, 2.0, 3.0, 6.0]
    df['odd_range'] = pd.cut(df['pred_odd'], bins=bin_edges, include_lowest=True)

    grouped = df.groupby('odd_range', observed=True)['is_correct'].value_counts().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.plot(kind='bar', stacked=False, color=['red', 'green'], ax=ax)
    plt.title('Number of Correct and Incorrect Predictions by Odds Range')
    plt.xlabel('Odds Range')
    plt.ylabel('Number of Predictions')
    plt.legend(['Incorrect', 'Correct'])

    st.pyplot(fig)

    st.write('------')


def evaluate_models(predictor, datasets, is_radiant=True, method='onehot'):
    features_list = []
    leaderboards = {}
    for dataset_name in datasets:
        dataset_path = datasets[dataset_name]
        st.subheader(dataset_name)
        data = load_data(dataset_path)
        features = features_dataset_encoded(data, radiant_first=is_radiant, method=method)
        leaderboard = predictor.leaderboard(features, silent=True)
        st.write(leaderboard)
        features_list.append(features)
        leaderboards[dataset_name] = leaderboard

    # Combine features and display combined leaderboard
    if features_list:
        st.subheader("Combined Datasets")
        concat_features = pd.concat(features_list).reset_index(drop=True)
        concat_leaderboard = predictor.leaderboard(concat_features, silent=True)
        st.write(concat_leaderboard)

        # Get top models
        top_models = concat_leaderboard.head(5)['model'].tolist()

        # Prepare comparison table
        rows = []
        for model in top_models:
            row = {'model': model}
            row['combined_score'] = concat_leaderboard[concat_leaderboard['model'] == model]['score_test'].values[0]
            for dataset_name in datasets:
                leaderboard = leaderboards[dataset_name]
                if model in leaderboard['model'].values:
                    score = leaderboard[leaderboard['model'] == model]['score_test'].values[0]
                    row[f'{dataset_name}_score'] = score
            rows.append(row)
        final_df = pd.DataFrame(rows)
        st.write(final_df)


def financial_metrics_for_model(model_name, predictor, datasets, is_radiant=True, method='onehot'):
    dfs = []
    total_bank = 0
    for dataset_name in datasets:
        dataset_path = datasets[dataset_name]
        data = load_data(dataset_path)
        df, bank = evaluate_tournament(data, predictor, model_name, only_odds_included=False, radiant_first=is_radiant, method=method)
        print_stat(df, total_bank=bank, method=dataset_name)
        dfs.append(df)
        total_bank += bank
    if dfs:
        df_cat = pd.concat(dfs).reset_index(drop=True)
        print_stat(df_cat, total_bank=total_bank, method='Combined Datasets')


with st.container():
    if st.button("Evaluate Models"):
        if predictor is not None:
            selected_dataset_paths = {name: datasets[name] for name in selected_datasets}
            evaluate_models(predictor, selected_dataset_paths, is_radiant=is_radiant, method=method)
        else:
            st.error("Model is not loaded.")

with st.container():
    model_name = st.text_input(label='Enter Model Name for Financial Metrics')
    if st.button("Calculate Financial Metrics"):
        if predictor is not None and model_name:
            selected_dataset_paths = {name: datasets[name] for name in selected_datasets}
            financial_metrics_for_model(model_name, predictor, selected_dataset_paths, is_radiant=is_radiant, method=method)
        else:
            st.error("Model is not loaded or model name is empty.")
