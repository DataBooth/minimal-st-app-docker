import pandas as pd
import streamlit as st


def list_diff(list1, list2):
    return list(set(list1).symmetric_difference(set(list2)))


def create_sidebar_main(df: pd.DataFrame, target="Survived"):
    st.sidebar.subheader("Column (Feature) selection:")

    all_features = list(df.columns)
    all_features.remove(target)  # remove the target variable from the list of features

    features_selected = all_features
    if "features_selected" not in st.session_state:
        st.session_state.features_selected = features_selected
    else:
        features_selected = st.session_state.features_selected

    features_discarded = None
    if "features_discarded" not in st.session_state:
        st.session_state.features_discarded = features_discarded
    else:
        features_discarded = st.session_state.features_discarded

    features_selected = st.sidebar.multiselect(label="Include columns (features):", 
        options=features_selected, default=features_selected)

    st.session_state.features_selected = features_selected

    features_discarded = list_diff(all_features, features_selected)
    st.session_state.features_discarded = features_discarded

    features_discarded = st.sidebar.multiselect(label="Discarded columns (features):", 
        options=features_discarded, default=features_discarded)

    st.session_state.features_discarded = features_discarded
    features_selected = list_diff(all_features, features_discarded)

    st.session_state.features_selected = features_selected
    return features_selected


def create_sidebar_utilities():
    st.sidebar.subheader("Utilities")