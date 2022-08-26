import pandas as pd
import streamlit as st


def list_diff(list1, list2):
    return list(set(list1).symmetric_difference(set(list2)))


def create_sidebar_main(df: pd.DataFrame, target="Survived"):
    st.sidebar.subheader("Column (Feature) selection:")

    feat_all = list(df.columns)
    feat_all.remove(target)    # remove target variable from features

    try:
        st.code(st.session_state.feat_select)
        st.code(st.session_state.feat_discard)
    except:
        pass

    # Initialise state variables for selected & discard features

    if "feat_select" not in st.session_state:
        feat_select = feat_all
        st.session_state.feat_select = feat_select

    if "feat_discard" not in st.session_state:
        feat_discard = None
        st.session_state.feat_discard = feat_discard

    st.session_state.feat_select = st.sidebar.multiselect(label="Include columns (feat):",
            options=st.session_state.feat_select, default=st.session_state.feat_select)

    st.session_state.feat_discard = list_diff(feat_all, st.session_state.feat_select)
    
    st.session_state.feat_discard = st.sidebar.multiselect(label="Discard columns (feat):",
            options=st.session_state.feat_discard, default=st.session_state.feat_discard)

    st.session_state.feat_select = list_diff(feat_all, st.session_state.feat_discard)


def create_sidebar_utilities():
    st.sidebar.subheader("Utilities")
