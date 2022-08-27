import pandas as pd
import streamlit as st
from config import settings

def list_diff(list1, list2):
    return sorted(list(set(list1).symmetric_difference(set(list2))))


def create_sidebar(columns, target="Survived"):
    st.sidebar.subheader("Column (Feature) selection:")

    feat_all = sorted(list(columns))
    feat_all.remove(target)    # remove target variable from features

    dropped_column = sorted(settings.DROPPED_COLUMN)
    feat_all = list_diff(feat_all, dropped_column)


    st.sidebar.markdown("#### Columns dropped:")
    st.sidebar.markdown(f"_{', '.join(dropped_column)}_")

    # Initialise state variables for selected & discard features

    if "feat_select" not in st.session_state:
        feat_select = feat_all
        st.session_state.feat_select = sorted(feat_select)

    if "feat_discard" not in st.session_state:
        feat_discard = None
        st.session_state.feat_discard = feat_discard

    st.session_state.feat_select = st.sidebar.multiselect(label="Include columns (feat):",
            options=st.session_state.feat_select, default=st.session_state.feat_select, key=1)

    st.session_state.feat_discard = list_diff(feat_all, st.session_state.feat_select)
    
    st.session_state.feat_discard = st.sidebar.multiselect(label="Discard columns (feat):",
            options=st.session_state.feat_discard, default=st.session_state.feat_discard, key=2)

    st.session_state.feat_select = list_diff(feat_all, st.session_state.feat_discard)

    # st.sidebar.write(st.session_state.feat_select)
    # st.sidebar.write(st.session_state.feat_discard)


def create_sidebar_utilities():
    st.sidebar.subheader("Utilities")
