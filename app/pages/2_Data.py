from pathlib import Path

import pandas as pd
import streamlit as st
from src.helper import render_markdown_file
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
from src.sidebar import create_sidebar
from src.datasource import load_data


def create_data_profile(df):
    return df.profile_report(minimal=True)


train, test = load_data()
create_sidebar(train.columns)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Profiling", "Exploration", "Expectations", "Governance"])

with tab1:
    render_markdown_file(Path.cwd()/"docs/meta_data.md")
    st.markdown(train.info())
    st.write(train.describe())


with tab2:
    st.markdown("Data profiling for the selected columns (features) for the **Train** and **Test** data")
    tab2_train, tab2_test = st.tabs(["Train", "Test"])
    with tab2_train:
        st_profile_report(create_data_profile(train[st.session_state.feat_select]))
    with tab2_test:
        st_profile_report(create_data_profile(test[st.session_state.feat_select]))


with tab3:
    render_markdown_file(Path.cwd()/"docs/eda_titanic_background.md")
    render_markdown_file(Path.cwd()/"docs/nbconvert/eda_titanic_data.md")


with tab4: 
    render_markdown_file(Path.cwd()/"docs/data_expectations.md")


with tab5: 
    render_markdown_file(Path.cwd()/"docs/data_governance.md")