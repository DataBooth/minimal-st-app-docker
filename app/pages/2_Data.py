from pathlib import Path

import pandas as pd
import streamlit as st
from src.helper import render_markdown_file
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling





def create_data_profile(df):
    return df.profile_report(minimal=True)


tab1, tab2, tab3, tab4 = st.tabs(["Data", "Profiling", "Exploration", "Governance"])

with tab1:
    st.info("TODO")


with tab2:
    titanic_training, titanic_test = load_data()
    tab2_train, tab2_test = st.tabs(["Train", "Test"])
    with tab2_train:
        st_profile_report(create_data_profile(titanic_training))
    with tab2_test:
        st_profile_report(create_data_profile(titanic_test))


with tab3:
    render_markdown_file(Path.cwd()/"docs/eda_titanic_background.md")
    render_markdown_file(Path.cwd()/"docs/nbconvert/eda_titanic_data.md")


with tab4: 
    render_markdown_file(Path.cwd()/"docs/data_governance.md")