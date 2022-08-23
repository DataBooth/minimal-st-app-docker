from pathlib import Path

import streamlit as st
from src.helper import render_markdown_file

tab1, tab2, tab3 = st.tabs(["Data", "Profile", "Exploration"])

with tab1:
    st.info("TODO")

with tab2:
    st.info("TODO - pandas profiling on both train/test.csv")


with tab3:
    render_markdown_file(Path.cwd()/"docs/eda_titanic_background.md")

    render_markdown_file(Path.cwd()/"docs/nbconvert/eda_titanic_data.md")
