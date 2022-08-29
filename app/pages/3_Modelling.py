from pathlib import Path

import streamlit as st
from src.datasource import load_data
from src.helper import render_markdown_file
from src.sidebar import create_sidebar


train, test = load_data()
create_sidebar(train.columns)

tab1, tab2 = st.tabs(["Modelling", "Background"])

with tab1:
    num_column = []
    cat_column = []
    for column in (column for column in train.columns if column in st.session_state.feat_select):
        #st.write(f"{column}: {train[column].dtype}")
        if train[column].dtype in ["int64", "float64"]:
            num_column.append(column)
        elif train[column].dtype in ["object"]:
            cat_column.append(column)
        else:
            st.error("Unknown pandas type")

    st.markdown("**Choose the columns (features) you wish to include/exclude from the model in the sidebar.**")
    st.markdown("##### Running logistic regression...")

    st.markdown(f"**Numeric columns:** {', '.join(num_column)}")
    st.markdown(f"**Category columns:** {', '.join(cat_column)}")

with tab2:
    render_markdown_file(Path.cwd()/"docs/modelling.md")