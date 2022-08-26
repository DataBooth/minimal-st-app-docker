# Main.py -- for creating a basic multi-page Streamlit app

import os
from pathlib import Path
import pandas as pd

import streamlit as st

from config import settings
from src.helper import render_markdown_file
from src.sidebar import create_sidebar
from src.datasource import load_data

# from helper_functions import read_render_markdown_file

APP_TITLE = settings.APP_TITLE
SUB_TITLE = settings.SUB_TITLE
APP_ABOUT = settings.APP_ABOUT


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    menu_items={
        "About": APP_ABOUT
    })


def test_env_variable():
    ENV_VAR = "API_KEY"
    try:
        test_env_var = os.environ[ENV_VAR]
    except Exception:
        # TODO: This is not a great solution (see config.py for files that are loaded). Have to hard code the env variable name
        st.info("Running locally the environment variable is not available using value from `.env_dockerfile.toml`")
        test_env_var = settings.API_KEY
    st.markdown(f"Test environment variable: `{ENV_VAR} = {test_env_var}`")


def create_app_header(app_title, subtitle=None):
    st.header(app_title)
    if subtitle is not None:
        st.subheader(subtitle)
    return None


create_app_header(APP_TITLE, SUB_TITLE)

train_df, test_df = load_data()

create_sidebar(train_df.columns)

render_markdown_file(Path.cwd()/"docs/Main.md")

test_env_variable()