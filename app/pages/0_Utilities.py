from pathlib import Path

import streamlit as st
from Main import APP_TITLE
from src.helper import get_system_info
from src.sidebar import create_sidebar_utilities
from seedir import seedir

ITEM_LIMIT = 50

def create_page_0_tab_1():
    for item in get_system_info():
        st.code(item)


def create_page_0_tab_2():
  depth = st.selectbox(label="Directory tree: Level depth", options=range(1, 6))

  my_path = Path.cwd().as_posix()
  exclude_folders = [folder.parts[-1] for folder in Path(my_path).iterdir()
                      if folder.is_dir() and "venv" in folder.as_posix()]
  exclude_folders.append(".git")
  dir_str = seedir(path=my_path, style='emoji', sort=True,
                     printout=False, itemlimit=ITEM_LIMIT, depthlimit=depth,
                     exclude_folders=exclude_folders)
  st.code(f"Exclude directory(s): {', '.join(exclude_folders)}")
  st.code(dir_str)

  create_sidebar_utilities()


TAB_NAMES = ["System Info", "Directory structure"]

tab1, tab2 = st.tabs(TAB_NAMES)

with tab1:
  create_page_0_tab_1()

with tab2:
  create_page_0_tab_2()
