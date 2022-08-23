import streamlit as st

from Main import APP_TITLE

st.markdown("API example")

tab1, tab2 = st.tabs(["Data", "Analysis"])

with tab1:
  st.write(f"{APP_TITLE} - Data / Quality")
  
  
with tab2:
  st.write(f"{APP_TITLE} - Analysis")
