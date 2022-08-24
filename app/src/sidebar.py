from urllib.parse import urlparse
import streamlit as st


def create_sidebar_main():
    st.sidebar.header("Sidebar Main Header")

    # DOESN'T WORK
    # frontend_url = request.META.get('HTTP_REFERER')
    # url = urlparse(frontend_url)
    # st.sidebar.write(url)


def create_sidebar_utilities():
    st.sidebar.subheader("Utilities")