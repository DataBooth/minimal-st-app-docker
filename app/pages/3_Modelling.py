from pathlib import Path

import streamlit as st
from src.datasource import load_data
from src.helper import render_markdown_file
from src.sidebar import create_sidebar
from src.modelling_logistic_simple import create_logistic_regression_pipe_simple, get_fit_predict
from millify import millify


train, test = load_data()    # no validation set in this simple exercise
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

    st.markdown(f"**Numeric columns:** {', '.join(num_column)}")
    st.markdown(f"**Category columns:** {', '.join(cat_column)}")

    if st.button(label="Run logistic regression..."):
        pipe = create_logistic_regression_pipe_simple(num_column, cat_column)

        X_train = train.copy()
        y_train = X_train["Survived"]
        X_train.drop("Survived", axis="columns")    # this is our target / variable to  be predicted

        # Testing data

        X_test = test.copy()

        fit_score, y_test = get_fit_predict(X_train, y_train, X_test, pipe)

        X_test["Survived"] = y_test

        st.metric(label="Regression fit score", value=millify(fit_score*100, 2), )

        st.write("Predictions for test set (**Survived**):")
        st.dataframe(X_test[train.columns])

with tab2:
    render_markdown_file(Path.cwd()/"docs/modelling.md")