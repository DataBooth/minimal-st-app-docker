# See: https://towardsdatascience.com/creating-custom-transformers-for-sklearn-pipelines-d3d51852ecc1

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

set_config("diagram")

# The ColumnsSelector class inherits from the sklearn.base classes
# (BaseEstimator, TransformerMixin). This makes it compatible with
# scikit-learn’s Pipelines

class DataFrameColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        # save the features list internally in the class
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # return the dataframe with the specified features
        return X[self.columns]


def create_logistic_regression_pipe_simple(numeric_features, categorical_features):

    # Define the pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # handle categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))])

    # Create a transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Run the classifier
    classifier = LogisticRegression()

    # Set the Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)])
    return pipe

def get_fit_predict(X_train, y_train, X_test, pipe):
    pipe.fit(X_train, y_train)
    score = pipe.score(X_train, y_train)
    y_test = pipe.predict(X_test)
    return score, y_test