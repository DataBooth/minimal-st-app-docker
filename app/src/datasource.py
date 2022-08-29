from streamlit import cache
import pandas as pd
import pandas_profiling
from pathlib import Path

CSV_URL_EXAMPLE = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"

#@cache
def load_data(local=False):
    data_url = Path.cwd().parent/"data" \
        if local else "https://raw.githubusercontent.com/Mjboothaus/titanic/main/data"
    try:
        titanic_train = pd.read_csv(f"{data_url}/train.csv")
        titanic_test = pd.read_csv(f"{data_url}/test.csv")
    except Exception as e:
        raise(e)
    return titanic_train, titanic_test


# @cache(hash_funcs={pandas_profiling.report.presentation.core.container.Container: lambda _: None})
def create_data_profile(df):
    return df.profile_report(minimal=True)
