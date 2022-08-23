import pandas as pd
from streamlit import cache
import pandas_profiling

CSV_URL_EXAMPLE = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"

@cache
def get_data_pandas_example(datasource=CSV_URL_EXAMPLE, sep=",", header=0):
    try:
        return pd.read_csv(datasource, sep=sep, header=header)
    except Exception as e:
        raise(e)


#@cache(hash_funcs={pandas_profiling.report.presentation.core.container.Container: lambda _: None})
def create_data_profile(df):
    return df.profile_report(minimal=True)