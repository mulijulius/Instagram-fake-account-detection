import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
