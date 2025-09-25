from typing import List, Tuple
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def build_scaler(method: str, X: pd.DataFrame):
    method = method.lower()
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown normalization method: use 'zscore' or 'minmax'")
    scaler.fit(X.values)
    return scaler


def apply_scaler(scaler, X: pd.DataFrame) -> pd.DataFrame:
    Xn = pd.DataFrame(scaler.transform(X.values), columns=X.columns, index=X.index)
    return Xn


def save_scaler(path: str, scaler, columns: List[str]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"scaler": scaler, "columns": columns}, f)


def load_scaler(path: str) -> Tuple[object | None, List[str]]:
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
            return obj["scaler"], obj["columns"]
    except FileNotFoundError:
        return None, []
