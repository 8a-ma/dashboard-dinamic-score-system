import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline


def save_df_to_csv(df: pd.DataFrame, output_file: Path) -> None:
    assert output_file.parent.exists()

    df.to_csv(output_file, index=False, encoding='utf-8')

    assert output_file.exists()


def load_csv_to_df(input_file: Path) -> pd.DataFrame:
    assert input_file.exists()

    df: pd.DataFrame = pd.read_csv(input_file)

    return df


def save_baseline_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    assert path.exists()


def load_baseline_model(path: Path) -> Pipeline:
    assert path.exists()

    with open(path, 'rb') as f:
        model: Pipeline = pickle.load(f)

    assert isinstance(model, Pipeline)

    return model


def save_dict_to_json(data: dict, path: Path, indent: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


    assert path.exists()
