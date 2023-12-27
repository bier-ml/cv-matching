import pandas as pd
import dvc.api
from pathlib import Path


if __name__ == '__main__':
    for path in Path(".").rglob("*.csv"):
        df = pd.read_csv('data.csv')
        dvc.api.save(df=df, path=path, name=path.name, repo='.')
