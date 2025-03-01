import pandas as pd
import os
from src.constants import ARTIFACT, RAW_PATH, TRAIN, TEST
from sklearn.model_selection import train_test_split
import prefect
from prefect import task, Flow


class DataLoader:

    def __init__(self) -> None:
        pass

    @task
    def data_load(self, path):

        try:

            df = pd.read_csv(path)

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            raw_data_path = os.path.join(ARTIFACT, RAW_PATH)
            os.makedirs(raw_data_path, exist_ok=True)
            # save train test to csv
            train_data.to_csv(os.path.join(raw_data_path, str(TRAIN)), index=False)
            test_data.to_csv(os.path.join(raw_data_path, str(TEST)), index=False)

            return train_data, test_data

        except Exception as e:
            raise e
