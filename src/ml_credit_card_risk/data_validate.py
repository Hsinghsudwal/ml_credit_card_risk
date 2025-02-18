from scipy.stats import ks_2samp
from src.constants import ARTIFACT, VALIDATE_PATH, TRAIN_VAL, TEST_VAL
import os


class DataValidation:

    def __init__(self) -> None:
        pass

    def data_validation(self, train_data, test_data):
        try:

            for feature in train_data.columns:
                ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])

                if p_value < 0.05:
                    return "Error drift data"

                else:
                    validate_data_path = os.path.join(ARTIFACT, VALIDATE_PATH)
                    os.makedirs(validate_data_path, exist_ok=True)
                    
                    
                    # save train test to csv
                    train_data.to_csv(
                        os.path.join(validate_data_path, str(TRAIN_VAL)),
                        index=False,
                    )
                    test_data.to_csv(
                        os.path.join(validate_data_path, str(TEST_VAL)),
                        index=False,
                    )

                    return train_data, test_data

        except Exception as e:
            raise e
