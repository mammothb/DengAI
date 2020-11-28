import pandas as pd


class Processor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.output_dir = data_dir.parent / "processed"

    def process_and_save(self):
        df_train_features = pd.read_csv(self.data_dir / "dengue_features_train.csv")
        df_train_labels = pd.read_csv(self.data_dir / "dengue_labels_train.csv")
        df_test_features = pd.read_csv(self.data_dir / "dengue_features_test.csv")
        df_train = df_train_features.copy()
        df_train["total_cases"] = df_train_labels["total_cases"]

        df_train.to_csv(self.output_dir / "train.csv", index=False)
        df_test_features.to_csv(self.output_dir / "test.csv", index=False)
