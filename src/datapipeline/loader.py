import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

from src.datapipeline.processor import Processor


def create_lag_features(data, n_in=1, n_out=1, dropna=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]

    df = pd.DataFrame(data)
    cols = []
    col_names = []
    for i in reversed(range(n_in)):
        cols.append(df.shift(i))
        col_names.extend(
            [f"var{j + 1}(t)" for j in range(n_vars)]
            if i == 0
            else [f"var{j + 1}(t-{i})" for j in range(n_vars)]
        )
    for i in range(1, n_out):
        cols.append(df.shift(-i))
        col_names.extend(f"var{j + 1}(t+{i})" for j in range(n_vars))
    lag_features = pd.concat(cols, axis=1)
    lag_features.columns = col_names
    if dropna:
        lag_features.dropna(inplace=True)
    return lag_features


def create_train_lag_features(data, n_in):
    data_wide = create_lag_features(data.drop("total_cases", axis=1), n_in)
    data_wide.drop(
        ["var1(t)", "var2(t)", "var3(t)", "var4(t)", "var5(t)"],
        axis=1,
        inplace=True,
    )
    data_wide["total_cases"] = data["total_cases"]
    return data_wide


def create_test_lag_features(train_data, test_data, city, n_in):
    data_wide = (
        train_data.drop(["total_cases"], axis=1)
        .iloc[-n_in:, :]
        .append(test_data.drop(["city", "weekofyear", "year"], axis=1))
        .reset_index(drop=True)
    )
    data_wide = create_lag_features(data_wide, n_in)
    data_wide.drop(
        ["var1(t)", "var2(t)", "var3(t)", "var4(t)", "var5(t)"],
        axis=1,
        inplace=True,
    )
    data_wide["city"] = city
    data_wide["year"] = test_data["year"]
    data_wide["weekofyear"] = test_data["weekofyear"]
    return data_wide


class Loader:
    def __init__(self, data_dir):
        processor = Processor(data_dir)
        processor.process_and_save()

        self.input_dir = processor.output_dir
        self.output_dir = self.input_dir.parent / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_sj = self.get_pipeline()
        self.pipeline_iq = self.get_pipeline()

        self.poly_data = None
        self.weekly_index = None

    def get_pipeline(self):
        poly_features = PolynomialFeatures(
            degree=2, interaction_only=False, include_bias=True
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("features", FeatureUnion([("poly", poly_features)])),
                ("scaler", RobustScaler()),
            ]
        )

    def transform_train_data(self, numeric_cols):
        df_train = pd.read_csv(self.input_dir / "train.csv")
        rows_sj = df_train["city"] == "sj"
        rows_iq = df_train["city"] == "iq"

        poly_variables = ["weekofyear", "total_cases"]
        weekly_cases_sj = df_train[rows_sj][poly_variables].groupby("weekofyear").mean()
        poly_fit_sj = np.polyfit(
            weekly_cases_sj.index, weekly_cases_sj["total_cases"], deg=4
        )
        poly_line_sj = np.poly1d(poly_fit_sj)

        weekly_cases_iq = df_train[rows_iq][poly_variables].groupby("weekofyear").mean()
        poly_fit_iq = np.polyfit(
            weekly_cases_iq.index, weekly_cases_iq["total_cases"], deg=4
        )
        poly_line_iq = np.poly1d(poly_fit_iq)
        self.poly_data = dict(
            sj=poly_line_sj(weekly_cases_sj.index),
            iq=poly_line_iq(weekly_cases_iq.index),
        )
        self.weekly_index = weekly_cases_sj.index
        for i in self.weekly_index:
            df_train.loc[
                rows_sj & (df_train["weekofyear"] == i), "poly_fit"
            ] = self.poly_data["sj"][i - 1]
            df_train.loc[
                rows_iq & (df_train["weekofyear"] == i), "poly_fit"
            ] = self.poly_data["iq"][i - 1]

        df_train.loc[:, numeric_cols].interpolate(inplace=True)
        df_train_sj = df_train[rows_sj]
        df_train_iq = df_train[rows_iq].reset_index().drop("index", axis=1)

        self.pipeline_sj.fit(df_train_sj[numeric_cols], df_train_sj["total_cases"])
        self.pipeline_iq.fit(df_train_iq[numeric_cols], df_train_iq["total_cases"])

        numeric_train_sj = self.pipeline_sj.transform(df_train_sj[numeric_cols])
        train_features_sj = pd.DataFrame(numeric_train_sj)
        train_features_sj["total_cases"] = df_train_sj["total_cases"]

        numeric_train_iq = self.pipeline_iq.transform(df_train_iq[numeric_cols])
        train_features_iq = pd.DataFrame(numeric_train_iq)
        train_features_iq["total_cases"] = df_train_iq["total_cases"]

        return train_features_sj, train_features_iq

    def transform_test_data(self, numeric_cols):
        df_test = pd.read_csv(self.input_dir / "test.csv")
        rows_sj = df_test["city"] == "sj"
        rows_iq = df_test["city"] == "iq"

        for i in self.weekly_index:
            df_test.loc[
                rows_sj & (df_test["weekofyear"] == i), "poly_fit"
            ] = self.poly_data["sj"][i - 1]
            df_test.loc[
                rows_iq & (df_test["weekofyear"] == i), "poly_fit"
            ] = self.poly_data["iq"][i - 1]

        df_test.loc[:, numeric_cols].interpolate(inplace=True)
        df_test_sj = df_test[rows_sj]
        df_test_iq = df_test[rows_iq].reset_index().drop("index", axis=1)

        numeric_test_sj = self.pipeline_sj.transform(df_test_sj[numeric_cols])
        test_features_sj = pd.DataFrame(numeric_test_sj)
        test_features_sj["city"] = df_test_sj["city"]
        test_features_sj["year"] = df_test_sj["year"]
        test_features_sj["weekofyear"] = df_test_sj["weekofyear"]

        numeric_test_iq = self.pipeline_iq.transform(df_test_iq[numeric_cols])
        test_features_iq = pd.DataFrame(numeric_test_iq)
        test_features_iq["city"] = df_test_iq["city"]
        test_features_iq["year"] = df_test_iq["year"]
        test_features_iq["weekofyear"] = df_test_iq["weekofyear"]

        return test_features_sj, test_features_iq

    def transform_and_save(self, numeric_cols, n_in):
        train_features_sj, train_features_iq = self.transform_train_data(numeric_cols)
        test_features_sj, test_features_iq = self.transform_test_data(numeric_cols)

        train_sj_wide = create_train_lag_features(train_features_sj, n_in)
        train_iq_wide = create_train_lag_features(train_features_iq, n_in)
        test_sj_wide = create_test_lag_features(
            train_features_sj, test_features_sj, "sj", n_in
        )
        test_iq_wide = create_test_lag_features(
            train_features_iq, test_features_iq, "iq", n_in
        )

        train_features_sj.to_csv(self.output_dir / "sj_train.csv", index=False)
        train_features_iq.to_csv(self.output_dir / "iq_train.csv", index=False)
        test_features_sj.to_csv(self.output_dir / "sj_test.csv", index=False)
        test_features_iq.to_csv(self.output_dir / "iq_test.csv", index=False)

        train_sj_wide.to_csv(self.output_dir / "sj_train_lag.csv", index=False)
        train_iq_wide.to_csv(self.output_dir / "iq_train_lag.csv", index=False)
        test_sj_wide.to_csv(self.output_dir / "sj_test_lag.csv", index=False)
        test_iq_wide.to_csv(self.output_dir / "iq_test_lag.csv", index=False)

        return train_sj_wide, train_iq_wide, test_sj_wide, test_iq_wide
