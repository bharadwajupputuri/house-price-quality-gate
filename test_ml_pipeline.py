
import json
import os
import unittest

import joblib
import pandas as pd


class TestHousePriceMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("house_prices_practice.csv"))

    def test_dataset_not_empty(self):
        df = pd.read_csv("house_prices_practice.csv")
        self.assertGreater(len(df), 0)

    def test_required_columns_exist(self):
        df = pd.read_csv("house_prices_practice.csv")

        required_columns = [
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "TotalBsmtSF",
            "YearBuilt",
            "FullBath",
            "BedroomAbvGr",
            "LotArea",
            "SalePrice"
        ]

        for column in required_columns:
            self.assertIn(column, df.columns)

    def test_trained_model_exists(self):
        self.assertTrue(os.path.exists("house_price_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["training_records"], 1)
        self.assertGreaterEqual(metrics["testing_records"], 1)

    def test_model_prediction_is_positive(self):
        model = joblib.load("house_price_model.pkl")

        sample = pd.DataFrame([{
            "OverallQual": 7,
            "GrLivArea": 1800,
            "GarageCars": 2,
            "TotalBsmtSF": 1000,
            "YearBuilt": 2005,
            "FullBath": 2,
            "BedroomAbvGr": 3,
            "LotArea": 8000
        }])

        prediction = model.predict(sample)[0]
        self.assertGreater(prediction, 0)


if __name__ == "__main__":
    unittest.main()
