import random

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import multiprocessing
from pipeline import utils as U
from sklearn.preprocessing import LabelEncoder
import json


class ModelSelector:

    def __init__(self, models: list, x: pd.DataFrame, y: pd.DataFrame):
        self.models = models
        self.info = {model.__class__.__name__: {}
                     for model in models}
        self.x = x
        self.y = y

    def tune(self):
        ...

    def train(self, train_data, test_data, val_data, run_id):
        features = list(train_data[0].columns)
        for model in self.models:
            model.fit(train_data[0], train_data[1])
            score = model.score(test_data[0], test_data[1])
            val_score = model.score(val_data[0], val_data[1])
            self.info[model.__class__.__name__][f"run_{run_id}"] = {
                # "features": features,
                "number of features": len(features),
                "test_score": score,
                "val_score": val_score
            }

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.05, random_state=0,
                                                            shuffle=False)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2, random_state=0,
                                                        shuffle=False)

        return [x_train, y_train], [x_test, y_test], [x_val, y_val]

    def select_features(self, method="random", num_runs=5):
        features = self.x.columns
        num_features = len(features)
        for run in range(num_runs):
            if method == "random":
                test_feat_num = random.randint(1, num_features)
                test_features = self.get_features(features, test_feat_num)
                print("Testing Features:", test_features)
                train_data, test_data, val_data = self.split_features(self.split_data(), test_features)
                # Training all models with subset features
                self.train(train_data, test_data, val_data, run)

    @staticmethod
    def split_features(data, features):
        for d in data:
            d[0] = d[0][features]
        return data

    @staticmethod
    def get_features(features, n):
        new_features = []
        i = 0
        while True:
            if i <= n-1:
                rand_feat = random.choice(features)
                if rand_feat not in new_features:
                    new_features.append(rand_feat)
                    i += 1
            else:
                break
        return new_features


if __name__ == "__main__":
    features = pd.read_csv("features.csv")
    labels = pd.read_csv("labels.csv")
    _, features = U.split_features(features)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    xgb = XGBClassifier()
    models = [rf, lr, xgb]
    model_s = ModelSelector(models, features, labels)
    model_s.select_features(num_runs=5)
    with open("model_runs.json", "w") as f:
        json.dump(model_s.info, f, indent=4)
    print(model_s.info.keys())
