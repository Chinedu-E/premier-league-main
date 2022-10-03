import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import multiprocessing
import utils as U
from sklearn.preprocessing import LabelEncoder
import json


class ModelSelector:

    def __init__(self, models: list, x: pd.DataFrame, y: pd.DataFrame):
        self.models = models
        self.info = {model.__class__.__name__: {"test_score": [],
                                                "val_score": [],
                                                "features": []}
                     for model in models}
        self.x = x
        self.y = y
        self.best_model = None

    def train(self, train_data, test_data, val_data):
        features = list(train_data[0].columns)
        for model in self.models:
            self._train(model, train_data, test_data, val_data, features)
        
    def _train(self, model, train_data, test_data, val_data, features):
        model.fit(train_data[0], train_data[1])
        score = model.score(test_data[0], test_data[1])
        val_score = model.score(val_data[0], val_data[1])
        self.info[model.__class__.__name__]["test_score"].append(score)
        self.info[model.__class__.__name__]["val_score"].append(val_score)
        self.info[model.__class__.__name__]["features"].append(features)

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.05, random_state=0,
                                                            shuffle=False)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2, random_state=0,
                                                        shuffle=False)

        return [x_train, y_train], [x_test, y_test], [x_val, y_val]

    def select_features(self, num_times):
        for _ in range(num_times):
            self._select_features()

    def _select_features(self):
        features = self.x.columns
        num_features = len(features)
        test_feat_num = random.randint(1, num_features)
        test_features = self.get_features(features, test_feat_num)
        train_data, test_data, val_data = self.split_features(self.split_data(), test_features)
        # Training all models with subset features
        self.train(train_data, test_data, val_data)

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

    def get_best_model(self):
        best_idx = 0
        best_model = ""
        for model_name, data in self.info.items():
            data["val_score"] = np.array(data["val_score"])
            max_val_score = np.argmax(data["val_score"])
            if data["val_score"][max_val_score] > data["val_score"][best_idx]:
                best_idx = max_val_score
                best_model = model_name
        
        return {"best model": best_model,
                "best features": self.info[best_model]["features"][best_idx],
                "accuracy": self.info[best_model]["val_score"][best_idx]}

    def tune_model(self, model_name, param_grid, features):
        train_data, _, _ = self.split_features(self.split_data(), features)
        model = [model for model in self.models if model.__class__.__name__ == model_name][0]
        grid_search = GridSearchCV(model, param_grid=param_grid,
                                   cv=5, n_jobs=-1, verbose=True)
        grid_search.fit(train_data[0], train_data[1])
        tuned_model = grid_search.best_estimator_
        tuned_model.fit(self.x[features], self.y)
        return tuned_model

    
def make_predictions(path, scaler, model):
    df_pred = pd.read_csv(path)
    _, df_pred = U.split_features(df_pred)
    df_pred = scaler.transform(df_pred)
    home, away, draw = [], [], []
    pred = model.predict_proba(df_pred)
    for prob in pred:
        home.append(f'{prob[2]:.2f}')
        draw.append(f'{prob[1]:.2f}')
        away.append(f'{prob[0]:.2f}')
    df_fixtures = pd.read_csv("fixtures.csv", encoding="cp1252")
    model_predictions = df_fixtures[df_fixtures["Div"] == "E0"][["Date", "Time", "HomeTeam", "AwayTeam"]]
    model_predictions["HomeChance"] = home
    model_predictions["DrawChance"] = draw
    model_predictions["AwayChance"] = away

    model_predictions.to_csv(f"predictions.csv", index=False)
    print(model_predictions)

def load_params(path):
    with open(path) as f:
        all_params = json.load(f)
    return all_params

def main():
    path = "pipeline/model_params.json"
    model_params = load_params(path)
    features = pd.read_csv("pipeline/features.csv")
    labels = pd.read_csv("pipeline/labels.csv")
    _, features = U.split_features(features)
    le = LabelEncoder()
    labels = le.fit_transform(labels.values.flatten())
    rf = RandomForestClassifier(n_jobs=-1)
    lr = LogisticRegression(max_iter=400, n_jobs=-1)
    xgb = XGBClassifier()
    models = [rf, lr, xgb]
    model_s = ModelSelector(models, features, labels)
    model_s.select_features(5)
    model_info = model_s.get_best_model()
    best_model_name = model_info["best model"]
    grid = model_params[best_model_name]["params"]
    features_used = model_info["best features"]
    model = model_s.tune_model(best_model_name, grid, features=features_used)
    make_predictions("pipeline/to_predict.csv", ..., model)


if __name__ == "__main__":
    main()
    