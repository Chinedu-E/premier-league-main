import utils as U
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelLogger:
    name: str
    test_accuracy: float = .0
    test_recall: float = .0
    test_precision: float = .0
    test_f1: float = .0

    def update_accuracy(self, y_pred, y_test):
        self.test_accuracy = accuracy_score(y_test, y_pred)

    def update_recall(self, y_pred, y_test):
        self.test_recall = recall_score(y_test, y_pred, average="macro")

    def update_precision(self, y_pred, y_test):
        self.test_precision = precision_score(y_test, y_pred, average="macro")

    def update_f1(self, y_pred, y_test):
        self.test_f1 = f1_score(y_test, y_pred, average="macro")

    def update_metrics(self, y_pred, y_test):
        self.update_f1(y_pred, y_test)
        self.update_precision(y_pred, y_test)
        self.update_accuracy(y_pred, y_test)
        self.update_recall(y_pred, y_test)


def train_and_predict() -> tuple[pd.DataFrame, ModelLogger]:
    features = pd.read_csv("pipeline/features.csv")
    labels = pd.read_csv("pipeline/labels.csv")

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    scaler = StandardScaler()

    _, features = U.split_features(features)
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.02, random_state=0, shuffle=False)

    print(f"shape of training data {X_train.shape}")
    print(f"shape of testing data {X_test.shape}")

    lr = LogisticRegression()
    param_grid = [{ "penalty": ["L1", "L2", "elasticnet", "none"],
                    "C": [100, 10, 1.0, 0.1, 0.01],
                    "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                    "max_iter": [100, 1000, 2500, 5000]}]

    grid_search = GridSearchCV(lr, param_grid=param_grid,
                                   cv=5, n_jobs=-1, verbose=True)

    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_

    tuned_model.fit(X_train, y_train)
    y_pred = tuned_model.predict(X_test)

    model_info = ModelLogger(lr.__class__.__name__)
    model_info.update_metrics(y_pred, y_test)

    df_pred = pd.read_csv("pipeline/to_predict.csv")
    _, df_pred = U.split_features(df_pred)
    df_pred = scaler.transform(df_pred)
    home, away, draw = [], [], []
    pred = tuned_model.predict_proba(df_pred)
    for prob in pred:
        home.append(f'{int(round(prob[2]*100, 0))}')
        draw.append(f'{int(round(prob[1]*100, 0))}')
        away.append(f'{int(round(prob[0]*100, 0))}')
    df_fixtures = pd.read_csv("fixtures.csv", encoding="cp1252")
    model_predictions = df_fixtures[df_fixtures["Div"] == "E0"][["Date", "Time", "HomeTeam", "AwayTeam"]]
    model_predictions["Home %"] = home
    model_predictions["Draw %"] = draw
    model_predictions["Away %"] = away

    return model_predictions, model_info

