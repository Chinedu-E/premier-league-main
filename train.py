from pipeline import utils as U
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# from keras.layers import Dense, Dropout, Input
# from keras.models import Model
# from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def main():
    features = pd.read_csv("pipeline/features.csv")
    num_teams = len(features["HomeTeam"].unique())
    labels = pd.read_csv("pipeline/labels.csv")

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # tokenizer = Tokenizer(num_words=num_teams)
    scaler = StandardScaler()

    teams, features = U.split_features(features)
    # team_sequences, tokenizer = U.teams_to_sequences(teams, tokenizer)
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.02, random_state=0, shuffle=False)

    print(f"shape of training data {X_train.shape}")
    print(f"shape of testing data {X_test.shape}")

    rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=0)
    lr.fit(X_train, y_train)

    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X_train, y_train)

    vc = VotingClassifier(estimators=[("lr", lr), ("rf", rf), ("ada", ada)],
                          weights=[1, 1.5, 1],
                          voting="soft")
    vc.fit(X_train, y_train)

    y_rf = rf.predict(X_test)
    y_lr = lr.predict(X_test)
    y_ada = ada.predict(X_test)
    y_vc = vc.predict(X_test)
    y_knn = knn.predict(X_test)

    rf_acc = accuracy_score(y_test, y_rf)
    lr_acc = accuracy_score(y_test, y_lr)
    ada_acc = accuracy_score(y_test, y_ada)
    vc_acc = accuracy_score(y_test, y_vc)
    knn_acc = accuracy_score(y_test, y_knn)

    rf_f1 = f1_score(y_test, y_rf, average="macro")
    lr_f1 = f1_score(y_test, y_lr, average="macro")
    ada_f1 = f1_score(y_test, y_ada, average="macro")
    vc_f1 = f1_score(y_test, y_vc, average="macro")
    knn_f1 = f1_score(y_test, y_knn, average="macro")

    rf_precision = precision_score(y_test, y_rf, average="macro")
    lr_precision = precision_score(y_test, y_lr, average="macro")
    ada_precision = precision_score(y_test, y_ada, average="macro")
    vc_precision = precision_score(y_test, y_vc, average="macro")
    knn_precision = precision_score(y_test, y_knn, average="macro")

    rf_recall = recall_score(y_test, y_rf, average="macro")
    lr_recall = recall_score(y_test, y_lr, average="macro")
    ada_recall = recall_score(y_test, y_ada, average="macro")
    vc_recall = recall_score(y_test, y_vc, average="macro")
    knn_recall = recall_score(y_test, y_knn, average="macro")

    print(f"Random Forest Classifier: \n\n Accuracy: {rf_acc}\nf1 score: {rf_f1}\nPrecision: {rf_precision}\nRecall: {rf_recall}\n")

    print(f"Logistic Regression: \n\n Accuracy: {lr_acc}\nf1 score: {lr_f1}\nPrecision: {lr_precision}\nRecall: {lr_recall}\n")

    print(f"AdaBoost Classifier: \n\n Accuracy: {ada_acc}\nf1 score: {ada_f1}\nPrecision: {ada_precision}\nRecall: {ada_recall}\n")

    print(f"Voting Classifier: \n\n Accuracy: {vc_acc}\nf1 score: {vc_f1}\nPrecision: {vc_precision}\nRecall: {vc_recall}\n")

    print(f"KNNeighbors Classifier: \n\n Accuracy: {knn_acc}\nf1 score: {knn_f1}\nPrecision: {knn_precision}\nRecall: {knn_recall}\n")
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(features, labels)
    df_pred = pd.read_csv("pipeline/to_predict.csv")
    _, df_pred = U.split_features(df_pred)
    df_pred = scaler.transform(df_pred)
    home, away, draw = [], [], []
    pred = rf.predict_proba(df_pred)
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


if __name__ == '__main__':
    main()
