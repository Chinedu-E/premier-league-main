from utils import fetch_fixtures, update_past_games
import os
from features import build_features
from train import train_and_predict
import datetime
from email.message import EmailMessage
import smtplib


fetch_fixtures(filename="fixtures.csv")
print("Downloaded fixtures")

update_past_games(filename="data/2022.csv")
print("Downloaded past matches")

print("Updating features...")
build_features()
print("Done")

model_prediction, model_info = train_and_predict()
# Replace current prediction with new one
os.remove("predictions/last_week_predictions.csv")
os.rename("predictions/predictions.csv", "predictions/last_week_predictions.csv")
model_prediction.to_csv("predictions/predictions.csv", index=False)

# Create a text/plain message
msg_txt = f"""Trained a {model_info.name} model\n\nMetrics:\n\t
        Accuracy score: {model_info.test_accuracy:.2f}\n\t
        Precision score: {model_info.test_precision:.2f}\n\t
        Recall score: {model_info.test_recall:.2f}\n\t
        F1 score: {model_info.test_f1:.2f}"""

print(msg_txt)