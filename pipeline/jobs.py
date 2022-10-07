from utils import fetch_fixtures, update_past_games


fetch_fixtures(filename="fixtures.csv")
print("Downloaded fixtures")
update_past_games(filename="data/2022.csv")
print("Downloaded past matches")
print("Updating features...")
import features
print("Done")