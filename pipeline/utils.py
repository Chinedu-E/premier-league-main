import pandas as pd
import requests
import numpy as np
import os
import math
from typing import Any


class Features:

    def __init__(self, data):
        self.features = self._clone(data)
        self.split = False
        self.to_predict: pd.DataFrame = ...

    def _clone(self, data):
        features = pd.DataFrame()
        features["HomeTeam"] = data["HomeTeam"]
        features["AwayTeam"] = data["AwayTeam"]
        return features

    def make_feature(self, name: str):
        if name not in self.columns:
            self.features[name] = np.zeros(len(self.features))

    def update_feature(self, feature: str, index: int, value: Any):
        if feature not in self.columns:
            self.make_feature(feature)
        self.features.loc[index, feature] = value

    def set_feature_value(self, feature: str, value: np.ndarray):
        self.features[feature] = value

    def __len__(self):
        return len(self.columns)

    def __str__(self) -> str:
        return str(self.features)

    def __array__(self):
        return self.features.values

    def to_csv(self):
        self.features.to_csv("pipeline/features.csv", index=False)
        if self.split:
            self.to_predict.to_csv("pipeline/to_predict.csv", index=False)

    def split_prediction(self, n: int):
        self.to_predict = self.features.iloc[-n:]
        self.features.drop(self.to_predict.index, inplace=True)
        self.split = True

    @property
    def columns(self):
        return self.features.columns


class Labels:
    def __init__(self, data, targets):
        self.targets = targets
        self.labels: pd.DataFrame = data[targets]

    def __len__(self):
        return len(self.columns)

    def __str__(self) -> str:
        return str(self.labels)

    def __array__(self):
        return self.labels.values

    def split_prediction(self, n: int):
        self.labels = self.labels.iloc[:-n]

    def to_csv(self):
        self.labels.to_csv("pipeline/labels.csv", index=False)

    @property
    def columns(self):
        return self.labels.columns


def initialize_table(teams: list[str]) -> pd.DataFrame:
    """Initializes a table for the season with given teams.

        Args:
            teams: team names to be put in the table.

        Returns:
            A pandas dataframe.

        """
    table = pd.DataFrame(columns=["Team", "HMP", "AMP", "W", "L", "D", "GSH",
                                  "GSA", "GCH", "GCA", "SoT",
                                  "GF", "GA", "GD", "Points"])
    table["Team"] = teams
    table.sort_values(by="Team", inplace=True)
    table.fillna(0, inplace=True)

    return table


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna(how='all', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def get_teams_from_season(data: pd.DataFrame) -> list[str]:
    """get the name of all teams in a season.

        Args:
            data: Dataframe of matches played in a season.

        Returns:
            A list of all team names.

        """
    teams = list(data["AwayTeam"].unique())
    return teams


def load_data(path: str) -> tuple[pd.DataFrame, int]:
    """Loads all csv files in a directory directly into memory.

        Args:
            path: Directory to all files.

        Returns:
            A dataframe containing all csv files.

        """
    files = [file for file in os.listdir(path)]
    print(f"{len(files)} files found")
    pdata = pd.DataFrame()
    for file in sorted(files):
        df = pd.read_csv(path + '/' + file, encoding='unicode_escape')
        pdata = pd.concat([pdata, df])
    pdata = merge_fixtures(pdata, load_fixtures())
    pdata = clean_data(pdata)
    return pdata, len(files)


def generate_data(path: str):
    """A generator that yields a dataframe file by file.

        Args:
            path: Directory to all files.

        Returns:
            Dataframe of a season.

        """
    files = [file for file in os.listdir(path)]
    for file in sorted(files):
        print(file)
        df = pd.read_csv(path + '/' + file, encoding='unicode_escape', parse_dates=["Date"])
        yield df


def update_season_table(game_week: pd.DataFrame, table: pd.DataFrame):
    """Updates the season table's columns based on what happened in the game week.

        Args:
            game_week: Dataframe only containing games from a game week.
            table: Initialized table of teams for a season.

        Returns:
            The return value. True for success, False otherwise.

        """
    new_table = table.copy()
    for i in game_week.index.values:
        home_team: str = game_week.loc[i]["HomeTeam"]
        home_goals: int = game_week.loc[i]["FTHG"]
        away_team: str = game_week.loc[i]["AwayTeam"]
        away_goals: int = game_week.loc[i]["FTAG"]
        home_shotontarget: int = game_week.loc[i]["HST"]
        away_shotontarget: int = game_week.loc[i]["AST"]
        result = game_week.loc[i]["FTR"]
        if result == "H":
            new_table.loc[new_table["Team"] == home_team, "W"] += 1
            new_table.loc[new_table["Team"] == home_team, "Points"] += 3
            new_table.loc[new_table["Team"] == away_team, "L"] += 1
        elif result == "A":
            new_table.loc[new_table["Team"] == away_team, "W"] += 1
            new_table.loc[new_table["Team"] == away_team, "Points"] += 3
            new_table.loc[new_table["Team"] == home_team, "L"] += 1
        elif result == "D":
            new_table.loc[new_table["Team"] == home_team, "D"] += 1
            new_table.loc[new_table["Team"] == away_team, "D"] += 1
            new_table.loc[new_table["Team"] == home_team, "Points"] += 1
            new_table.loc[new_table["Team"] == away_team, "Points"] += 1
        else:
            return table

        new_table.loc[new_table["Team"] == home_team, "GSH"] += home_goals
        new_table.loc[new_table["Team"] == home_team, "GCH"] += away_goals

        new_table.loc[new_table["Team"] == home_team, "GF"] += home_goals
        new_table.loc[new_table["Team"] == home_team, "GA"] += away_goals
        new_table.loc[new_table["Team"] == home_team, "GD"] += (home_goals - away_goals)

        new_table.loc[new_table["Team"] == away_team, "GSA"] += away_goals
        new_table.loc[new_table["Team"] == away_team, "GCA"] += home_goals

        new_table.loc[new_table["Team"] == away_team, "GF"] += away_goals
        new_table.loc[new_table["Team"] == away_team, "GA"] += home_goals
        new_table.loc[new_table["Team"] == away_team, "GD"] += (away_goals - home_goals)

        new_table.loc[new_table["Team"] == home_team, "HMP"] += 1
        new_table.loc[new_table["Team"] == away_team, "AMP"] += 1

        new_table.loc[new_table["Team"] == home_team, "SoT"] += home_shotontarget
        new_table.loc[new_table["Team"] == away_team, "SoT"] += away_shotontarget

    new_table.sort_values(by=["Points", "GD"], ascending=False, inplace=True)
    new_table.reset_index(drop=True, inplace=True)
    return new_table


def season_to_gameweeks(data: pd.DataFrame) -> list[pd.DataFrame]:
    """Splits a season into game weeks.

        Args:
            data: Dataframe of matches played in a season.

        Returns:
            A list of the season as game weeks.

        """
    game_weeks = [data.iloc[i:i + 10] for i in range(len(data)) if i % 10 == 0]

    return game_weeks


def get_position_on_table(team: str, table: pd.DataFrame) -> int:
    """Gets the position of a team from the table.

        Args:
            team: Name of team.
            table: Initialized table of teams for a season.

        Returns:
            The current position of that team in the table.

        """
    index = table.loc[table["Team"] == team].index[0]
    position = index + 1
    return position


def league_avg_goals_scored(table: pd.DataFrame) -> tuple[float, float]:
    """Average number of goals scored by all teams in the table.

        Args:
            table: Initialized table of teams for a season.

        Returns:
            A tuple of the league average goals scored at home and away.

        """
    home_avg_goals = table['GSH'].sum() / (table["HMP"].sum())
    away_avg_goals = table['GSA'].sum() / (table["AMP"].sum())
    return home_avg_goals, away_avg_goals


def team_goals_scored(team: str, table: pd.DataFrame) -> tuple[int, int]:
    """Example function with PEP 484 type annotations.

        Args:
            team: Name of team.
            table: Initialized table of teams for a season.

        Returns:
            A tuple of the team's goals scored at home and away.

        """
    home_goals = table[table["Team"] == team]["GSH"].values[0]
    away_goals = table[table["Team"] == team]["GSA"].values[0]
    return home_goals, away_goals


def team_goals_conceded(team: str, table: pd.DataFrame) -> tuple[int, int]:
    """Example function with PEP 484 type annotations.

        Args:
            team: Name of team.
            table: Initialized table of teams for a season.

        Returns:
            A tuple of the team's goals conceded at home and away.

        """
    home_goalc = table[table["Team"] == team]["GCH"].values[0]
    away_goalc = table[table["Team"] == team]["GCA"].values[0]
    return home_goalc, away_goalc


def get_matches_played(team: str, table: pd.DataFrame) -> tuple[int, int]:
    """Example function with PEP 484 type annotations.

        Args:
            team: Name of team.
            table: Initialized table of teams for a season.

        Returns:
            A tuple of the number of matches a team played at home and away.

        """
    home_mp = table[table["Team"] == team]["HMP"].values[0]
    away_mp = table[table["Team"] == team]["AMP"].values[0]
    return home_mp, away_mp


def get_win_draw_loss(team: str, table: pd.DataFrame) -> tuple[int, int, int]:
    """Example function with PEP 484 type annotations.

        Args:
            team: Name of team.
            table: Initialized table of teams for a season.

        Returns:
            A tuple of the number of wins, losses and draws for a team.

        """
    wins = table[table["Team"] == team]["W"].values[0]
    draws = table[table["Team"] == team]["D"].values[0]
    losses = table[table["Team"] == team]["L"].values[0]
    return wins, draws, losses


def get_shot_on_target(team: str, table: pd.DataFrame) -> int:
    """Example function with PEP 484 type annotations.

        Args:
            team: Name of team.
            table: Initialized table of teams for a season.

        Returns:
            The return value. True for success, False otherwise.

        """
    sot = table[table["Team"] == team]["SoT"].values[0]
    return sot


def past_h2h_table(data: pd.DataFrame, home_team: str, away_team: str):
    """Isolates head-to-head matches for the home and away team.

        Args:
            data: Dataframe of matches played in a season.
            home_team: Name of home team.
            away_team: Name of away team.

        Returns:
            A dataframe containing all matches played by the home and away team against each other.

        """
    df = data[((data["HomeTeam"] == home_team) & (data["AwayTeam"] == away_team)) |
              ((data["HomeTeam"] == away_team) & (data["AwayTeam"] == home_team))]

    return df


def get_past_h2h(data: pd.DataFrame, home_team: str,
                 away_team: str, main_index: int) -> int:
    """Gets the result of the past head-to-head of both teams.

        Args:
            data: Dataframe of matches played in a season.
            home_team: Name of home team.
            away_team: Name of away team.
            main_index: Index of current game

        Returns:
            An integer. 0 indicating an away win, 1 for a draw and 2 for a home win.

        """
    past_table = past_h2h_table(data, home_team, away_team)
    indexes = past_table.index
    found = np.where(indexes == main_index)[0][0]
    if found == 0:
        past_result = 1
        return past_result
    else:
        past_match_index = indexes[found - 1]
        past_result = get_match_result(data, home_team, away_team, past_match_index)
        return past_result


def get_match_result(data: pd.DataFrame, home_team: str,
                     away_team: str, index: int) -> int:
    """Gets the result of a match at a given index.

        Args:
            data: Dataframe of matches played in a season.
            home_team: Name of home team.
            away_team: Name of away team.
            index: Index of current game

        Returns:
            An integer. 0 indicating an away win, 1 for a draw and 2 for a home win.

        """
    result = data.loc[index]["FTR"]
    hteam = data.loc[index]["HomeTeam"]
    ateam = data.loc[index]["AwayTeam"]
    if result == "H" and (hteam == home_team or ateam == away_team):
        return 2
    if result == "H" and hteam == away_team:
        return 0
    if result == "A" and (hteam == home_team or ateam == away_team):
        return 0
    if result == "A" and hteam == away_team:
        return 2
    if result == "D":
        return 1


def all_team_matches(data: pd.DataFrame, team: str):
    """Example function with PEP 484 type annotations.

        Args:
            data: Dataframe of matches played in a season..
            team: Name of team.

        Returns:
            The return value. True for success, False otherwise.

        """
    team_table = data[(data["HomeTeam"] == team) | (data["AwayTeam"] == team)]
    return team_table


def get_past_form(data: pd.DataFrame, team: str,
                  main_index: int) -> tuple[int, int, int]:
    """Example function with PEP 484 type annotations.

        Args:
            data: Dataframe of matches played in a season..
            team: Name of team.
            main_index: Index of current game

        Returns:
            A tuple of integers. 0 indicating an away win, 1 for a draw and 2 for a home win.

        """
    past_result1 = 1
    past_result2 = 1
    past_result3 = 1
    team_table = all_team_matches(data, team)
    indexes = team_table.index
    found = np.where(indexes == main_index)[0][0]
    if found == 0:
        return past_result1, past_result2, past_result3
    match_indexes = indexes[:found]
    if len(match_indexes) == 1:
        past_result1 = get_match_result(data, team, team, match_indexes[-1])
        return past_result1, past_result2, past_result3
    if len(match_indexes) == 2:
        past_result1 = get_match_result(data, team, team, match_indexes[-1])
        past_result2 = get_match_result(data, team, team, match_indexes[-2])
        return past_result1, past_result2, past_result3
    if len(match_indexes) >= 3:
        past_result1 = get_match_result(data, team, team, match_indexes[-1])
        past_result2 = get_match_result(data, team, team, match_indexes[-2])
        past_result3 = get_match_result(data, team, team, match_indexes[-3])
        return past_result1, past_result2, past_result3


def fetch_fixtures(url: str = None, filename: str = None) -> None:
    """Example function with PEP 484 type annotations.

        Args:
            url: The first parameter.
            filename: The second parameter.

        Returns:
            None.

        """
    download_url = url if url else 'https://www.football-data.co.uk/fixtures.csv'

    req = requests.get(download_url)
    filename = filename if filename else req.url[download_url.rfind('/') + 1:]

    with open(filename, 'wb') as f:
        for chunk in req.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def update_past_games(url: str = None, filename: str = None) -> bool:
    """Example function with PEP 484 type annotations.

        Args:
            url: The first parameter.
            filename: The second parameter.

        Returns:
            The return value. True for success, False otherwise.

        """
    download_url = url if url else "https://www.football-data.co.uk/mmz4281/2223/E0.csv"
    req = requests.get(download_url)
    filename = filename if filename else req.url[download_url.rfind('/') + 1:]

    with open(filename, 'wb') as f:
        for chunk in req.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return True


def split_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Example function with PEP 484 type annotations.

        Args:
            features: The first parameter.

        Returns:
            The return value. True for success, False otherwise.

        """
    team = features[["HomeTeam", "AwayTeam"]]
    features.drop(labels=["HomeTeam", "AwayTeam"], axis=1, inplace=True)
    return team, features

def merge_fixtures(x1, x2):
    merged = pd.concat([x1, x2])
    merged.reset_index(drop=True, inplace=True)
    return merged


def load_fixtures():
    df = pd.read_csv("/Users/chinedu/Desktop/Desktop - Ekeruche’s MacBook Pro/premier-league-main/fixtures.csv")
    df = df[df["Div"] == "E0"]
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def poisson(k, l):
    value = ((l ** k) * (2.71828 ** (-l))) / math.factorial(k)
    return value


def poisson_goal_distribution(att_str: tuple[float, float],
                              def_str: tuple[float, float],
                              league_ave: tuple[float, float],
                              k: int = 5):
    has, aas = att_str
    hds, ads = def_str
    hleague_ave, aleague_ave = league_ave

    hsr = has * ads * hleague_ave
    asr = aas * hds * aleague_ave
    # Initialize probability placeholder array for home and away team
    distribution = np.zeros((2, k))
    for i in range(k):
        distribution[0][i] = poisson(i, hsr)
        distribution[1][i] = poisson(i, asr)

    return distribution
