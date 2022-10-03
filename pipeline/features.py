import pipeline.utils as U


DATA_DIR = "/Users/chinedu/Desktop/premier-league-main/data/"


def main():

    df, file_num = U.load_data(DATA_DIR)
    datagen = U.generate_data(DATA_DIR)

    features = U.Features(data=df)
    labels = U.Labels(data=df, targets=["FTR"])

    features.make_feature("Home_curr_pos")
    features.make_feature("Away_curr_pos")
    features.make_feature("Home_W/MP")
    features.make_feature("Away_W/MP")
    features.make_feature("Home_D/MP")
    features.make_feature("Away_D/MP")
    features.make_feature("Home_L/MP")
    features.make_feature("Away_L/MP")
    features.make_feature("HSoT/90")
    features.make_feature("ASoT/90")
    features.make_feature("HAS")
    features.make_feature("AAS")
    features.make_feature("HDS")
    features.make_feature("ADS")
    features.make_feature("past_h2h")

    main_counter = 0
    for k in range(file_num):
        season = next(datagen)
        if k == file_num-1:
            fixtures = U.load_fixtures()
            season = U.merge_fixtures(season, fixtures)
        teams = U.get_teams_from_season(season)
        game_weeks = U.season_to_gameweeks(season)
        table = U.initialize_table(teams)
        for j, game_week in enumerate(game_weeks):
            for i in game_week.index.values:
                home_team: str = game_week.loc[i]["HomeTeam"]
                away_team: str = game_week.loc[i]["AwayTeam"]
                ftr: str = game_week.loc[i]["FTR"]
                fhg: str = game_week.loc[i]["FTHG"]
                fag: str = game_week.loc[i]["FTAG"]
                home_pos = U.get_position_on_table(home_team, table)
                away_pos = U.get_position_on_table(away_team, table)
                features.update_feature("Home_curr_pos", main_counter, home_pos)
                features.update_feature("Away_curr_pos", main_counter, away_pos)

                home_avg_goals, away_avg_goals = U.league_avg_goals_scored(table)
                home_gsh, _ = U.team_goals_scored(home_team, table)
                _, away_gsa = U.team_goals_scored(away_team, table)
                home_gch, _ = U.team_goals_conceded(home_team, table)
                _, away_gca = U.team_goals_conceded(away_team, table)
                home_matches_playedh, home_matches_playeda = U.get_matches_played(home_team, table)
                away_matches_playedh, away_matches_playeda = U.get_matches_played(away_team, table)
                total_home_matches = home_matches_playedh + home_matches_playeda
                total_away_matches = away_matches_playedh + away_matches_playeda
                has = (home_gsh / home_matches_playedh) / home_avg_goals
                aas = (away_gsa / away_matches_playeda) / away_avg_goals
                hds = (home_gch / home_matches_playedh) / away_avg_goals
                ads = (away_gca / away_matches_playeda) / home_avg_goals
                features.update_feature("HAS", main_counter, has)
                features.update_feature("AAS", main_counter, aas)
                features.update_feature("HDS", main_counter, hds)
                features.update_feature("ADS", main_counter, ads)

                home_results = U.get_win_draw_loss(home_team, table)
                away_results = U.get_win_draw_loss(away_team, table)
                features.update_feature("Home_W/MP", main_counter,
                                        home_results[0] / total_home_matches)
                features.update_feature("Home_D/MP", main_counter,
                                        home_results[1] / total_home_matches)
                features.update_feature("Home_L/MP", main_counter,
                                        home_results[2] / total_home_matches)
                features.update_feature("Away_W/MP", main_counter,
                                        away_results[0] / total_away_matches)
                features.update_feature("Away_D/MP", main_counter,
                                        away_results[1] / total_away_matches)
                features.update_feature("Away_L/MP", main_counter,
                                        away_results[2] / total_away_matches)

                home_sot = U.get_shot_on_target(home_team, table)
                away_sot = U.get_shot_on_target(away_team, table)
                features.update_feature("HSoT/90", main_counter,
                                        home_sot / total_home_matches)
                features.update_feature("ASoT/90", main_counter,
                                        away_sot / total_away_matches)

                past_h2h_result = U.get_past_h2h(df, home_team, away_team, main_counter)
                features.update_feature("past_h2h", main_counter,
                                        past_h2h_result)

                home_past_form = U.get_past_form(df, team=home_team, main_index=main_counter)
                away_past_form = U.get_past_form(df, team=away_team, main_index=main_counter)
                if k == file_num-1 and k > (len(df) - len(fixtures)):
                    dist = U.poisson_goal_distribution(att_str=(has, aas),
                                                        def_str=(hds, ads),
                                                        league_ave=(home_avg_goals, away_avg_goals))
                    print(home_team, away_team, dist)
                # print(home_team, away_team, ftr, fhg, fag, dist)
                features.update_feature("home_past_form1", main_counter,
                                        home_past_form[0])
                features.update_feature("home_past_form2", main_counter,
                                        home_past_form[1])
                features.update_feature("home_past_form3", main_counter,
                                        home_past_form[2])
                features.update_feature("away_past_form1", main_counter,
                                        away_past_form[0])
                features.update_feature("away_past_form2", main_counter,
                                        away_past_form[1])
                features.update_feature("away_past_form3", main_counter,
                                        away_past_form[2])
                main_counter += 1
            table = U.update_season_table(game_week, table)

    features.set_feature_value("B365H", df["B365H"])
    features.set_feature_value("B365D", df["B365D"])
    features.set_feature_value("B365A", df["B365A"])
    features.features.fillna(0, inplace=True)
    features.split_prediction(len(fixtures))
    labels.split_prediction(len(fixtures))
    features.to_csv()
    labels.to_csv()


if __name__ == "__main__":
    main()
