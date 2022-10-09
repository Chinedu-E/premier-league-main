import pandas as pd
import streamlit as st

st.title("Premier League Predictions")
st.markdown("Upcoming matches")

df = pd.read_csv("predictions/predictions.csv")
past_df = pd.read_csv("predictions/last_week_predictions.csv")

st.dataframe(df)

st.markdown("Past matches")

st.dataframe(past_df)