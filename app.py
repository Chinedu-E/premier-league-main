import pandas as pd
import streamlit as st

st.title("Premier League Predictions")
st.markdown("Upcoming matches")

df = pd.read_csv("predictions.csv")

st.dataframe(df)