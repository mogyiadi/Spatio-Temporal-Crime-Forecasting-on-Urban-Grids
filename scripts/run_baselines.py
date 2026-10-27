import pandas as pd

from src.evaluation import armse
from src.evaluation import last_value
from src.evaluation import past_mean


df = pd.read_csv("data/processed/nypd_sample_daily_counts.csv", index_col="day")

actual, out = last_value(df, 1)
score = armse(actual, out)
print("previous day armse", score)

actual, out = past_mean(df)
score = armse(actual, out)
print("past mean armse", score)

if len(df) > 7:
    actual, out = last_value(df, 7)
    score = armse(actual, out)
    print("previous week armse", score)
