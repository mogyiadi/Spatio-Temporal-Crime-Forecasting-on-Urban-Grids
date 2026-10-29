import json
import os

import pandas as pd

from src.preprocessing.crime import make_daily_counts
from src.preprocessing.crime import save_counts


f = open("data/raw/nypd_2012_07_01_to_03.json", "r")
vals = json.load(f)
f.close()

df = pd.DataFrame(vals)

# sample grid only
x0 = 900000
y0 = 110000
side = 6561.68

out = make_daily_counts(df, x0, y0, side)

if os.path.isdir("data") == False:
    os.mkdir("data")

if os.path.isdir("data/processed") == False:
    os.mkdir("data/processed")

save_counts(out, "data/processed/nypd_sample_daily_counts.csv")

print("days", len(out))
print("cells", len(out.columns))
print("crimes", out.to_numpy().sum())
