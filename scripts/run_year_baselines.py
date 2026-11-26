import json

import numpy as np
import pandas as pd

from src.evaluation import armse


data = np.load("data/processed/nypd_2012_2013.npz")
counts = data["y"]
f = open("data/processed/nypd_2012_2013_meta.json", "r")
meta = json.load(f)
f.close()

history = 14
test_start = int(len(counts) * (0.70 + 0.15))

for h in [1, 7]:
    actual = []
    last = []
    week = []
    means = []
    days = []

    for origin in range(test_start + history - 1, len(counts) - h):
        target = origin + h
        actual.append(counts[target])
        last.append(counts[origin])
        week.append(counts[target - 7])
        avg = counts[:origin + 1].mean(axis=0)
        means.append(avg)
        days.append(meta["dates"][target])

    if len(days) == 0:
        raise ValueError("not enough test days")

    actual = pd.DataFrame(actual, index=days, columns=meta["cell_ids"])
    last = pd.DataFrame(last, index=days, columns=meta["cell_ids"])
    week = pd.DataFrame(week, index=days, columns=meta["cell_ids"])
    means = pd.DataFrame(means, index=days, columns=meta["cell_ids"])

    print("horizon", h)
    print("first target day", days[0])
    print("last target day", days[len(days) - 1])
    print("target days", len(days))
    print("last observed armse", armse(actual, last))
    print("previous week armse", armse(actual, week))
    print("historical mean armse", armse(actual, means))
