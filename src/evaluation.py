import math

import pandas as pd


def armse(actual, pred):
    if actual.shape != pred.shape:
        raise ValueError("shapes are different")

    if len(actual) == 0:
        raise ValueError("no rows")

    if len(actual.columns) == 0:
        raise ValueError("no columns")

    vals = []

    for j in range(len(actual.columns)):
        total = 0

        for i in range(len(actual)):
            a = actual.iloc[i, j]
            b = pred.iloc[i, j]
            dif = a - b
            total = total + dif * dif

        avg = total / len(actual)
        val = math.sqrt(avg)
        vals.append(val)

    total = 0

    for val in vals:
        total = total + val

    out = total / len(vals)
    return out


def last_value(df, lag):
    if lag <= 0:
        raise ValueError("lag must be positive")

    if len(df) <= lag:
        raise ValueError("not enough rows")

    vals = {}
    real = {}
    days = []

    for a in df.columns:
        vals[a] = []
        real[a] = []

    for i in range(lag, len(df)):
        day = df.index[i]
        days.append(day)

        for j in range(len(df.columns)):
            a = df.columns[j]
            x = df.iloc[i - lag, j]
            y = df.iloc[i, j]
            vals[a].append(x)
            real[a].append(y)

    out = pd.DataFrame(vals, index=days)
    actual = pd.DataFrame(real, index=days)
    return actual, out


def past_mean(df):
    if len(df) <= 1:
        raise ValueError("not enough rows")

    vals = {}
    real = {}
    days = []

    for a in df.columns:
        vals[a] = []
        real[a] = []

    for i in range(1, len(df)):
        day = df.index[i]
        days.append(day)

        for j in range(len(df.columns)):
            total = 0

            for k in range(i):
                x = df.iloc[k, j]
                total = total + x

            avg = total / i
            a = df.columns[j]
            y = df.iloc[i, j]
            vals[a].append(avg)
            real[a].append(y)

    out = pd.DataFrame(vals, index=days)
    actual = pd.DataFrame(real, index=days)
    return actual, out
