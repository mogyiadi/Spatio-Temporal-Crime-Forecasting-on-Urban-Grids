import math

import pandas as pd
from pyproj import Transformer


def make_daily_counts(df, x0, y0, side):
    cols = ["cmplnt_fr_dt", "cmplnt_fr_tm", "longitude", "latitude"]

    for a in cols:
        if a not in df.columns:
            raise ValueError("missing column " + a)

    if side <= 0:
        raise ValueError("grid side must be positive")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    vals = {}
    cells = []
    days = []

    for i in range(len(df)):
        d = df.iloc[i, df.columns.get_loc("cmplnt_fr_dt")]
        t = df.iloc[i, df.columns.get_loc("cmplnt_fr_tm")]
        lon = df.iloc[i, df.columns.get_loc("longitude")]
        lat = df.iloc[i, df.columns.get_loc("latitude")]
        ts = pd.to_datetime(str(d) + " " + str(t), errors="coerce")
        lon = pd.to_numeric(lon, errors="coerce")
        lat = pd.to_numeric(lat, errors="coerce")

        if pd.isna(ts):
            continue

        if pd.isna(lon):
            continue

        if pd.isna(lat):
            continue

        if lon == 0:
            continue

        if lat == 0:
            continue

        if lon < -180:
            continue

        if lon > 180:
            continue

        if lat < -90:
            continue

        if lat > 90:
            continue

        x, y = tr.transform(lon, lat)
        c = math.floor((x - x0) / side)
        r = math.floor((y - y0) / side)
        cell = "r" + str(r) + "_c" + str(c)
        day = ts.floor("D")
        key = (day, cell)

        if key in vals:
            vals[key] = vals[key] + 1
        else:
            vals[key] = 1

        if cell not in cells:
            cells.append(cell)

        if day not in days:
            days.append(day)

    if len(vals) == 0:
        out = pd.DataFrame()
        return out

    cells.sort()
    days.sort()
    all_days = pd.date_range(days[0], days[len(days) - 1], freq="D")
    out = {}

    for cell in cells:
        out[cell] = []

    for day in all_days:
        for cell in cells:
            key = (day, cell)
            n = 0

            if key in vals:
                n = vals[key]

            out[cell].append(n)

    out = pd.DataFrame(out, index=all_days)
    out.index.name = "day"
    return out


def save_counts(df, file_name):
    df.to_csv(file_name, index=True)
