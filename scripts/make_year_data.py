import json
import math
import os

import numpy as np
import pandas as pd
from pyproj import Transformer


raw_path = "data/raw/nypd_2012_2013.jsonl"
check_path = "data/raw/nypd_2012_2013_complete.json"
data_path = "data/processed/nypd_2012_2013.npz"
meta_path = "data/processed/nypd_2012_2013_meta.json"

if os.path.exists(data_path):
    raise ValueError("processed data already exists")

if os.path.exists(meta_path):
    raise ValueError("metadata already exists")

if os.path.isdir("data/processed") == False:
    os.mkdir("data/processed")

f = open(check_path, "r")
check = json.load(f)
f.close()

days = pd.date_range("2012-07-01", "2013-06-30", freq="D")
dates = []
for day in days:
    dates.append(day.strftime("%Y-%m-%d"))
date_set = set(dates)

train_end = int(len(dates) * 0.70)
cutoff = dates[train_end]
tr = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
x0 = 900000
y0 = 110000
side = 6561.68
counts = {}
cells = set()
total = 0
valid = 0
skipped = 0

f = open(raw_path, "r")
for line in f:
    total = total + 1
    val = json.loads(line)
    day = val.get("cmplnt_fr_dt", "")[:10]

    try:
        lat = float(val.get("latitude", ""))
        lon = float(val.get("longitude", ""))
    except (TypeError, ValueError):
        skipped = skipped + 1
        continue

    if day not in date_set:
        skipped = skipped + 1
        continue
    if math.isfinite(lat) == False:
        skipped = skipped + 1
        continue
    if math.isfinite(lon) == False:
        skipped = skipped + 1
        continue
    if lat < 40.45 or lat > 40.93:
        skipped = skipped + 1
        continue
    if lon < -74.27 or lon > -73.70:
        skipped = skipped + 1
        continue

    x, y = tr.transform(lon, lat)
    col = math.floor((x - x0) / side)
    row = math.floor((y - y0) / side)
    cell = (row, col)
    key = (day, cell)

    if key not in counts:
        counts[key] = 0
    counts[key] = counts[key] + 1
    valid = valid + 1

    if day < cutoff:
        cells.add(cell)

f.close()

if total != check["records"]:
    raise ValueError("raw record count does not match completed download")

cells = sorted(cells)
if len(cells) == 0:
    raise ValueError("no training grid cells")

out = np.zeros((len(dates), len(cells)), dtype=np.float32)
cell_ids = []
grid_coords = []

for cell in cells:
    row, col = cell
    cell_ids.append("r" + str(row) + "_c" + str(col))
    grid_coords.append([row, col])

for i in range(len(dates)):
    for j in range(len(cells)):
        key = (dates[i], cells[j])
        if key in counts:
            out[i, j] = counts[key]

features = out.reshape(len(dates), len(cells), 1)
np.savez_compressed(data_path, x=features, y=out)

meta = {}
meta["cell_ids"] = cell_ids
meta["grid_coords"] = grid_coords
meta["dates"] = dates
meta["feature_names"] = ["crime_count"]
meta["grid_note"] = "prototype grid; cells observed in training period"
meta["grid_crs"] = "EPSG:2263"
meta["grid_origin_feet"] = [x0, y0]
meta["grid_side_feet"] = side
meta["coordinate_bounds"] = [-74.27, -73.70, 40.45, 40.93]
meta["raw_records"] = total
meta["valid_records"] = valid
meta["skipped_records"] = skipped
meta["outside_training_cells"] = valid - int(out.sum())

f = open(meta_path, "x")
json.dump(meta, f)
f.close()

print("records", total)
print("skipped records", skipped)
print("outside training cells", meta["outside_training_cells"])
print("days", len(dates))
print("cells", len(cells))
print("included crimes", int(out.sum()))
