import json
import os
import time

import requests


url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
path = "data/raw/nypd_2012_2013.jsonl"
check_path = "data/raw/nypd_2012_2013_complete.json"

if os.path.isdir("data") == False:
    os.mkdir("data")

if os.path.isdir("data/raw") == False:
    os.mkdir("data/raw")

if os.path.exists(check_path):
    raise ValueError("download record already exists")

dates = [
    "2012-07-01", "2012-08-01", "2012-09-01", "2012-10-01",
    "2012-11-01", "2012-12-01", "2013-01-01", "2013-02-01",
    "2013-03-01", "2013-04-01", "2013-05-01", "2013-06-01",
    "2013-07-01",
]

saved = {}
if os.path.exists(path):
    f = open(path, "r")
    for line in f:
        val = json.loads(line)
        month = val["cmplnt_fr_dt"][:7]
        if month not in saved:
            saved[month] = 0
        saved[month] = saved[month] + 1
    f.close()

f = open(path, "a")
total = 0

for i in range(len(dates) - 1):
    start = dates[i] + "T00:00:00"
    end = dates[i + 1] + "T00:00:00"
    where = "cmplnt_fr_dt >= '" + start + "' AND cmplnt_fr_dt < '" + end + "'"

    params = {}
    params["$select"] = "count(*)"
    params["$where"] = where
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 429:
            if r.status_code < 500:
                break
        if attempt < 3:
            time.sleep(3)
    r.raise_for_status()
    expected = int(r.json()[0]["count"])

    month = dates[i][:7]
    offset = 0
    if month in saved:
        offset = saved[month]
    if offset > expected:
        raise ValueError("saved records exceed the current source count")
    limit = 10000
    while offset < expected:
        params = {}
        params["$select"] = "cmplnt_fr_dt,latitude,longitude"
        params["$where"] = where
        params["$order"] = ":id"
        params["$limit"] = limit
        params["$offset"] = offset
        for attempt in range(4):
            r = requests.get(url, params=params, timeout=120)
            if r.status_code != 429:
                if r.status_code < 500:
                    break
            if attempt < 3:
                time.sleep(3)
        r.raise_for_status()
        vals = r.json()

        if len(vals) == 0:
            raise ValueError("download ended before the expected count")

        for val in vals:
            f.write(json.dumps(val) + "\n")

        offset = offset + len(vals)

    if offset != expected:
        raise ValueError("download count does not match")

    total = total + offset

f.close()
f = open(check_path, "x")
json.dump({"records": total}, f)
f.close()
print("downloaded records", total)
