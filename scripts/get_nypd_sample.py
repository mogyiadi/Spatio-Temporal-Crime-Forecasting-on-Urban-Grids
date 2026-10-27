import json

import requests


url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
params = {}
params["$select"] = "cmplnt_num,cmplnt_fr_dt,cmplnt_fr_tm,latitude,longitude,ofns_desc"
params["$where"] = "cmplnt_fr_dt >= '2012-07-01T00:00:00' AND cmplnt_fr_dt < '2012-07-04T00:00:00'"
params["$limit"] = 5000

r = requests.get(url, params=params, timeout=60)
r.raise_for_status()
vals = r.json()

f = open("data/raw/nypd_2012_07_01_to_03.json", "w")
json.dump(vals, f)
f.close()

print("rows", len(vals))
