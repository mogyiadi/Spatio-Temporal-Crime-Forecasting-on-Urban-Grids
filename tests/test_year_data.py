import json
import runpy
from pathlib import Path

import numpy as np

from src.data.window_dataset import CrimeWindowDataModule
from src.data.window_dataset import WindowConfig

PROJECT = Path(__file__).resolve().parents[1]


def test_year_data_matches_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    vals = [
        {"cmplnt_fr_dt": "2012-07-01T00:00:00", "latitude": "40.70", "longitude": "-74.00"},
        {"cmplnt_fr_dt": "2012-07-02T00:00:00", "latitude": "40.70", "longitude": "-73.97"},
        {"cmplnt_fr_dt": "2013-06-15T00:00:00", "latitude": "40.70", "longitude": "-73.71"},
        {"cmplnt_fr_dt": "2013-06-16T00:00:00"},
    ]

    f = open("data/raw/nypd_2012_2013.jsonl", "w")
    for val in vals:
        f.write(json.dumps(val) + "\n")
    f.close()

    f = open("data/raw/nypd_2012_2013_complete.json", "w")
    json.dump({"records": len(vals)}, f)
    f.close()

    runpy.run_path(str(PROJECT / "scripts" / "make_year_data.py"))
    data = np.load("data/processed/nypd_2012_2013.npz")
    f = open("data/processed/nypd_2012_2013_meta.json", "r")
    meta = json.load(f)
    f.close()

    assert data["x"].shape == (365, 2, 1)
    assert data["y"].shape == (365, 2)
    assert int(data["y"].sum()) == 2
    assert meta["skipped_records"] == 1
    assert meta["outside_training_cells"] == 1

    cfg = WindowConfig()
    module = CrimeWindowDataModule(
        "data/processed/nypd_2012_2013.npz",
        "data/processed/nypd_2012_2013_meta.json",
        cfg,
    )
    split = module.get_split("test")
    x, y = split[0]
    assert x.shape == (14, 2, 1)
    assert y.shape == (2,)
    assert split.target_dates[0] > split.origin_dates[0]


def test_year_baselines_use_forecast_origin(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    counts = np.arange(365, dtype=np.float32).reshape(365, 1)
    np.savez_compressed("data/processed/nypd_2012_2013.npz", y=counts)
    dates = []
    for i in range(365):
        dates.append(str(i))
    f = open("data/processed/nypd_2012_2013_meta.json", "w")
    json.dump({"dates": dates, "cell_ids": ["r0_c0"]}, f)
    f.close()

    runpy.run_path(str(PROJECT / "scripts" / "run_year_baselines.py"))
    out = capsys.readouterr().out
    assert "last observed armse 1.0" in out
    assert "last observed armse 7.0" in out
    assert out.count("previous week armse 7.0") == 2
