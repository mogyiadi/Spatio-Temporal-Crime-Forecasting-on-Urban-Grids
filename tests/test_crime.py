import pandas as pd
import pytest

from src.evaluation import armse
from src.evaluation import last_value
from src.evaluation import past_mean
from src.preprocessing.crime import make_daily_counts


def test_counts_skip_bad_rows():
    df = pd.DataFrame(
        {
            "cmplnt_fr_dt": ["07/01/2012", "07/01/2012", "bad date", "07/03/2012"],
            "cmplnt_fr_tm": ["10:00:00", "11:00:00", "10:00:00", "10:00:00"],
            "longitude": [-74.00, -73.99, -74.00, 0],
            "latitude": [40.70, 40.70, 40.70, 40.70],
        }
    )

    out = make_daily_counts(df, 900000, 110000, 6561.68)

    assert out.to_numpy().sum() == 2


def test_counts_fill_empty_days():
    df = pd.DataFrame(
        {
            "cmplnt_fr_dt": ["07/01/2012", "07/03/2012"],
            "cmplnt_fr_tm": ["10:00:00", "10:00:00"],
            "longitude": [-74.00, -74.00],
            "latitude": [40.70, 40.70],
        }
    )

    out = make_daily_counts(df, 900000, 110000, 6561.68)

    assert len(out.index) == 3
    assert out.iloc[1].sum() == 0


def test_armse():
    a = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    b = pd.DataFrame({"a": [2, 2], "b": [4, 2]})

    out = armse(a, b)

    assert out == pytest.approx(1.5)


def test_last_value():
    df = pd.DataFrame({"a": [1, 2, 3]})

    actual, out = last_value(df, 1)

    assert actual.iloc[0, 0] == 2
    assert out.iloc[0, 0] == 1


def test_past_mean():
    df = pd.DataFrame({"a": [1, 3, 5]})

    actual, out = past_mean(df)

    assert actual.iloc[1, 0] == 5
    assert out.iloc[1, 0] == 2
