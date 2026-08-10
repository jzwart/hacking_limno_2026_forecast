# Data sources per variable

How to get a target timeseries into `forecast_workshop.ipynb`. Two paths:
**pull from published data** (a preset in Section 1) or **upload a CSV**
(`TARGET_MODE = "upload"`).

## CSV upload format (works for any variable)
A CSV with at least two columns:

| column | notes |
|---|---|
| a date/time column | first column; anything `pandas.to_datetime` can parse |
| a numeric value column | first numeric column after the time column |

See [`example_timeseries.csv`](example_timeseries.csv) for a minimal template
(a `date` column, a numeric value column, and a couple of blank rows showing that
gaps are fine).

The notebook resamples to **daily** and keeps gaps as `NaN` — Chronos-2 forecasts
from an incomplete history. Also set `UPLOAD_META` (site lat/lon and
`location_mode`) so the weather covariates are drawn from the right place — the
notebook now **errors out** if you leave lat/lon unset, so a forgotten edit fails
loudly instead of forecasting the wrong location's weather.

## Streamflow — published
[RivRetrieve](https://github.com/kratzert/RivRetrieve-Python) is a unified
interface to national streamflow archives. Swap the fetcher in the preset:

| Fetcher | Region | Notes |
|---|---|---|
| `USAFetcher` | USA (USGS NWIS) | **default preset** (Delaware/Montague, `01438500`); `gauge_id` = USGS site number |
| `UKEAFetcher` | UK (Environment Agency) | Thames/Kingston (commented backup preset) |
| `SloveniaFetcher`, `CzechFetcher`, ... | various | see RivRetrieve docs |

Set `source: "rivretrieve:<Fetcher>"`, the `gauge_id`, and `rr_variable`
(e.g. `discharge_daily_mean`). Use `location_mode: "delineate"` so the upstream
basin is fetched from the Global Watersheds API.

## Stream temperature — published or upload
- **USA:** USGS NWIS reports water temperature (and many other daily variables) at
  thousands of gauges. RivRetrieve focuses on discharge, so the simplest robust path
  is to pull a daily series yourself and **upload the CSV**.
- The easiest way to pull it in Python is USGS
  [**`dataretrieval`**](https://doi-usgs.github.io/dataretrieval-python/) (the Python
  port of the R [`dataRetrieval`](https://water.code-pages.usgs.gov/dataRetrieval/)
  package, whose [`readNWISdv`](https://water.code-pages.usgs.gov/dataRetrieval/reference/readNWISdv.html)
  is the daily-values equivalent). Use its `waterdata.get_daily()` for daily water
  temperature (parameter code `00010`, statistic `00003` = daily mean), then write
  the two columns this notebook expects:

  ```python
  import dataretrieval.waterdata as wd  # pip install dataretrieval

  df, _ = wd.get_daily(
      monitoring_location_id="USGS-01438500",  # note the "USGS-" prefix
      parameter_code="00010",                  # water temperature, degC
      statistic_id="00003",                    # 00003 = daily mean
      time="2010-01-01/2026-01-01",            # ISO date range
  )
  (df[["time", "value"]]
      .rename(columns={"value": "water_temperature_c"})
      .sort_values("time")
      .to_csv("my_timeseries.csv", index=False))
  ```

  `get_daily` returns a *long* frame (one row per day/statistic) plus a metadata
  object — filtering to a single `statistic_id` gives the one value column the
  notebook's heuristic needs. See the dataretrieval-python
  [WaterData demo](https://doi-usgs.github.io/dataretrieval-python/examples/WaterData_demo.html)
  for finding sites and other parameter/statistic codes.
- Use `location_mode: "delineate"` (river point) if you want basin-averaged
  weather, or `"buffer"` for a local box.

## Lake temperature — upload (recommended)
There is no single global published API bundled in this workshop, so for lakes
use `TARGET_MODE = "upload"` with your own lake surface water temperature (LSWT)
series and set a site point via `UPLOAD_META` with `location_mode: "buffer"`.

Candidate sources to build that CSV:
- **Satellite LSWT** — e.g. Copernicus/ESA CCI Lakes, or NASA/USGS Landsat &
  MODIS-derived lake temperature products.
- **GLTC** (Global Lake Temperature Collaboration) historical summer means.
- **In-situ** buoy/logger records from your own monitoring program.

## Weather covariates (all variables, automatic)
Pulled from [dynamical.org](https://dynamical.org) — no configuration needed:
- **History:** NOAA GEFS analysis (`noaa-gefs-analysis`).
- **Forecast:** NOAA GEFS 35-day (`noaa-gefs-forecast-35-day`) and ECMWF IFS ENS
  15-day (`ecmwf-ifs-ens-forecast-15-day-0-25-degree`).

Browse the full catalog — it is LLM-friendly — at
<https://stac.dynamical.org/catalog.json>.
