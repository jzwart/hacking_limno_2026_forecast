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

The notebook resamples to **daily** and keeps gaps as `NaN` — Chronos-2 forecasts
from an incomplete history. Also set `UPLOAD_META` (site lat/lon and
`location_mode`) so the weather covariates are drawn from the right place.

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
- **USA:** USGS NWIS reports water temperature at many gauges. RivRetrieve focuses
  on discharge, so the simplest robust path is to download a daily temperature
  series from [NWIS](https://waterdata.usgs.gov/nwis) and **upload the CSV**.
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
