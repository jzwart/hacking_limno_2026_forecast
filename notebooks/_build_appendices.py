"""Build the two appendix notebooks. Run: `python _build_appendices.py`."""
import json
import pathlib


def build(cells, out_name, meta_title):
    nb = {
        "cells": [
            {
                "cell_type": t,
                "metadata": {},
                "source": s.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if t == "code" else {}),
            }
            for t, s in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = pathlib.Path(__file__).parent / out_name
    out.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {out} ({meta_title}) with {len(cells)} cells")


# ============================================================ zonal stats
zonal = []
zonal.append(("markdown", r"""# Appendix: area-weighted zonal statistics with `xvec`

The core notebook averages weather over the contributing area with a hard clip
(`.rio.clip(...).mean()`): every grid cell whose center falls in the polygon
counts equally, and partial cells at the boundary are in-or-out. For small basins
or coarse grids this is crude. [`xvec`](https://xvec.readthedocs.io/en/stable/zonal_stats.html)
computes **intersection-weighted** zonal statistics: each cell contributes in
proportion to the fraction of its area inside the polygon — a more faithful
basin average.

This appendix is optional and standalone; it reuses the same data sources as the
core notebook.
"""))
zonal.append(("code", r"""# numpy pinned first so later installs don't leave a half-upgraded numpy.
!pip install -q "numpy>=1.26,<2.1" truststore icechunk pystac rioxarray xvec exactextract geopandas requests"""))
zonal.append(("code", r'''# Route TLS through the OS trust store so https works behind TLS-inspecting
# proxies (e.g. USGS) that inject a self-signed root CA. No-op if not installed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import icechunk
import geopandas as gpd
import pandas as pd
import pystac
import requests
import rioxarray  # noqa: F401
import xarray as xr
import xvec  # noqa: F401  registers the .xvec accessor
from shapely.geometry import shape


def open_dynamical(dataset_id):
    """Open a dynamical.org dataset via its STAC `icechunk-https` asset."""
    catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
    asset = catalog.get_child(dataset_id).assets["icechunk-https"]
    repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None)

# Same reference basin as the core notebook (Delaware River at Montague, NJ).
LAT, LON = 41.3123, -74.7960
geo = requests.get(
    "https://mghydro.com/app/watershed_api",
    params={"lat": LAT, "lng": LON, "precision": "low"},
).json()["features"][0]["geometry"]
basin = shape(geo).simplify(0.01)
basin_gdf = gpd.GeoDataFrame(geometry=[basin], crs="EPSG:4326")
minx, miny, maxx, maxy = basin.bounds
'''))
zonal.append(("code", r'''# A recent slice of GEFS-analysis temperature & precipitation over the basin box.
ds = (
    open_dynamical("noaa-gefs-analysis")
    .sel(time=slice("2025-12-01", "2026-01-08"))
    [["temperature_2m", "precipitation_surface"]]
    .sel(latitude=slice(maxy + 0.5, miny - 0.5), longitude=slice(minx - 0.5, maxx + 0.5))
    .load()
)
ds["precipitation_surface"] *= 86_400  # -> mm/day
ds
'''))
zonal.append(("markdown", r"""## The grid vs. the basin

Before averaging, look at how the weather grid actually falls over the basin. Each
cell is one GEFS grid box; the shaded polygon is the delineated basin. **Green** cells
are the ones the hard clip in Method 1 keeps — `rio.clip` (default `all_touched=False`)
keeps a cell when its *center* falls inside the polygon — and grey cells are dropped.
Notice the boundary cells: several that clearly overlap the basin get dropped because
their center sits just outside, which is exactly what the area-weighted mean fixes.
"""))
zonal.append(("code", r'''import matplotlib.pyplot as plt
import numpy as np

lats = ds["latitude"].values
lons = ds["longitude"].values
# Cell edges: midpoints between centers, extended by half a step at each end.
def _edges(c):
    c = np.sort(c)
    step = np.diff(c).mean()
    return np.concatenate([[c[0] - step / 2], (c[:-1] + c[1:]) / 2, [c[-1] + step / 2]])
lat_edges, lon_edges = _edges(lats), _edges(lons)

fig, ax = plt.subplots(figsize=(8, 8))
# Grid lines at the cell edges.
for x in lon_edges:
    ax.axvline(x, color="0.7", lw=0.8, zorder=1)
for y in lat_edges:
    ax.axhline(y, color="0.7", lw=0.8, zorder=1)
# Basin polygon on top.
basin_gdf.plot(ax=ax, facecolor="tab:blue", edgecolor="navy", alpha=0.25, lw=2, zorder=2)
# Grid-cell centers (the points that carry the weather values). rio.clip keeps a
# cell when its CENTER falls inside the polygon (the default all_touched=False), so
# color each center by that test — the green cells are exactly the ones that survive
# .rio.clip(...).mean() in Method 1.
from shapely.geometry import Point
LON2D, LAT2D = np.meshgrid(lons, lats)
used = np.array([basin.contains(Point(x, y))
                 for x, y in zip(LON2D.ravel(), LAT2D.ravel())]).reshape(LON2D.shape)
ax.scatter(LON2D[~used], LAT2D[~used], s=14, color="0.6", zorder=3,
           label="dropped by clip")
ax.scatter(LON2D[used], LAT2D[used], s=28, color="tab:green", edgecolor="darkgreen",
           zorder=4, label=f"used by clip ({int(used.sum())} cells)")
ax.set_xlim(lon_edges[0], lon_edges[-1])
ax.set_ylim(lat_edges[0], lat_edges[-1])
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("GEFS grid cells over the basin polygon")
ax.set_aspect("equal"); ax.legend(loc="upper right")
fig.tight_layout()
'''))
zonal.append(("markdown", r"""## Method 1 — hard clip (what the core notebook does)"""))
zonal.append(("code", r'''clip_mean = (
    ds.rio.write_crs("EPSG:4326")
    .rio.clip([basin], crs="EPSG:4326")
    .mean(dim=("latitude", "longitude"))
    .to_dataframe()[["temperature_2m", "precipitation_surface"]]
)
clip_mean.head()
'''))
zonal.append(("markdown", r"""## Method 2 — intersection-weighted zonal mean with `xvec`

`xvec.zonal_stats` intersects each grid cell with the polygon and weights by the
overlapping area fraction. Boundary cells contribute partially instead of being
dropped or fully counted.
"""))
zonal.append(("code", r'''zonal_ds = ds.xvec.zonal_stats(
    basin_gdf.geometry,
    x_coords="longitude",
    y_coords="latitude",
    stats="mean",
    method="exactextract",   # area-weighted; falls back to "rasterize" if unavailable
)
weighted_mean = (
    zonal_ds.isel(geometry=0)
    .to_dataframe()[["temperature_2m", "precipitation_surface"]]
)
weighted_mean.head()
'''))
zonal.append(("markdown", r"""## Compare

The two basin-mean series differ most for precipitation and for small basins,
where boundary cells are a larger share of the total. Use the weighted version in
the core notebook by replacing `basin_daily`'s spatial mean with an `xvec` call.
"""))
zonal.append(("code", r'''import matplotlib.pyplot as plt

fig, (a, b) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
a.plot(clip_mean.index, clip_mean["temperature_2m"], label="hard clip", lw=2)
a.plot(weighted_mean.index, weighted_mean["temperature_2m"], label="xvec weighted", lw=2, ls="--")
a.set_ylabel("Temp [°C]"); a.legend(); a.set_title("Basin-mean temperature")
b.plot(clip_mean.index, clip_mean["precipitation_surface"], label="hard clip", lw=2)
b.plot(weighted_mean.index, weighted_mean["precipitation_surface"], label="xvec weighted", lw=2, ls="--")
b.set_ylabel("Precip [mm/day]"); b.legend(); b.set_title("Basin-mean precipitation")
fig.tight_layout()
'''))
zonal.append(("markdown", r"""## One-to-one comparison & difference stats

The overlaid timeseries hide how much the two methods actually diverge. A one-to-one
plot makes it obvious: points on the 1:1 line mean the methods agree, and spread off
the line is the disagreement the area-weighting introduces. The table quantifies it
(`weighted − hard clip`): mean bias, mean absolute difference, and the largest
single-step difference.
"""))
zonal.append(("code", r'''import pandas as pd

cmp = pd.DataFrame({
    "temp_clip": clip_mean["temperature_2m"], "temp_wt": weighted_mean["temperature_2m"],
    "precip_clip": clip_mean["precipitation_surface"], "precip_wt": weighted_mean["precipitation_surface"],
}).dropna()

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, clip, wt, lab, unit in [
    (a, "temp_clip", "temp_wt", "Temperature", "°C"),
    (b, "precip_clip", "precip_wt", "Precipitation", "mm/day"),
]:
    ax.scatter(cmp[clip], cmp[wt], s=18, alpha=0.6, edgecolor="none")
    lo = float(min(cmp[clip].min(), cmp[wt].min()))
    hi = float(max(cmp[clip].max(), cmp[wt].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel(f"hard clip [{unit}]"); ax.set_ylabel(f"xvec weighted [{unit}]")
    ax.set_title(lab); ax.set_aspect("equal"); ax.legend(loc="upper left")
fig.tight_layout()

def _stats(clip, wt):
    d = cmp[wt] - cmp[clip]
    denom = cmp[clip].abs().mean()
    return {
        "mean_bias": d.mean(),
        "mean_abs_diff": d.abs().mean(),
        "max_abs_diff": d.abs().max(),
        "rmse": (d ** 2).mean() ** 0.5,
        "mean_abs_diff_%": 100 * d.abs().mean() / denom if denom else float("nan"),
    }

diff_stats = pd.DataFrame({
    "temperature_2m": _stats("temp_clip", "temp_wt"),
    "precipitation_surface": _stats("precip_clip", "precip_wt"),
}).T
diff_stats
'''))
build(zonal, "appendix_zonal_stats.ipynb", "zonal stats")


# ============================================================ model deep dive
dive = []
dive.append(("markdown", r"""# Appendix: Chronos-2 — do covariates help, and what about data gaps?

This appendix runs on the **same real data as the core notebook** — observed
streamflow for the Delaware River at Montague, NJ (USGS-01438500) and basin-averaged
weather from dynamical.org — and asks two questions the core notebook only asserts:

1. **Do the weather covariates actually improve the forecast?** We hold out the last
   two weeks of the observed record as "truth", then forecast that window twice: once
   from the streamflow history **alone**, and once **conditioned on air temperature
   and precipitation** over the forecast window. Comparing both to the held-out truth
   shows what the covariates buy you.
2. **What happens when the target history has gaps?** Real gauge records have missing
   days. Chronos-2 ingests NaNs directly rather than failing — we punch increasing
   gaps into the *real* Montague history and watch skill degrade gracefully.

> **Backtest, not a live forecast.** Because the forecast window is in the past, we
> can use the dynamical.org **analysis** (observed weather) as the future-known
> covariate. That is an *idealized* upper bound on covariate value — a real forecast
> uses an imperfect weather *forecast* (§4–5 of the core notebook), so expect the
> live benefit to be smaller than what you see here.
"""))
dive.append(("code", r"""# CPU torch first so the chronos install doesn't pull the large CUDA torch build;
# then numpy pinned so later installs don't leave a half-upgraded numpy. If a numpy
# ImportError appears on Colab, restart & re-run. (Same stack as the core notebook,
# plus RivRetrieve for the observed streamflow.)
!pip install -q "torch<2.10" --index-url https://download.pytorch.org/whl/cpu
!pip install -q "numpy>=1.26,<2.1" "chronos-forecasting[extras]>=2.2" truststore \
    icechunk pystac rioxarray geopandas requests pandas matplotlib \
    "git+https://github.com/kratzert/RivRetrieve-Python.git"
# Colab preinstalls CUDA torchvision/torchaudio; drop them so their C++ ops don't
# clash with our CPU torch and break transformers' lazy import. Chronos-2 CPU
# inference needs neither. (Harmless if they aren't installed.)
!pip uninstall -q -y torchvision torchaudio"""))
dive.append(("code", r'''# Route TLS through the OS trust store so https works behind TLS-inspecting proxies
# (e.g. USGS) that inject a self-signed root CA. No-op if truststore isn't installed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Same reference site as the core notebook: Delaware River at Montague, NJ.
LAT, LON = 41.3123, -74.7960
GAUGE_ID = "01438500"           # USGS site number
HORIZON = 14                    # forecast/backtest window (days)
HISTORY_DAYS = 365 * 6          # context window fed to the model
COVS = ["temperature_2m", "precipitation_surface"]  # future-known covariates

# Backtest issue date (t0): forecast the HORIZON days AFTER this. It must be far
# enough in the past that both the streamflow record and the GEFS analysis cover the
# forecast window. Change it to test a different event.
INIT = pd.Timestamp("2025-06-01")
'''))
dive.append(("markdown", r"""## Load the real data (streamflow + basin weather)

Exactly the sources the core notebook uses: observed discharge via
[RivRetrieve](https://github.com/kratzert/RivRetrieve-Python), the upstream basin
polygon from the [Global Watersheds API](https://mghydro.com/watersheds/), and
basin-averaged temperature & precipitation from the dynamical.org **GEFS analysis**.
"""))
dive.append(("code", r'''import icechunk
import geopandas as gpd  # noqa: F401  (ensures the geospatial stack is importable)
import pystac
import requests
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr
from shapely.geometry import shape

# --- observed streamflow (m3/s) ---
import rivretrieve
obs = rivretrieve.USAFetcher().get_data(
    gauge_id=GAUGE_ID,
    variable="discharge_daily_mean",
    start_date="2015-01-01",
    end_date=str((INIT + pd.Timedelta(days=HORIZON)).date()),
)["discharge_daily_mean"]
obs.index = pd.to_datetime(obs.index)
obs.name = "streamflow"

# --- upstream basin polygon (for clipping the weather grid) ---
geo = requests.get(
    "https://mghydro.com/app/watershed_api",
    params={"lat": LAT, "lng": LON, "precision": "low"},
).json()["features"][0]["geometry"]
area = shape(geo).simplify(0.01)
minx, miny, maxx, maxy = area.bounds


def open_dynamical(dataset_id):
    """Open a dynamical.org dataset via its STAC `icechunk-https` asset."""
    catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
    asset = catalog.get_child(dataset_id).assets["icechunk-https"]
    repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None)


# --- basin-averaged weather over [history .. end of forecast window] ---
hist_start = INIT - pd.Timedelta(days=HISTORY_DAYS)
meteo = (
    open_dynamical("noaa-gefs-analysis")
    .sel(time=slice(hist_start, INIT + pd.Timedelta(days=HORIZON)))
    [COVS]
    .sel(latitude=slice(maxy + 0.5, miny - 0.5), longitude=slice(minx - 0.5, maxx + 0.5))
    .rio.clip([area], crs="EPSG:4326")
    .mean(dim=("latitude", "longitude"))
    .resample(time="1D").mean()
    .load()
)
meteo["precipitation_surface"] *= 86_400  # surface flux -> mm/day
print("Loaded", int(obs.notna().sum()), "streamflow days and",
      meteo.sizes["time"], "days of basin weather.")
'''))
dive.append(("markdown", r"""## Assemble the backtest: history, held-out truth, future covariates

We forecast in **log space** (streamflow is strictly positive and right-skewed, like
the core notebook), and split the record at `INIT`: everything before is context,
the `HORIZON` days from `INIT` on are the held-out truth.
"""))
dive.append(("code", r'''# One daily frame: target + covariates, aligned on the calendar.
df = pd.DataFrame({
    "streamflow": obs,
    "temperature_2m": meteo["temperature_2m"].to_pandas(),
    "precipitation_surface": meteo["precipitation_surface"].to_pandas(),
})
df.index.name = "timestamp"

# History (context) and the future window we will forecast.
hist = df.loc[hist_start:INIT - pd.Timedelta(days=1)].reset_index()
hist = hist.dropna(subset=COVS)          # covariates must be present; target gaps OK
future_index = pd.date_range(INIT, periods=HORIZON, freq="D")
future = df.reindex(future_index).rename_axis("timestamp").reset_index()
truth = future.set_index("timestamp")["streamflow"]

assert future[COVS].notna().all().all(), "Missing covariate in the forecast window."
print(f"Context: {len(hist)} days ({hist['timestamp'].min().date()} -> "
      f"{hist['timestamp'].max().date()}), target gaps: {hist['streamflow'].isna().sum()}")
print(f"Forecasting {HORIZON} days from {INIT.date()}; "
      f"truth has {truth.notna().sum()}/{HORIZON} observed days.")
'''))
dive.append(("markdown", r"""## Load Chronos-2"""))
dive.append(("code", r'''from chronos import Chronos2Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
print("Chronos-2 loaded on", device)
'''))
dive.append(("markdown", r"""## Forecast helper (covariates optional)

One helper drives both experiments. Pass `future_df` (temperature + precipitation
over the horizon) to condition on the weather; omit it for a target-only forecast.
Streamflow is transformed with `log1p` on the way in and `expm1` on the way out, so
the returned quantiles are in physical units (m³/s).
"""))
dive.append(("code", r'''_Q = [0.1, 0.5, 0.9]


def chronos_forecast(context_hist, future_df=None):
    """Forecast HORIZON days of streamflow. With future_df -> covariate-conditioned.

    context_hist: history frame with columns [timestamp, streamflow, *COVS].
    future_df:    None (target-only) or frame [timestamp, *COVS] over the horizon.
    Returns a DataFrame indexed by date with columns q0.1 / q0.5 / q0.9 (m3/s).
    """
    ctx = context_hist.copy()
    ctx["streamflow"] = np.log1p(ctx["streamflow"])   # forecast in log space
    ctx["item"] = "series"
    cols = ["item", "timestamp", "streamflow"] + (COVS if future_df is not None else [])
    kwargs = dict(
        prediction_length=HORIZON, quantile_levels=_Q,
        id_column="item", timestamp_column="timestamp", target="streamflow",
    )
    if future_df is not None:
        kwargs["future_df"] = future_df.assign(item="series")[["item", "timestamp", *COVS]]
    pred = chronos.predict_df(ctx[cols], **kwargs).sort_values("timestamp")
    out = {f"q{q}": np.expm1(pred[str(q)].to_numpy()) for q in _Q}  # back to m3/s
    return pd.DataFrame(out, index=pd.DatetimeIndex(pred["timestamp"], name="timestamp"))
'''))
dive.append(("markdown", r"""## Experiment 1 — with vs. without covariates

Same history, same held-out truth; the only difference is whether Chronos-2 sees the
future weather. We score the median forecast with RMSE and MAE against the observed
truth.
"""))
dive.append(("code", r'''fc_bare = chronos_forecast(hist)                       # target history only
fc_cov = chronos_forecast(hist, future_df=future)      # + temperature & precip

def _score(fc):
    d = fc["q0.5"].to_numpy() - truth.to_numpy()
    return {"RMSE": float(np.sqrt(np.nanmean(d ** 2))),
            "MAE": float(np.nanmean(np.abs(d)))}

skill_cov = pd.DataFrame({
    "target only": _score(fc_bare),
    "+ temp & precip": _score(fc_cov),
}).T
skill_cov["RMSE improvement %"] = (
    100 * (skill_cov.loc["target only", "RMSE"] - skill_cov["RMSE"])
    / skill_cov.loc["target only", "RMSE"]
)
skill_cov
'''))
dive.append(("code", r'''# Plot both forecasts against the observed record and the held-out truth.
recent = df["streamflow"].loc[INIT - pd.Timedelta(days=30):INIT - pd.Timedelta(days=1)]

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(recent.index, recent.values, color="black", lw=2, label="observed history")
ax.plot(truth.index, truth.values, color="black", lw=2.5, ls=":", label="held-out truth")
ax.axvline(INIT, color="gray", ls="--", alpha=0.6)
for fc, color, label in [(fc_bare, "tab:orange", "target only"),
                         (fc_cov, "tab:blue", "+ temp & precip")]:
    ax.plot(fc.index, fc["q0.5"], color=color, lw=2.5, label=f"{label} (median)")
    ax.fill_between(fc.index, fc["q0.1"], fc["q0.9"], color=color, alpha=0.15)
ax.set_ylabel("streamflow [m³/s]")
ax.set_title(f"Delaware R. at Montague — {HORIZON}-day backtest from {INIT.date()}")
ax.legend(loc="best")
fig.tight_layout()
'''))
dive.append(("markdown", r"""Whether the covariates help depends on the window: during
a rain-driven rise or recession the weather carries real predictive signal and the
conditioned forecast tracks the truth better; during flat baseflow the streamflow
history alone is already enough and the two are close. Try a different `INIT` (e.g. a
storm week) to see the gap widen. Remember this is the *idealized* case — the future
weather here is analysis, not a forecast.
"""))
dive.append(("markdown", r"""## Experiment 2 — gaps in the target history

Real gauge records have missing days. We drop an increasing fraction of the *recent*
year of the **real** Montague history and re-forecast (target-only). Chronos-2 accepts
the NaNs directly — no imputation needed.
"""))
dive.append(("code", r'''def with_gaps(context_hist, frac, seed):
    r = np.random.default_rng(seed)
    s = context_hist.copy()
    recent = s.index[-365:]
    drop = r.choice(recent, size=int(frac * len(recent)), replace=False)
    s.loc[drop, "streamflow"] = np.nan
    return s

records = []
for frac in [0.0, 0.1, 0.3, 0.5, 0.7]:
    gapped = with_gaps(hist.set_index("timestamp"), frac, seed=1).reset_index()
    fc = chronos_forecast(gapped)                       # native NaN handling
    d = fc["q0.5"].to_numpy() - truth.to_numpy()
    records.append({"gap_frac": frac, "Chronos-2 RMSE": float(np.sqrt(np.nanmean(d ** 2)))})
skill_gaps = pd.DataFrame(records).set_index("gap_frac")
skill_gaps
'''))
dive.append(("code", r'''fig, ax = plt.subplots(figsize=(9, 5))
skill_gaps.plot(marker="o", ax=ax, legend=False)
ax.set_xlabel("Fraction of last-year history missing")
ax.set_ylabel(f"{HORIZON}-day forecast RMSE [m³/s]")
ax.set_title("Chronos-2 skill vs. gaps in the real Montague history")
fig.tight_layout()
'''))
dive.append(("markdown", r"""## Takeaways

- **Covariates are optional but often worthwhile.** Feeding future-known weather can
  sharpen the forecast, most visibly during precipitation-driven events; on quiet
  baseflow days the streamflow history alone already captures the dynamics. This
  backtest uses *observed* weather as the future covariate, so it's an upper bound —
  a live forecast (core notebook §5) conditions on an imperfect weather *forecast*.
- **Chronos-2 ingests NaNs natively** and degrades gracefully as target gaps grow
  rather than failing — which is why the core notebook keeps target gaps and only
  requires the *covariates* to be present.
- For a very gappy or short record, expect wider uncertainty; a longer history window
  (`HISTORY_DAYS`) gives the model more seasonal context.
"""))
build(dive, "appendix_model_deep_dive.ipynb", "model deep dive")
