"""Generate notebooks/forecast_workshop.ipynb from a list of (type, source) cells.

This builder exists so the notebook can be authored as readable Python/markdown
strings and assembled into valid .ipynb JSON. Run once: `python _build_core.py`.
It is a dev tool, not part of the workshop; safe to delete after building.
"""
import json
import pathlib

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


# ---------------------------------------------------------------- title
md(r"""# Zero-shot water forecasting with open weather data

**AEMON-J / DSOS "Hacking Limnology" 2026 — Day 2: Climate Data**
Workshop: *How to access petabytes of weather forecasts from your laptop*
(Jake Zwart, USGS · Alden Keefe Sampson, Dynamical)

You will generate a short-range forecast of a **water variable** — streamflow,
stream temperature, lake temperature, or **your own uploaded timeseries** — using
[TiRex-2](https://github.com/NX-AI/tirex-2), a zero-shot time-series foundation
model, conditioned on open ensemble weather forecasts from
[dynamical.org](https://dynamical.org). At the end you **submit** your forecast
for an eventual global evaluation.

> **Run it as-is first.** With no edits, this notebook reproduces a working
> forecast for the River Thames at Kingston. Then change the single **config
> block** in Section 1 to forecast *your* site.

### How to run
- **Google Colab (recommended):** click the badge in the README, or open this
  file via `colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/forecast_workshop.ipynb`.
  No install needed beyond the first cell. A Google account helps for submission
  but is not required to run.
- **Locally with `uv`** (no Google account, better for slow connections): see
  `docs/local_setup.md`.
""")

# ---------------------------------------------------------------- install
md(r"""## 0. Setup

Installs the forecasting stack. On Colab this takes ~1–2 min.

*Optional:* set a Hugging Face token as the Colab secret `HF_TOKEN` to avoid
occasional rate limits when downloading model weights (TiRex-2 is public, so this
is not required).

> ⚠️ **On Colab the runtime restarts once, automatically, after this cell.** The
> install changes `numpy`, and Colab has already imported the old one — the
> restart loads a clean environment. This is expected: **re-run from the top**
> after it restarts (the install is cached, so it will be quick), then continue.
>
> We install the **CPU build of PyTorch** first. TiRex-2 depends on `flashrnn`,
> whose CUDA kernels require a newer GPU than Colab's free tier provides and
> otherwise fail to build; the CPU torch build sidesteps that and is fast enough
> for a single-site forecast. If you have a capable GPU locally, see
> `docs/local_setup.md` for the GPU path.""")

code(r'''import os
import subprocess
import sys

# A disk sentinel survives the kernel restart (kernel *state* does not), so we
# install + restart exactly once even under "Run all" / repeated top-to-bottom.
# It is written ONLY after a fully successful install, so a failure re-runs clean.
_SENTINEL = "/tmp/.forecast_workshop_installed"


def _pip(*pkgs, extra_args=()):
    """pip install as a subprocess so we can check the exit code and see output."""
    cmd = [sys.executable, "-m", "pip", "install", *extra_args, *pkgs]
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            "Dependency install failed (see output above). Nothing was cached, so "
            "re-run this cell after resolving the offending package."
        )


if not os.path.exists(_SENTINEL):
    # 1) CPU PyTorch first. This avoids compiling flashrnn's CUDA kernels (a
    #    tirex-2 dependency) which fail to build on Colab's default GPU.
    _pip("torch<2.10", extra_args=("--index-url", "https://download.pytorch.org/whl/cpu"))
    # 2) numpy pinned so the heavier installs below don't half-upgrade it (the
    #    classic "cannot import name '_center' from numpy._core.umath" crash).
    _pip("numpy>=1.26,<2.1")
    # 3) the forecasting stack.
    _pip("dynamical-catalog", "rioxarray", "cartopy", "geopandas", "tirex-2",
         "git+https://github.com/kratzert/RivRetrieve-Python.git")
    open(_SENTINEL, "w").close()

    # On Colab, restart the runtime once so the freshly installed numpy is the
    # one that gets imported. Re-run from the top after the restart.
    try:
        import google.colab  # noqa: F401
        import IPython
        print("Install complete — restarting the Colab runtime. "
              "Re-run from the top when it comes back.")
        IPython.Application.instance().kernel.do_shutdown(restart=True)
    except ImportError:
        print("Local run — no restart needed; continue to the next cell.")
else:
    print("Dependencies already installed — skipping install and restart.")''')

code(r"""import json
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dynamical_catalog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401  registers the .rio accessor
from shapely.geometry import shape

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update(
    {"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12}
)

IN_COLAB = "google.colab" in str(get_ipython())  # noqa: F821
print("Running in Colab" if IN_COLAB else "Running locally")""")

# ---------------------------------------------------------------- config
md(r"""## 1. Choose your target  ⬅️ **the only block most people edit**

Pick a **preset** (published data pulled automatically) or **upload a CSV** of
your own timeseries. Everything downstream reads from this block.

### Option A — a published preset
Set `TARGET_MODE = "published"` and choose a `PRESET` key below. The default
`thames_streamflow` reproduces the reference forecast. Commented presets show how
to point at other gauges / sources.

### Option B — bring your own data
Set `TARGET_MODE = "upload"`, then run the upload cell. Your CSV needs two
columns: a date/timestamp column and a numeric value column. Gaps are fine —
TiRex-2 forecasts from an incomplete history (we explore this in
`appendix_model_deep_dive.ipynb`).""")

code(r'''# ============================ EDIT HERE ============================
TARGET_MODE = "published"      # "published"  or  "upload"
PRESET = "thames_streamflow"   # used only when TARGET_MODE == "published"

INIT_TIME = pd.Timestamp("2026-01-08")  # forecast issue date (t0)
FORECAST_DAYS = 15                       # horizon

# --- your details, baked into the submission file (see Section 7) ---
PARTICIPANT = {
    "name": "",            # e.g. "Ada Lovelace"
    "affiliation": "",
    "email": "",
    "orcid": "",           # optional
    "coauthor_optin": False,  # True = consent to co-authorship on a global-eval output
}
# ==================================================================

# Published presets. Each describes WHERE the target obs come from and HOW to
# locate the contributing area for basin-averaged weather.
#   location_mode: "delineate" (point -> upstream basin, for rivers)
#                  "buffer"    (point -> square box, for lakes/points)
PRESETS = {
    # --- default: reference forecast, reproduces the example notebook ---
    "thames_streamflow": {
        "variable": "streamflow",
        "units": "m3/s",
        "source": "rivretrieve:UKEAFetcher",
        "gauge_id": "8496ce69-482c-406a-a2f0-ac418ef8f099",
        "rr_variable": "discharge_daily_mean",
        "lat": 51.4155, "lon": -0.3076,
        "location_mode": "delineate",
    },
    # --- US streamflow via RivRetrieve USGS fetcher (uncomment & edit) ---
    # "usgs_streamflow": {
    #     "variable": "streamflow", "units": "m3/s",
    #     "source": "rivretrieve:USAFetcher",
    #     "gauge_id": "06892350",          # a USGS site number
    #     "rr_variable": "discharge_daily_mean",
    #     "lat": 39.05, "lon": -94.60,
    #     "location_mode": "delineate",
    # },
    # --- lake surface temperature: easiest path is CSV upload ---------
    #     There is no single global published API bundled here, so for lakes
    #     use TARGET_MODE = "upload" with your own LSWT timeseries, and set a
    #     point below via UPLOAD_META (buffer box for weather). See
    #     docs/data_sources.md for candidate sources (e.g. satellite LSWT).
}
'''
)

code(r'''# Upload path — run this cell only if TARGET_MODE == "upload".
# Provides `uploaded_obs` (a daily pandas Series) and target metadata.
UPLOAD_META = {
    "variable": "custom",   # "streamflow" | "stream_temperature" | "lake_temperature" | "custom"
    "units": "",            # e.g. "m3/s", "degC"
    "lat": 51.4155,          # site location -> weather covariates
    "lon": -0.3076,
    "location_mode": "buffer",  # "buffer" (point+box) or "delineate" (river point)
    "buffer_deg": 0.25,          # half-width of weather box when location_mode=="buffer"
}

uploaded_obs = None
if TARGET_MODE == "upload":
    if IN_COLAB:
        from google.colab import files
        up = files.upload()
        csv_name = next(iter(up))
        raw = pd.read_csv(csv_name)
    else:
        # Local: set the path to your CSV.
        raw = pd.read_csv("my_timeseries.csv")

    # Heuristic column detection: first datetime-parseable col = time, first
    # numeric col = value. Override by renaming your columns explicitly.
    time_col = raw.columns[0]
    val_col = next(c for c in raw.columns[1:] if pd.api.types.is_numeric_dtype(raw[c]))
    uploaded_obs = (
        raw.assign(_t=pd.to_datetime(raw[time_col]))
        .set_index("_t")[val_col]
        .sort_index()
        .resample("1D").mean()   # coerce to daily; leaves gaps as NaN
    )
    uploaded_obs.name = UPLOAD_META["variable"]
    print(f"Loaded {uploaded_obs.notna().sum()} daily values "
          f"({uploaded_obs.index.min().date()} to {uploaded_obs.index.max().date()}), "
          f"{uploaded_obs.isna().sum()} gaps.")
'''
)

code(r'''# Resolve the active target config from whichever mode is selected.
if TARGET_MODE == "published":
    cfg = dict(PRESETS[PRESET])
elif TARGET_MODE == "upload":
    cfg = dict(UPLOAD_META)
else:
    raise ValueError("TARGET_MODE must be 'published' or 'upload'")

TARGET_LAT, TARGET_LON = cfg["lat"], cfg["lon"]
VARIABLE = cfg["variable"]
UNITS = cfg.get("units", "")
print(f"Target: {VARIABLE} [{UNITS}] at ({TARGET_LAT}, {TARGET_LON}) "
      f"via {TARGET_MODE}; horizon {FORECAST_DAYS} d from {INIT_TIME.date()}")
'''
)

# ---------------------------------------------------------------- location
md(r"""## 2. Define the contributing area

Weather covariates are averaged over an area around your site. For **rivers**
(`location_mode="delineate"`) we fetch the upstream drainage polygon from the
[Global Watersheds API](https://mghydro.com/watersheds/). For **lakes / points**
(`location_mode="buffer"`) we use a square box around the point. Advanced users:
`appendix_zonal_stats.ipynb` shows area-weighted zonal stats with `xvec`.""")

code(r'''from shapely.geometry import box

location_mode = cfg.get("location_mode", "buffer")

if location_mode == "delineate":
    resp = requests.get(
        "https://mghydro.com/app/watershed_api",
        params={"lat": TARGET_LAT, "lng": TARGET_LON, "precision": "low"},
    ).json()
    area = shape(resp["features"][0]["geometry"]).simplify(0.01)
else:
    b = cfg.get("buffer_deg", 0.25)
    area = box(TARGET_LON - b, TARGET_LAT - b, TARGET_LON + b, TARGET_LAT + b)

minx, miny, maxx, maxy = area.bounds

fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([minx - 1, maxx + 1, miny - 0.5, maxy + 0.5])
ax.add_feature(cfeature.LAND, facecolor="#f5f0e8")
ax.add_feature(cfeature.OCEAN, facecolor="#cfe2f3")
ax.add_feature(cfeature.COASTLINE, lw=0.5)
ax.add_feature(cfeature.RIVERS, color="steelblue")
ax.add_geometries([area], crs=ccrs.PlateCarree(),
                  facecolor="tab:blue", edgecolor="navy", alpha=0.35)
ax.plot(TARGET_LON, TARGET_LAT, "k*", markersize=20, transform=ccrs.PlateCarree())
ax.set_title(f"Contributing area for weather covariates ({location_mode})")
'''
)

# ---------------------------------------------------------------- obs
md(r"""## 3. Observed target timeseries

For a published preset we pull observations through
[RivRetrieve](https://github.com/kratzert/RivRetrieve-Python), a unified
interface to global streamflow data (swap the fetcher to change country). For an
upload, we use the series you loaded in Section 1.""")

code(r'''if TARGET_MODE == "published" and cfg["source"].startswith("rivretrieve:"):
    import rivretrieve
    fetcher_name = cfg["source"].split(":", 1)[1]
    Fetcher = getattr(rivretrieve, fetcher_name)
    obs = Fetcher().get_data(
        gauge_id=cfg["gauge_id"],
        variable=cfg["rr_variable"],
        start_date="2010-01-01",
        end_date=str(pd.Timestamp.today().date()),
    )[cfg["rr_variable"]]
    obs.name = VARIABLE
elif TARGET_MODE == "upload":
    obs = uploaded_obs
else:
    raise ValueError(f"Unsupported source: {cfg.get('source')}")

obs.index.name = "time"
fig, ax = plt.subplots(figsize=(13, 4))
obs.plot(ax=ax, color="black", lw=1)
ax.set_ylabel(f"{VARIABLE} [{UNITS}]")
ax.set_title(f"Observed {VARIABLE}")
'''
)

# ---------------------------------------------------------------- covariates
md(r"""## 4. Weather covariates from dynamical.org

We average temperature and precipitation over the contributing area:
past values from the **NOAA GEFS analysis** (the history the model sees) and
**15-day ensemble forecasts** from both **NOAA GEFS** (31 members) and
**ECMWF IFS ENS** (51 members). Each ensemble member becomes one plausible future.

> 💡 The dynamical.org STAC catalog at
> `https://stac.dynamical.org/catalog.json` is designed to be AI-friendly — point
> an LLM at it to discover datasets.""")

code(r'''# How much past history to feed the model. ~6 years of daily values is plenty of
# seasonal context for TiRex-2 while keeping the dynamical.org pull modest; lower
# this on a slow connection (see docs/local_setup.md).
HISTORY_DAYS = 365 * 6
history_window = slice(INIT_TIME - pd.Timedelta(days=HISTORY_DAYS),
                       INIT_TIME - pd.Timedelta(days=1))
print(f"History window: {history_window.start.date()} -> {history_window.stop.date()} "
      f"({HISTORY_DAYS} days)")
'''
)

code(r'''def basin_daily(ds, time_dim):
    """Clip to the contributing area, average over space, resample to daily.

    Precipitation is converted from a surface flux to mm/day.
    """
    out = (
        ds[["temperature_2m", "precipitation_surface"]]
        .sel(latitude=slice(maxy + 0.5, miny - 0.5), longitude=slice(minx - 0.5, maxx + 0.5))
        .rio.clip([area], crs="EPSG:4326")
        .mean(dim=("latitude", "longitude"))
        .resample({time_dim: "1D"}).mean()
    )
    out["precipitation_surface"] *= 86_400
    return out.load()


def forecast_meteo(dataset_id):
    return basin_daily(
        dynamical_catalog.open(dataset_id)
        .sel(init_time=INIT_TIME, lead_time=slice("0h", f"{FORECAST_DAYS}d"))
        .swap_dims({"lead_time": "valid_time"}),
        "valid_time",
    ).isel(valid_time=slice(0, FORECAST_DAYS))


analysis_meteo = basin_daily(
    dynamical_catalog.open("noaa-gefs-analysis").sel(time=slice(history_window.start, None)),
    "time",
)
gefs_fc_basin = forecast_meteo("noaa-gefs-forecast-35-day")
ifs_fc_basin = forecast_meteo("ecmwf-ifs-ens-forecast-15-day-0-25-degree")

history_df = (
    pd.concat(
        {
            VARIABLE: obs,
            "temperature_2m": analysis_meteo["temperature_2m"].to_pandas(),
            "precipitation_surface": analysis_meteo["precipitation_surface"].to_pandas(),
        },
        axis=1,
    )
    .loc[history_window]
    .reset_index()
    .rename(columns={"time": "timestamp"})
)
# Keep target gaps (NaN) — the foundation models handle them — but require
# covariates to be present.
history_df = history_df.dropna(subset=["temperature_2m", "precipitation_surface"])
print(f"History rows: {len(history_df)}, target gaps: {history_df[VARIABLE].isna().sum()}")
'''
)

# ---------------------------------------------------------------- model
md(r"""## 5. Run the forecast with TiRex-2

TiRex-2 receives the past target + past weather, plus a future weather trace per
ensemble member, and returns one forecast trace per member — same history,
different future weather.

- **TiRex-2** — `TimeseriesType(target, past_covariates, future_covariates)` fed
  to `model.forecast`; returns quantiles (we take the median per member).

> This runs on CPU by default (no GPU needed); `device` is auto-selected below.""")

code(r'''def _future_long(future_basin):
    """dynamical forecast DataArray -> long DataFrame [member,timestamp,temp,precip]."""
    return (
        future_basin.to_dataframe().reset_index()
        .rename(columns={"valid_time": "timestamp", "ensemble_member": "member"})
        [["member", "timestamp", "temperature_2m", "precipitation_surface"]]
        .astype({"member": str})
    )
'''
)

code(r'''import torch
from tirex2 import TimeseriesType, load_model

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tirex = load_model("NX-AI/TiRex-2", device=_DEVICE)

# Index of the median (0.5) among the model's native quantiles — read from the
# model rather than hardcoded, so this survives a change in quantile count.
_Q = [round(float(q), 3) for q in tirex.quantiles.detach().cpu().numpy()]
_MEDIAN_Q = _Q.index(0.5)

# Temperature and precipitation are FUTURE-KNOWN covariates: we have both the
# analysis over the history window and a forecast over the horizon. TiRex-2 wants
# such covariates as one series spanning context_length + prediction_length.
_COVS = ["temperature_2m", "precipitation_surface"]


def run_tirex2(future_basin):
    """One TiRex-2 (median) trace per ensemble member. Returns DataFrame [time x member].

    Builds one TimeseriesType per member: a shared target history plus that
    member's future-known weather covariates (history covariates concatenated
    with the member's forecast covariates). Returns the median quantile.
    """
    hist = history_df.set_index("timestamp")
    target = hist[VARIABLE].to_numpy(dtype="float32")[None, :]   # (1, context_len), NaNs ok
    hist_cov = hist[_COVS].to_numpy(dtype="float32").T           # (n_cov, context_len)
    fut = _future_long(future_basin)

    out = {}
    valid_time = None
    for member, g in fut.groupby("member"):
        g = g.sort_values("timestamp")
        fut_cov = g[_COVS].to_numpy(dtype="float32").T           # (n_cov, horizon)
        future_covariates = np.concatenate([hist_cov, fut_cov], axis=1)  # context+horizon
        ts = TimeseriesType(
            target=torch.from_numpy(target),
            past_covariates=None,
            future_covariates=torch.from_numpy(future_covariates),
        )
        fc = tirex.forecast(timeseries=[ts], prediction_length=FORECAST_DAYS,
                            output_type="numpy")[0]
        # fc shape: (n_targets, n_quantiles, horizon).
        out[member] = np.asarray(fc)[0, _MEDIAN_Q, :]
        valid_time = g["timestamp"].to_numpy()
    return pd.DataFrame(out, index=pd.DatetimeIndex(valid_time, name="timestamp"))
'''
)

code(r'''MODELS = {"TiRex-2": run_tirex2}
COVARIATE_SOURCES = {"GEFS": gefs_fc_basin, "IFS ENS": ifs_fc_basin}

# predictions[model][covariate_source] -> DataFrame [valid_time x member]
predictions = {}
for model_name, runner in MODELS.items():
    predictions[model_name] = {}
    for cov_name, fc_basin in COVARIATE_SOURCES.items():
        print(f"Running {model_name} with {cov_name} covariates ...")
        predictions[model_name][cov_name] = runner(fc_basin)
print("Done.")
'''
)

# ---------------------------------------------------------------- plots
md(r"""## 6. Forecast plots

Ensemble spread (thin lines) and ensemble mean (thick) for each weather source,
over the recent observed record.""")

code(r'''PLOT_WINDOW = slice(INIT_TIME - pd.Timedelta(days=30),
                    INIT_TIME + pd.Timedelta(days=FORECAST_DAYS))


def plot_traces(ax, pred, color, label):
    ax.plot(pred.index, pred.values, color=color, lw=0.6, alpha=0.30)
    ax.plot(pred.index, pred.mean(axis=1), color=color, lw=3, label=f"{label} mean")


def plot_obs_and_init(ax):
    o = obs.loc[PLOT_WINDOW]
    ax.plot(o.index, o.values, color="black", lw=2.5, label="Observed")
    ax.axvline(INIT_TIME, color="gray", ls="--", alpha=0.6)
    ax.set_ylabel(f"{VARIABLE} [{UNITS}]")
    ax.legend(loc="upper left")


fig, axes = plt.subplots(len(MODELS), 1, figsize=(13, 5 * len(MODELS)), sharex=True)
axes = np.atleast_1d(axes)
cov_colors = {"GEFS": "tab:green", "IFS ENS": "tab:blue"}
for ax, (model_name, by_cov) in zip(axes, predictions.items()):
    for cov_name, pred in by_cov.items():
        plot_traces(ax, pred, cov_colors[cov_name], cov_name)
    plot_obs_and_init(ax)
    ax.set_title(f"{model_name} — {FORECAST_DAYS}-day forecast, initialized {INIT_TIME.date()}")
fig.tight_layout()
'''
)

code(r'''# Weather-source comparison: TiRex-2 ensemble mean and 10-90% spread per source.
fig, ax = plt.subplots(figsize=(13, 5))
by_cov = predictions["TiRex-2"]
for cov_name, pred in by_cov.items():
    ax.plot(pred.index, pred.mean(axis=1), color=cov_colors[cov_name],
            lw=3, label=f"{cov_name} mean")
    ax.fill_between(pred.index, pred.quantile(0.1, axis=1), pred.quantile(0.9, axis=1),
                    color=cov_colors[cov_name], alpha=0.15)
plot_obs_and_init(ax)
ax.set_title("TiRex-2 forecast — GEFS vs IFS ENS weather (ensemble mean, 10–90%)")
'''
)

# ---------------------------------------------------------------- submit
md(r"""## 7. Export & submit your forecast

We write a **standardized forecast file** plus a **metadata sidecar** using values
already in scope. Then submit both through the workshop Google Form (which also
handles the optional co-authorship consent).

**Submit here:** https://forms.gle/DMqsNGiZtV1wjYP56  ·  details in `docs/submission.md`.""")

code(r'''# Tidy long-format forecast: one row per (model, covariate_source, member, valid_time).
rows = []
for model_name, by_cov in predictions.items():
    for cov_name, pred in by_cov.items():
        long = pred.reset_index().melt(
            id_vars="timestamp", var_name="member", value_name="prediction")
        long["model"] = model_name
        long["covariate_source"] = cov_name
        rows.append(long)
forecast_long = pd.concat(rows, ignore_index=True).rename(columns={"timestamp": "valid_time"})

slug = (PRESET if TARGET_MODE == "published" else VARIABLE).replace(" ", "-")
stem = f"forecast_{slug}_{INIT_TIME.date()}"
forecast_long.to_csv(f"{stem}.csv", index=False)

metadata = {
    "participant": PARTICIPANT,
    "variable": VARIABLE,
    "units": UNITS,
    "location": {"lat": TARGET_LAT, "lon": TARGET_LON,
                 "location_mode": cfg.get("location_mode")},
    "init_time": str(INIT_TIME.date()),
    "forecast_days": FORECAST_DAYS,
    "models": list(MODELS),
    "covariate_sources": list(COVARIATE_SOURCES),
    "target_mode": TARGET_MODE,
    "preset": PRESET if TARGET_MODE == "published" else None,
    "coauthor_optin": PARTICIPANT["coauthor_optin"],
    "license": "CC-BY-4.0",
}
with open(f"{stem}.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Wrote {stem}.csv ({len(forecast_long)} rows) and {stem}.metadata.json")
if not PARTICIPANT["name"]:
    print("⚠️  PARTICIPANT['name'] is empty — fill in Section 1 before submitting.")
'''
)

code(r'''# On Colab, download the two files so you can attach them to the Google Form.
if IN_COLAB:
    from google.colab import files
    files.download(f"{stem}.csv")
    files.download(f"{stem}.metadata.json")
else:
    print(f"Files written to the working directory: {stem}.csv, {stem}.metadata.json")
'''
)

md(r"""## Where to go next
- `appendix_zonal_stats.ipynb` — area-weighted covariate extraction with `xvec`.
- `appendix_model_deep_dive.ipynb` — how TiRex-2 behaves with gaps in the target
  history, and how covariates change the forecast.
- Swap the fetcher / gauge in Section 1 to forecast a different river; see
  `docs/data_sources.md` for options.""")


nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": s.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if t == "code" else {}),
        }
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "forecast_workshop.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} with {len(CELLS)} cells")
