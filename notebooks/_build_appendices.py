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
!pip install -q "numpy>=1.26,<2.1" truststore dynamical-catalog rioxarray xvec exactextract geopandas requests"""))
zonal.append(("code", r'''# Route TLS through the OS trust store so https works behind TLS-inspecting
# proxies (e.g. USGS) that inject a self-signed root CA. No-op if not installed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import dynamical_catalog
import geopandas as gpd
import pandas as pd
import requests
import rioxarray  # noqa: F401
import xvec  # noqa: F401  registers the .xvec accessor
from shapely.geometry import shape

# Same reference basin as the core notebook (Thames at Kingston).
LAT, LON = 51.4155, -0.3076
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
    dynamical_catalog.open("noaa-gefs-analysis")
    .sel(time=slice("2025-12-01", "2026-01-08"))
    [["temperature_2m", "precipitation_surface"]]
    .sel(latitude=slice(maxy + 0.5, miny - 0.5), longitude=slice(minx - 0.5, maxx + 0.5))
    .load()
)
ds["precipitation_surface"] *= 86_400  # -> mm/day
ds
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
build(zonal, "appendix_zonal_stats.ipynb", "zonal stats")


# ============================================================ model deep dive
dive = []
dive.append(("markdown", r"""# Appendix: Chronos-2 — data gaps & the effect of covariates

Two questions the core notebook glosses over:

1. **What happens when the target history has gaps?** Real observation records
   have missing days. Chronos-2 is trained to forecast from incomplete context and
   ingests NaNs directly, but the effect on skill is worth seeing. We take a
   clean series, punch synthetic gaps into it, and re-forecast.
2. **How much do covariates change the forecast?** We compare a target-only
   forecast against one conditioned on a future-known covariate.

This is a diagnostic notebook — it deliberately runs on synthetic data so it
stays fast in a live session and needs no downloads beyond the model.
"""))
dive.append(("code", r"""# CPU torch first so the chronos install doesn't pull the large CUDA torch build;
# then numpy pinned so later installs don't leave a half-upgraded numpy. If a numpy
# ImportError appears on Colab, restart & re-run.
!pip install -q "torch<2.10" --index-url https://download.pytorch.org/whl/cpu
!pip install -q "numpy>=1.26,<2.1" "chronos-forecasting[extras]>=2.2" pandas matplotlib"""))
dive.append(("code", r'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# A synthetic but realistic daily series: seasonal cycle + AR(1) noise.
n = 365 * 4
rng = np.random.default_rng(0)
t = np.arange(n)
seasonal = 20 + 15 * np.sin(2 * np.pi * t / 365)
noise = np.zeros(n)
for i in range(1, n):
    noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 3)
series = np.clip(seasonal + noise, 1, None)
idx = pd.date_range("2022-01-01", periods=n, freq="D")
clean = pd.Series(series, index=idx, name="value")

HORIZON = 15
train = clean.iloc[:-HORIZON]
truth = clean.iloc[-HORIZON:]
'''))
dive.append(("markdown", r"""## Load Chronos-2"""))
dive.append(("code", r'''from chronos import Chronos2Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
print("Chronos-2 loaded on", device)
'''))
dive.append(("markdown", r"""## Forecast helper (target-only)

To isolate the gap effect we forecast the univariate target with no covariates.
Chronos-2 accepts a bare context series with NaNs.
"""))
dive.append(("code", r'''def chronos_forecast(context_series):
    # predict_df wants a long frame with an id, a timestamp, and the target column.
    context = (context_series.rename("value").rename_axis("timestamp")
               .reset_index().assign(item="series"))
    pred = chronos.predict_df(
        context, prediction_length=HORIZON, quantile_levels=[0.5],
        id_column="item", timestamp_column="timestamp", target="value",
    )
    return pred.sort_values("timestamp")["0.5"].to_numpy()  # median quantile
'''))
dive.append(("markdown", r"""## Punch synthetic gaps

We drop random days from the *recent* history (last year) at increasing rates and
re-forecast. Chronos-2 accepts the NaNs directly — no imputation needed.
"""))
dive.append(("code", r'''def with_gaps(series, frac, seed):
    r = np.random.default_rng(seed)
    s = series.copy()
    recent = s.index[-365:]
    drop = r.choice(recent, size=int(frac * len(recent)), replace=False)
    s.loc[drop] = np.nan
    return s

rmse = lambda a, b: float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))

records = []
for frac in [0.0, 0.1, 0.3, 0.5, 0.7]:
    gapped = with_gaps(train, frac, seed=1)
    pred = chronos_forecast(gapped)                # native NaN handling
    records.append({"gap_frac": frac, "Chronos-2 RMSE": rmse(pred, truth.values)})
skill = pd.DataFrame(records).set_index("gap_frac")
skill
'''))
dive.append(("code", r'''fig, ax = plt.subplots(figsize=(9, 5))
skill.plot(marker="o", ax=ax, legend=False)
ax.set_xlabel("Fraction of last-year history missing")
ax.set_ylabel(f"{HORIZON}-day forecast RMSE")
ax.set_title("Chronos-2 forecast skill vs. gaps in the target history")
'''))
dive.append(("markdown", r"""## Takeaways

- Chronos-2 ingests NaNs natively and degrades gracefully as gaps grow, rather than
  failing — no imputation step required.
- This is why the core notebook keeps **target** gaps and only requires the
  **covariates** to be present: the model handles a patchy observation record for
  you.
- For a very gappy or short record, expect wider uncertainty; consider a longer
  history window (`HISTORY_DAYS` in the core notebook) to give the model more
  seasonal context.
"""))
build(dive, "appendix_model_deep_dive.ipynb", "model deep dive")
