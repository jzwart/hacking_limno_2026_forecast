# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Workshop materials for the AEMON-J / DSOS "Hacking Limnology" 2026 summit (Day 2,
Climate Data). Participants run a Google Colab notebook to forecast a water
variable (streamflow, stream/lake temperature, or an uploaded timeseries) with the
zero-shot foundation model **Chronos-2**, conditioned on open ensemble weather
forecasts from **dynamical.org**, then submit the result via a Google Form for an
eventual global evaluation. Colab is the primary target; a local `uv` path exists
for participants without a Google account or with slow connections.

## Critical workflow: notebooks are generated, not hand-edited

The `.ipynb` files are **build artifacts**. Their source of truth is the
generator scripts:

- `notebooks/_build_core.py` → `notebooks/forecast_workshop.ipynb`
- `notebooks/_build_appendices.py` → `appendix_zonal_stats.ipynb` + `appendix_model_deep_dive.ipynb`

**Never edit the `.ipynb` directly** — edit the `_build_*.py` script and regenerate:

```bash
cd notebooks
python _build_core.py          # rebuild the core notebook
python _build_appendices.py    # rebuild both appendices
```

Each script defines cells as ordered `md(...)` / `code(...)` calls, then serializes
them to `.ipynb` JSON. After editing, always rebuild and validate:

```bash
python -c "import json; json.load(open('forecast_workshop.ipynb')); print('valid')"
```

## Runtime environment (why the install is fragile)

The notebooks run in Colab / Jupyter, not as a package. The install cell order in
`_build_core.py` is deliberate and load-bearing:

1. **CPU PyTorch first** (`--index-url .../whl/cpu`, Linux only). Installing the CPU
   torch build first keeps the heavier installs below from dragging in the large
   CUDA torch (and, via `chronos-forecasting[extras]` → transformers/torchvision,
   a mismatched CUDA torch that triggers `torchvision::nms does not exist`).
2. **numpy pinned** (`>=1.26,<2.1`) so later installs don't leave a half-upgraded
   numpy (`cannot import name '_center' from numpy._core.umath`).
3. **The rest of the stack** (dynamical-catalog, rioxarray, cartopy, geopandas,
   chronos-forecasting[extras], RivRetrieve).

On Colab the install cell restarts the runtime **once** (guarded by a `/tmp`
sentinel written only on success, so it never loops and a failed install re-runs
clean). `chronos-forecasting[extras]` pulls transformers/torchvision, so its
version churn is the main install risk — installing CPU torch first (step 1)
contains it; re-verify the whole install before adding packages near this chain.

## Core notebook structure (`forecast_workshop.ipynb`)

Additive-and-defaulted design: running top-to-bottom untouched reproduces a working
Delaware-River-at-Montague streamflow forecast. Participants change **only the config
block in Section 1**.

- **§1 config** — `TARGET_MODE` (`"published"` | `"upload"`), a `PRESETS` dict
  (default `delaware_streamflow`), and `PARTICIPANT` metadata. `location_mode` is
  `"delineate"` (river point → upstream basin via Global Watersheds API) or
  `"buffer"` (point → box, for lakes/uploads).
- **§2 area / §3 obs / §4 covariates** — basin geometry; RivRetrieve or uploaded
  obs; `basin_daily` clips dynamical.org temp/precip to the area and averages.
  `HISTORY_DAYS` (~6 yr) sets the context window.
- **§5 Chronos-2** — `run_chronos2(future_basin)` calls `Chronos2Pipeline.predict_df`
  once with the target history replicated per weather ensemble member (tagged by
  `member`) as `context`, and that member's future weather as `future_df`. Temp/precip
  are **future-known covariates**. It requests quantile levels `_Q = [0.1, 0.5, 0.9]`
  and keeps every quantile as its own `<member>_q<level>` trace (not just the median),
  so the ensemble carries weather-member spread AND the model's predictive quantiles.
  Runs on both GEFS and IFS ENS covariate sources.
- **§6 plots / §7 submission** — writes `forecast_<slug>_<init>.csv` in the **EFI
  standard** (long: project_id, model_id, datetime, reference_datetime, duration,
  site_id, variable, family="ensemble", parameter, prediction; each weather source is
  its own `model_id`, each trace an integer `parameter`) + a `.metadata.json` sidecar;
  on Colab downloads both for the Google Form.

Target-mode gaps: uploaded/observed **target** series may contain NaNs (Chronos-2
handles them); **covariates** must be present (`history_df.dropna(subset=[...covs])`).

## Local development

```bash
uv sync                 # install pinned deps (pyproject.toml + uv.lock)
uv sync --extra zonal   # add xvec/exactextract for the zonal-stats appendix
uv run jupyter lab
```

`uv.lock` is committed. There is no test suite; "validation" means rebuilding the
notebooks and confirming valid JSON. The notebooks cannot be fully executed without
network access and model downloads, so end-to-end verification happens in Colab.

## Conventions

- Generated forecast outputs (`forecast_*.csv`, `*.metadata.json`) and
  `my_timeseries.csv` are gitignored — do not commit them.
- Colab badge / form / clone URLs are hardcoded to
  `github.com/jzwart/hacking_limno_2026_forecast` and
  `forms.gle/DMqsNGiZtV1wjYP56`; update all of README, `docs/submission.md`, and
  the notebook if the repo or form moves.
- Model API facts (chronos-forecasting install, `Chronos2Pipeline.predict_df`
  signature, `future_df` covariate shape) come from the vendor repo
  `github.com/amazon-science/chronos-forecasting`; verify against its README /
  getting-started example before changing §5.
