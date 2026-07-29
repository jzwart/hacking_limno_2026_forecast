# Zero-shot water forecasting with open weather data

Workshop materials for **AEMON-J / DSOS "Hacking Limnology" 2026 — Day 2: Climate Data**
(Tue 11 Aug 2026). Workshop: *How to access petabytes of weather forecasts from your
laptop* — Jake Zwart (USGS) & Alden Keefe Sampson (Dynamical).

You generate a short-range forecast of a **water variable** — streamflow, stream
temperature, lake temperature, or **your own uploaded timeseries** — using two
zero-shot time-series foundation models ([Chronos-2](https://huggingface.co/amazon/chronos-2)
and [TiRex-2](https://github.com/NX-AI/tirex-2)) driven by open ensemble weather
forecasts from [dynamical.org](https://dynamical.org). Then you **submit** your
forecast for an eventual global evaluation.

> Run it once with no edits and it reproduces a working forecast for the River
> Thames at Kingston. Then edit a single config block to forecast *your* site.

## Notebooks

| Notebook | What it does | Open in Colab |
|---|---|---|
| `notebooks/forecast_workshop.ipynb` | **Start here.** End-to-end forecast + submission. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/forecast_workshop.ipynb) &nbsp; [▶ Open](https://colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/forecast_workshop.ipynb) |
| `notebooks/appendix_zonal_stats.ipynb` | Area-weighted covariate extraction with `xvec`. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/appendix_zonal_stats.ipynb) &nbsp; [▶ Open](https://colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/appendix_zonal_stats.ipynb) |
| `notebooks/appendix_model_deep_dive.ipynb` | Context windows + behavior with data gaps. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/appendix_model_deep_dive.ipynb) &nbsp; [▶ Open](https://colab.research.google.com/github/jzwart/hacking_limno_2026_forecast/blob/main/notebooks/appendix_model_deep_dive.ipynb) |

## Two ways to run

### Google Colab (recommended)
Click a badge above. The first cell installs everything (~1–2 min). A Google
account is helpful for submitting via the form but is **not required to run** the
notebook — you can run it and download the output files without one.

### Locally with `uv` (no Google account; better on slow connections)
See [`docs/local_setup.md`](docs/local_setup.md). Short version:

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv run --with jupyter jupyter lab
# then open notebooks/forecast_workshop.ipynb
```

## Choosing your target
- **Published data** — pick a preset in Section 1 (RivRetrieve for global
  streamflow; USGS NWIS, etc.). Default is Thames streamflow.
- **Upload your own** — a CSV with a time column and a value column; gaps are
  fine. Works for lake temperature and any other water variable.

See [`docs/data_sources.md`](docs/data_sources.md) for sources per variable and
the CSV format.

## Submitting your forecast
The notebook writes a standardized `forecast_*.csv` + `*.metadata.json`. Submit
both through the workshop **Google Form** (link in the notebook and in
[`docs/submission.md`](docs/submission.md)). The form includes an **opt-in**
co-authorship consent for any resulting global-evaluation output.

## Repository layout
```
notebooks/
  forecast_workshop.ipynb        core workshop notebook
  appendix_zonal_stats.ipynb
  appendix_model_deep_dive.ipynb
  _build_*.py                    dev tools that regenerate the notebooks
docs/
  local_setup.md
  submission.md
  data_sources.md
pyproject.toml                   pinned deps for the local uv path
```
