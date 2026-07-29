# Running the notebooks locally with `uv`

For participants without a Google account, or with a slow/unreliable connection
where Colab is painful, run everything locally. [`uv`](https://github.com/astral-sh/uv)
is a fast, single-binary Python installer/runner — no pre-existing Python setup
needed.

## 1. Install uv
One-liner from <https://docs.astral.sh/uv/getting-started/installation/>:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Get the code
```bash
git clone https://github.com/jzwart/hacking_limno_2026_forecast.git
cd hacking_limno_2026_forecast
```

## 3a. Quick launch (no project install)
Matches the [dynamical.org notebooks](https://github.com/dynamical-org/notebooks) workflow:

```bash
uv run --with jupyter jupyter lab
```

Open `notebooks/forecast_workshop.ipynb`. The notebook's first cell installs the
forecasting dependencies into the environment.

## 3b. Reproducible environment (recommended for repeat use)
```bash
uv sync                      # installs pinned deps from pyproject.toml
uv sync --extra zonal        # also install xvec/exactextract for the zonal appendix
uv run jupyter lab
```
Then select the `.venv` interpreter (`.venv/bin/python`, or
`.venv\Scripts\python` on Windows) in your editor/IDE. With `uv sync` done, you
can **skip the `!uv pip install` cell** at the top of each notebook.

## Slow / low-bandwidth tips
- **Model weights** download once from Hugging Face and are cached under
  `~/.cache/huggingface`. The first run is the slow one; later runs reuse the cache.
- **Shrink the context.** In the core notebook, a smaller history window means
  less weather data to pull from dynamical.org. Chronos-2 sizes its own window,
  but you can trim `INIT_TIME - history` by editing `HISTORY_DAYS`.
- **CPU is fine.** No GPU required. TiRex-2 auto-selects `device="cpu"` when no
  CUDA GPU is present; Section 5 just runs slower.
- **Fewer ensemble members.** For a quick test, subset `gefs_fc_basin` /
  `ifs_fc_basin` to a handful of `ensemble_member`s before Section 5.
- **Optional HF token.** Setting `HF_TOKEN` in your environment avoids occasional
  download rate limits, but the models are public so it is not required.

## Troubleshooting
- **`cartopy` build issues:** `uv` usually resolves wheels; if not, install GEOS
  and PROJ via your system package manager, or run the notebook on Colab instead.
- **`tirex-2` platform:** tested on Linux and macOS. On Windows, prefer Colab or
  WSL.
