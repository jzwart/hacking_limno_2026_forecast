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
uv run --with jupyter jupyter lab
# then open notebooks/forecast_workshop.ipynb
```
Then select the `.venv` interpreter (`.venv/bin/python`, or
`.venv\Scripts\python` on Windows) in your editor/IDE. With `uv sync` done, you
can **skip the `!uv pip install` cell** at the top of each notebook.

## Slow / low-bandwidth tips
- **Model weights** download once from Hugging Face and are cached under
  `~/.cache/huggingface`. The first run is the slow one; later runs reuse the cache.
- **Shrink the context.** In the core notebook, a smaller history window means
  less weather data to pull from dynamical.org. Lower `HISTORY_DAYS` (default
  ~6 years) to speed up the covariate step.
- **CPU is fine.** No GPU required. Chronos-2 auto-selects `device="cpu"` when no
  CUDA GPU is present; Section 5 just runs slower.
- **Fewer ensemble members.** For a quick test, subset `gefs_fc_basin` /
  `ifs_fc_basin` to a handful of `ensemble_member`s before Section 5.
- **Optional HF token.** Setting `HF_TOKEN` in your environment avoids occasional
  download rate limits, but Chronos-2 is public so it is not required.

## GPU vs CPU for Chronos-2
The notebook installs the **CPU build of PyTorch** on purpose, so the install
doesn't drag in the large CUDA torch build. CPU is fast enough for the single-site
forecast in this workshop.

If you have a capable local GPU and want the CUDA path, install a CUDA torch build
*before* `chronos-forecasting` (matching your CUDA toolkit), e.g.:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install "chronos-forecasting[extras]>=2.2"
```
and set `device_map="cuda"` where the model loads.

## Troubleshooting
- **`cannot import name '_center' from numpy._core.umath`:** a half-upgraded
  numpy. On Colab, restart the runtime and re-run from the top (the install cell
  pins numpy and restarts once automatically). Locally, `uv sync` avoids it.
- **`torchvision::nms does not exist` / torch–transformers version mismatch:**
  `chronos-forecasting[extras]` pulls `transformers`; install the CPU torch build
  first (the default in the notebook) so a mismatched CUDA torch isn't pulled in.
- **`SSL: CERTIFICATE_VERIFY_FAILED` / "self-signed certificate in certificate
  chain":** you are behind a TLS-inspecting proxy (common on corporate/USGS
  networks) whose root CA your OS trusts but Python's bundled `certifi` store does
  not. The notebook installs [`truststore`](https://pypi.org/project/truststore/)
  and calls `truststore.inject_into_ssl()` in the imports cell to use the OS trust
  store instead — just re-run the cells. If you hit this in your own script,
  `import truststore; truststore.inject_into_ssl()` before any `requests` call.
- **`cartopy` build issues:** `uv` usually resolves wheels; if not, install GEOS
  and PROJ via your system package manager, or run the notebook on Colab instead.
- **`chronos-forecasting` platform:** tested on Linux and macOS. On Windows, prefer
  Colab or WSL.
