# Submitting your forecast

At the end of `forecast_workshop.ipynb` (Section 7) the notebook writes two files:

- `forecast_<target>_<init_date>.csv` — the forecast itself
- `forecast_<target>_<init_date>.metadata.json` — who/what/where

Submit **both** through the workshop Google Form.

> **Submit here:** <https://forms.gle/DMqsNGiZtV1wjYP56>

## Forecast file schema (`.csv`)
The [Ecological Forecasting Initiative (EFI) standard](https://projects.ecoforecast.org/neon4cast-docs/Submission-Instructions.html)
long format used by the NEON / USGS forecasting challenges — one row per
`model_id` × ensemble member × day. `family="ensemble"`, so each row is one forecast
trace and no distribution is fitted:

| column | type | description |
|---|---|---|
| `project_id` | string | collection id, `hacking_limno_2026` |
| `model_id` | string | model + weather source, e.g. `chronos2_gefs`, `chronos2_ifs_ens` |
| `datetime` | date | the day being forecast |
| `reference_datetime` | date | forecast init date (t0) |
| `duration` | string | ISO 8601 timestep, `P1D` (daily) |
| `site_id` | string | target id (e.g. `delaware_streamflow` or your uploaded variable) |
| `variable` | string | forecast variable, e.g. `streamflow` |
| `family` | string | `ensemble` |
| `parameter` | integer | ensemble member index (weather member × Chronos-2 quantile) |
| `prediction` | float | forecast value in the target's units |

## Metadata file schema (`.metadata.json`)
```json
{
  "participant": {"name": "", "affiliation": "", "email": "", "orcid": "",
                  "coauthor_optin": false},
  "standard": "EFI",
  "project_id": "hacking_limno_2026",
  "site_id": "delaware_streamflow",
  "variable": "streamflow",
  "units": "m3/s",
  "duration": "P1D",
  "family": "ensemble",
  "location": {"lat": 51.4155, "lon": -0.3076, "location_mode": "delineate"},
  "reference_datetime": "2026-01-08",
  "forecast_days": 10,
  "model_ids": ["chronos2_gefs", "chronos2_ifs_ens"],
  "models": ["Chronos-2"],
  "covariate_sources": ["GEFS", "IFS ENS"],
  "target_mode": "published",
  "preset": "delaware_streamflow",
  "coauthor_optin": false,
  "license": "CC-BY-4.0"
}
```

## Google Form spec (for the organizers to create)
A form with these fields, mirroring the metadata so submissions are self-describing
even if someone skips the JSON file:

1. **Name** (short answer)
2. **Affiliation** (short answer)
3. **Email** (short answer)
4. **ORCID** (short answer, optional)
5. **Target variable** (multiple choice: streamflow / stream temperature /
   lake temperature / other)
6. **Site latitude** and **Site longitude** (short answer)
7. **Forecast init date** (date)
8. **Model used** (short answer; defaults to Chronos-2)
9. **Forecast file** (file upload — accept `.csv`)
10. **Metadata file** (file upload — accept `.json`)
11. **Co-authorship consent** (checkbox, opt-in):
    > *"I consent to being included as a co-author on any publication or public
    > output arising from the global evaluation of these submitted forecasts. I
    > understand this is optional and my forecast will be included in the
    > evaluation either way."*
12. **License** (multiple choice, default CC-BY-4.0)

File uploads route to a Drive folder owned by the organizers. Responses collect
into a Sheet that seeds the eventual global evaluation.

### No Google account?
Colab lets you run and **download** the two files without submitting via the form.
If you cannot use the form at all, email both files to <jzwart@usgs.gov> with
the metadata fields above in the body.
