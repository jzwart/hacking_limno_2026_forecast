# Submitting your forecast

At the end of `forecast_workshop.ipynb` (Section 7) the notebook writes two files:

- `forecast_<target>_<init_date>.csv` — the forecast itself
- `forecast_<target>_<init_date>.metadata.json` — who/what/where

Submit **both** through the workshop Google Form.

> **Submit here:** <https://forms.gle/DMqsNGiZtV1wjYP56>

## Forecast file schema (`.csv`)
Tidy long format — one row per model × weather source × ensemble member × day:

| column | type | description |
|---|---|---|
| `valid_time` | date | the day being forecast |
| `member` | string | ensemble member id (weather-driven) |
| `prediction` | float | forecast value in the target's units |
| `model` | string | `Chronos-2` or `TiRex-2` |
| `covariate_source` | string | `GEFS` or `IFS ENS` |

## Metadata file schema (`.metadata.json`)
```json
{
  "participant": {"name": "", "affiliation": "", "email": "", "orcid": "",
                  "coauthor_optin": false},
  "variable": "streamflow",
  "units": "m3/s",
  "location": {"lat": 51.4155, "lon": -0.3076, "location_mode": "delineate"},
  "init_time": "2026-01-08",
  "forecast_days": 15,
  "models": ["Chronos-2", "TiRex-2"],
  "covariate_sources": ["GEFS", "IFS ENS"],
  "target_mode": "published",
  "preset": "thames_streamflow",
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
8. **Models used** (checkboxes: Chronos-2 / TiRex-2)
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
