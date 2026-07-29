# Flight delay model card

## Problem

Predict whether a US hub flight will arrive **15 or more minutes late**
(`arr_delay_15`), matching the common BTS on-time definition.

## Data

| Source | Role |
|---|---|
| BTS On-Time Performance | Labels, schedule, taxi, delay causes, volume |
| Open-Meteo Archive | Origin/destination daily weather |
| OurAirports | Airport coordinates / metadata |

Default hubs: LAX, JFK, ORD, DEN, ATL, IAD, DFW.

## Features (leakage-aware)

**Schedule / calendar**
- Carrier, origin, dest, month, day-of-week, departure hour
- Distance, scheduled elapsed time
- Weekend + peak-hour flags
- Same-day scheduled flights at origin hour / dest hour / origin day (bank congestion)

**Weather**
- Origin/dest temp, precip, wind, weather code

**Historical congestion & reliability (lookups, not same-flight outcomes)**
- Origin-hour: avg taxi-out, NAS/carrier/weather/late-aircraft delay, ops volume, delay rate
- Dest-hour: avg taxi-in, ops volume, delay rate
- Route×carrier historical delay rate
- Carrier-wide delay / taxi / late-aircraft averages

**Not used as model inputs (kept in lake for analysis only)**
- Same-flight `taxi_out` / `taxi_in` / `nas_delay` / cause minutes / `dep_delay`
  (these are after-the-fact and would leak the label)

## Model

sklearn `Pipeline`: `OneHotEncoder` (categoricals) + `XGBClassifier`
(fallback: `HistGradientBoostingClassifier` if XGBoost/OpenMP unavailable).

Tracked with MLflow experiment `flight-delay`. Local artifact:
`models/local/model.joblib`.

## Evaluation

Metrics written to `models/local/metrics.json` (accuracy, precision, recall, F1,
ROC-AUC, and per-origin F1).

## Limitations

- Hub-scoped; not a national network model
- Historical airport-hour / route aggregates can mildly leak if computed on the full sample (MVP)
- Sample/synthetic mode is for demos and CI, not production accuracy claims
- Weather is daily, not METAR/TAF flight-time specific
- No live FAA ASPM feed yet — congestion is derived from BTS history + schedule volume

## Ethical / ops notes

Predictions are probabilistic decision support, not guarantees. Always combine
with airline/airport operational notices.
