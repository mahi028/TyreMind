# TyreMind — Frontend & 3D Handoff

**For whoever builds the dashboard and the 3D track view.**

Everything here is real and running right now. Every response shape below was
copied from a live call, not written from memory. If something in this document
disagrees with the API, the API is right and this file is stale — tell me.

---

## 1. What this product has to answer

Four questions. The mentors stated them as a scenario, and the UI is judged on
whether a race engineer can answer them at a glance:

> *"Lap 1 took 90 s, lap 2 took 92 s, lap 3 took 95 s. Where is the time going?"*

| # | Question | Panel | Endpoint |
|---|---|---|---|
| 1 | **Where is the time going?** | stacked contribution chart | `/decompose-run` |
| 2 | **How fast is he losing performance?** | degradation rate + interval | `/degradation` |
| 3 | **What happens after 3 laps? 15 laps?** | forward projection with a band | `/projection` |
| 4 | **When do I box?** | pit window with a cost sweep | `/pit-window` |
| 5 | **Can I trust this?** | calibration panel | `/trust` |

**The single design rule for this whole UI:**

> **Every number on screen carries its interval. No bare point estimates, anywhere.**

That is the product. A competitor can show "0.198 s/lap". Only we can show
"0.198 ± 0.022, and when we say 95% we are right 95% of the time". If a panel
drops the interval to look cleaner, it has thrown away the thing we are selling.

---

## 2. Start here — run it locally

```bash
git clone <repo> && cd TyreMind
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# Serves the API and the built frontend on http://127.0.0.1:8000
.venv/Scripts/python -m uvicorn tyremind.api.main:app --reload
```

**No internet, no API key, no database.** Eight sessions ship in `data/demo/`, so
a clean clone runs offline. Check `GET /api/health` returns `offline_ready: true`.

The React app lives in `apps/web/` (Vite + TypeScript). `apps/web/dist/` is what
the API serves at `/`.

---

## 3. Track geometry — ready now

```bash
.venv/Scripts/python scripts/export_track_geometry.py --all-demo
```

Writes `data/geometry/<circuit>.json`. Available today:

| Circuit | Lap length | Elevation gain | Corners |
|---|---|---|---|
| Barcelona | 4,601 m | **29.9 m** | 14 |
| Monza | 5,749 m | 12.5 m | 11 |
| Silverstone | 5,813 m | 11.3 m | 18 |
| Zandvoort | 4,200 m | 9.2 m | 14 |

### Format

```json
{
  "circuit": "Monza",
  "units": "metres",
  "rotation_deg": 95.0,
  "lap_length_m": 5748.7,
  "elevation_range_m": [184.2, 196.7],
  "elevation_gain_m": 12.5,
  "n_points": 600,
  "centreline": [[-136.9, -62.2, 187.2], [-136.08, -52.63, 187.23], ...],
  "corners": [
    {"number": 1, "letter": null, "x": -57.0, "y": 815.4,
     "angle_deg": 153.8, "distance_m": 876.8}
  ],
  "start_finish": [-136.9, -62.2, 187.2]
}
```

**Things worth knowing before you build the mesh:**

- **Coordinates are metres.** `[x, y, z]`, z is real elevation from the
  positioning feed — not synthesised, not flat.
- **600 points, evenly spaced by distance.** Raw telemetry samples on a fixed
  *time* interval, which bunches points in slow corners and stretches them on
  straights; that is backwards for a mesh. This is already resampled.
- **The loop is closed** — last point meets the first, no seam at start/finish.
- **`rotation_deg` orients the map the way broadcast does.** Apply it or Monza
  looks sideways.
- **This is a racing line, not a centreline.** It is the fastest lap's actual
  path, so it cuts corners and reads slightly short of the official lap length
  (5,749 m against an official 5,793 m). If you need the white lines, say so and
  we will pull track edges separately.

---

## 4. API contract

All routes are `GET` unless marked. Base: `http://127.0.0.1:8000`.

### `/api/health`
```json
{"status":"ok","sessions_available":8,"sessions_cached":8,"offline_ready":true}
```

### `/api/sessions`
```json
[{"session_id":"2024-monza-FP2","year":2024,"grand_prix":"Monza",
  "session":"FP2","label":"2024 Monza - FP2","cached":true}]
```

### `/api/session/{id}/runs` — stint list

```json
[{"driver":"VER","run_id":17,"compound":"MEDIUM","laps":12,
  "first_lap":11,"last_lap":22,"start_age":7,"end_age":18,
  "median_lap_time":83.4,
  "curve":{"regime":"linear","n_laps":12,"slope":0.274,
           "changepoint_age":null,"delta":null,"slope_after":null,
           "position":null,"linear_bic":9.21,"broken_bic":13.13,"rmse":1.08}}]
```

`curve.regime` is one of **`linear`, `warm-up`, `cliff`, `recovery`** — colour the
stint by it. It can be `null` on a stint too short to fit, which is not an error.

### `/api/session/{id}/decompose-run?driver=VER&run_id=17` — **question 1**

**Use this, not `/decompose`.** The single-lap version compares against a fixed
reference and accumulates driver noise; on some laps it leaves 73% of the gap as
residual. The per-lap run view is clean (residuals around 0.08 s).

```json
{"rows":[{"session_lap":12,"tyre_age":8.0,"compound":"MEDIUM",
  "observed_delta":0.004,"residual":-0.080,"tyre_share":0.489,
  "tyre":0.196,"fuel":-0.078,"track":-0.033,"traffic":0.0,
  "tyre_sd":0.083,"fuel_sd":0.016,"track_sd":0.013,"traffic_sd":0.0}]}
```

Stack `tyre / fuel / track / traffic`. **Render `residual` as its own visible
band labelled "unexplained".** Hiding it would claim we explain more than we do,
and an engineer who spots a hidden residual will not trust anything else on the
screen.

**`tyre_share` is a fraction, not a percentage, and it is not safe to render as
one.** It is `tyre_seconds / observed_delta`, so multiply by 100 yourself. Two
things it does that a percentage cannot, both of which have already produced a
bug in this repository:

- **It can exceed 1.** Measured between 0.82 and 15.79 across the demo sessions.
  When the car barely slowed but the tyre was still costing time, the other
  terms cancelled the rest, and the ratio is large and correct. A progress bar
  fed this value runs off the end of its track.
- **It is `null` when the car did not slow down at all**, because a share of a
  non-slowdown is not a quantity. Check for null before formatting.

If what you want is "how much of this lap was the tyre" on a bounded scale, use
`tyre` against `observed_delta` directly and say which you plotted.

### `/api/session/{id}/projection?driver=VER&lap=18` — **question 3**

```json
{"driver":"VER","from_lap":18,"tyre_age":14.0,"compound":"MEDIUM",
 "threshold_s":0.8,
 "competitive_life_laps":1.0,"competitive_life_lower":1.0,"competitive_life_upper":1.0,
 "horizon":[1,2,3,...,20],
 "loss":[1.98,2.27,2.55,...],
 "loss_sd":[0.46,0.54,0.63,...],
 "rate":[0.287,0.287,0.287,...],
 "rate_sd":[0.041,0.041,0.041,...],
 "breach_probability":[0.91,0.94,0.96,...],
 "applicability":[1.0,1.0,1.0,...],
 "is_model_estimate":true}
```

Plot `loss` as a line and `loss ± 1.96 × loss_sd` as the band. **The band widens
with horizon — that is the honest part, do not clamp it.**

**Every field after `threshold_s` except `is_model_estimate` is an array, one
entry per horizon step, and all of them are the same length as `horizon`.** That
includes `rate`, `rate_sd`, `breach_probability` and `applicability`. An earlier
version of this document showed `rate` and `breach_probability` as scalars; they
never were, and a client reading `data.rate.toFixed()` gets a runtime error
rather than a wrong number.

### `/api/session/{id}/pit-window?driver=VER&lap=18` — **question 4**

```json
{"driver":"VER","from_lap":18,"total_laps":25,"new_compound":"SOFT",
 "optimum_lap":19,"optimum_expected_time":632.78,
 "stay_out_expected_time":627.82,
 "window_within_1s":[19,20],
 "sweep":[{"pit_lap":19,"expected_time":632.78,"downside":637.18,
           "best_case":626.56,"runs_past_cliff":0.0}],
 "n_sims":..., "note":"..."}
```

Draw `sweep` as a cost curve with `best_case`/`downside` as a band, mark
`optimum_lap`, shade `window_within_1s`. Returns **400** with "no laps remain" on
the final lap of a run — that is correct behaviour, render it as a message.

### `/api/session/{id}/trust` — **question 5**

```json
{"consensus":{"MEDIUM":{"consensus":0.1977,"consensus_sd":0.0216,
  "spread":0.0296,"agreement":0.710,"disagreement_flagged":false,
  "estimates":{"baseline":{"mean":0.1957,"sd":0.0433},
               "fuel prior +1sd":{"mean":0.2105,"sd":0.0433}, ...},
  "explanation":"Four independent methods put medium degradation within 0.030 s/lap of each other..."}}}
```

`explanation` is written to be shown verbatim. Surface `disagreement_flagged`
prominently when true.

### `/ws/replay/{id}?speed=0` — live animation

WebSocket. Frames are `{"type":"lap", "state":{...}}` then `{"type":"complete"}`.
Drives the 3D car markers. **Non-finite values are already converted to `null`**
(a bare `NaN` is invalid JSON and poisons the whole frame — this bug has bitten us
once), so handle `null`, never `NaN`.

### Others available

`/api/session/{id}` · `/degradation` · `/track` · `/validation` · `/strategy` ·
`/regret` · `/counterfactual` · `/health-timeline` · `/narrate` ·
`/api/experiments` · `/api/business` · `/api/cross-industry` ·
`/api/physics/corner-energy` · `/api/physics/track-geometry` · `/api/ask`

---

## 5. Build this first, in this order

| Order | Work | Needs |
|---|---|---|
| 1 | Track mesh + camera + lighting | `data/geometry/` only |
| 2 | Car markers animating round the lap | geometry + `/runs` |
| 3 | Stint timeline coloured by `curve.regime` | `/runs` |
| 4 | The four question panels | endpoints above |
| 5 | Live replay wired to the markers | `/ws/replay` |

**Items 1–3 are safe to start immediately.** They depend on shapes that are not
going to move.

**Item 4 has a caveat, read it before you start:** the *layout container* is
stable, but a consolidated "state object" schema is still being designed. Build
the panels against the endpoints above and keep the data-fetch layer thin and in
one place, so a schema change is one file and not twenty components.

---

## 6. Two things that will sink the demo if they are got wrong

### 6.1 The 3D comparison's truth engine must not be our model

The demo shows two cars: one pitting on a naive estimate, one on TyreMind. The
gap at the flag is the payoff.

> **If the simulated race is driven by TyreMind *and* predicted by TyreMind, the
> demo proves nothing.** A technical judge sees it in about thirty seconds.

The car without our model has to lose **because an independent tyre truth punished
its bad estimate**. Put that in a code comment next to the simulation loop so
nobody "simplifies" it later.

Also: the two cars do **not** differ in how they drive. They differ in **when they
pit**. Same pace physics, two strategies. Our model does not make a car faster; it
produces a better estimate, which produces a better pit window, which produces a
better finishing position.

### 6.2 Do not drop the intervals to make it look clean

Covered in §1, repeated here because it is the one that will happen by accident at
2 a.m. when a chart looks busy.

---

## 7. Branch discipline

```bash
git checkout main
git checkout -b <yourname>/frontend
```

**Branch off `main`, not off `nitya/dev`.** The model branch has heavy
experiment churn and you will inherit merge pain for no benefit.

The geometry exporter currently lives on `nitya/dev`. Until it merges to main:

```bash
git cherry-pick 61baccb
```

Nobody commits to `main` directly.

---

## 8. Numbers you can put on screen today

These are measured, reproducible from `experiments/results/`, and safe to quote.

| Claim | Number |
|---|---|
| Real sessions / clean laps | 203 / 91,867 |
| Degradation-rate error vs known truth | **0.0037 s/lap** — best of 9 models |
| Closest published model's error | 0.0158 (4.3× worse) |
| Our 95% interval actually covers | **100%** — theirs 38% |
| Live interval coverage | **95.2%** over 69,206 laps |
| Naive method gives an impossible answer | **74%** of 77 races |
| Fit time | 12 s per session, laptop, no GPU |

**Do not put these on screen:** "best in the world", "perfect", or any accuracy
claim not in the table above. We are 4th of 9 on lap-time forecasting and tied
on pit timing, and a judge with the result files open will check.

---

## 9. Questions

Anything about the API shapes, the geometry format, or which endpoint feeds which
panel — ask rather than guess. A wrong assumption here costs a rebuild, and most
of these answers take thirty seconds.
