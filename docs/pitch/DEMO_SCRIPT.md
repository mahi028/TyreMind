# TyreMind — Live Demo Runbook

**What to click. What to say. What number appears. What to do when it doesn't.**

Companions: `docs/pitch/STORY.md` (the arc), `docs/pitch/JUDGE_QA.md` (questions).

**One rule above all others: read the number off the screen, not off this page.**
Every figure below is what the last rehearsal produced. If the screen disagrees,
the screen is right and you say the screen's number. A presenter who reads a
memorised number that contradicts the display loses the room permanently.

---

# 0. T-minus checklist

Do this **the day before**, not on the day.

| When | Do | Why |
|---|---|---|
| **Day before** | `cd "D:\TrackShift Innovation Challenge"` then `pip install -r requirements.txt && pip install -e .` | The server will not start otherwise. |
| **Day before** | `cd apps/web && npm run build` | The shipped `dist/` bundle predates the latest CSS and the new pit-confidence fields. Build it *now*, never on the day. |
| **Day before** | `pytest -q` | 81 tests. This is your fallback if the server dies. Know that it passes. |
| **T-20 min** | Launch (§1) and wait for `8 ready` | Cold start warms 8 sessions in ~22 s. |
| **T-15 min** | **PRE-WARM THE HIDDEN CACHE** (§1.2) | **The single biggest live risk in the product.** Read §1.2. |
| **T-10 min** | Confirm the header status dot reads **"runs offline"** in green | Your offline claim is on screen. If it reads "needs network", something is missing from `data/demo/`. |
| **T-5 min** | Open a second browser tab on `/demo.html` and leave it on the gate screen | Your instant fallback. Do not start it. |
| **T-5 min** | Unplug the wifi | Seriously. Then say so in the room. It is the cheapest credibility you will ever buy. |

---

# 1. Starting it

## 1.1 The command

From the **repo root** — this matters, every data path in the app is relative:

```bash
cd "D:\TrackShift Innovation Challenge"
python -m tyremind.serve
```

Serves API **and** dashboard from one process on **http://127.0.0.1:8077** and opens
a browser after ~1.2 s.

Banner to wait for:

```
sessions 8 (8 cached locally)
warming  fitting cached sessions...
         8 ready in 22.4s
dashboard http://127.0.0.1:8077
api docs  http://127.0.0.1:8077/docs
```

**If your setup uses port 8000** (the raw uvicorn path some handoff docs describe):

```bash
python -m uvicorn tyremind.api.main:app --host 127.0.0.1 --port 8000
```

That path does **not** print the warm-up banner and does **not** pre-fit anything,
so §1.2 becomes mandatory rather than merely wise. Flags for the packaged server:
`--host`, `--port`, `--no-browser`, `--no-warm`.

> **Launch from the wrong directory and nothing errors.** The session list, the 3D
> geometry and every evidence panel just come up empty. If a screen is blank, check
> your working directory before you check anything else.

## 1.2 The pre-warm — do not skip this

The advanced routes (`Tyre twin`, `When to pit`, the narration on `Why is the car
slow`) hold **their own model cache, separate from the one the start-up banner
fills**. The banner says "8 ready". Those three screens are not.

So the **first** click on each of them triggers a full cold re-fit — **6 to 15
seconds of "Fitting the session…"** — and on `Tyre twin` the robustness panel
re-fits the model **three more times** under perturbed priors on top of that.

**Before the judges are in the room, on Monza race (and on every other session you
intend to show):**

1. Click **Tyre twin**. Wait until "Should you believe this?" finishes.
2. Click **When to pit**. Wait until the recommendation renders.
3. Click **Why is the car slow**. Wait for the "In plain language" paragraph.

It caches per session. Do it once and the demo is instant.

**If you get caught cold on the day**, do not apologise for the spinner. It reads
**"Fitting the session…"**, not a generic wait. Say:

> *"That is the model actually fitting, right now, on this laptop. Twelve seconds, no
> GPU, no cloud. Nothing on this screen was precomputed for you."*

The honest latency is a better moment than a hidden one.

---

# 2. The failure playbook — read this before the script

Know these six before you know the script. A demo is judged on how you handle the
thing that breaks.

| Symptom | Cause | Do this |
|---|---|---|
| **Screens blank, session list empty** | Launched from the wrong directory | `cd` to repo root, relaunch. 20 s. |
| **A screen sits on "Fitting the session…"** | The §1.2 cold cache | **Do not click again.** Talk over it using the line in §1.2. It will land inside 15 s. |
| **3D canvas is black, no error** | WebGL context exhausted after many circuit switches | Hard-refresh the tab (Ctrl+Shift+R). Do not switch circuits on that screen again. |
| **Live replay never finishes** | You picked a race session at slow pace — Zandvoort race is 1,350 laps ≈ 68 s | Drag the **pace** slider to **max** *before* pressing Start. It cannot be moved once streaming. Or use Monza **FP2** (143 laps ≈ 7 s). |
| **"When to pit" is slow or the window panel vanishes** | `/pit-window` sweeps every remaining lap at 1,200 sims each, with no debounce on the slider; and it correctly returns 400 at the end of a stint | Move the lap slider in **one deliberate drag**, or click the track once. Do not scrub. |
| **Server will not start at all** | Environment | Run `pytest -q` on screen. 81 tests. Say: *"The science is in the library, not the browser. Here it is, passing. Let me walk you through the code and the result files instead."* Then open `experiments/results/` and the STORY. |
| **Browser is broken but the server is up** | Frontend build | Go to **`/docs`**. The FastAPI interactive docs hit every real endpoint and return real numbers. Fire `/api/session/2024-monza-R` live. It is less pretty and equally convincing. |
| **Anything at all, and the clock is running** | — | Open the second tab: **`http://127.0.0.1:8077/demo.html`**, press **Space**. A self-driving 4:55 tour that captions itself and skips any control it cannot find. Press **R** to restart, **F11** first for fullscreen. |

**No network at any point is not a failure.** It is the feature. Say so out loud
the first time anything loads.

---

# 3. The run

Three cuts. Pick by the clock you are given.

| Cut | Beats | Use when |
|---|---|---|
| **Short — 3 min** | 1, 2, 7 | You have a booth slot or the panel is behind schedule |
| **Core — 8 min** | 1–8 | The default. Rehearse this one. |
| **Extended — 12 min** | 1–8 plus §3.9 and §3.10 | A technical judge is leaning forward |

**Session to open on: `2024 Monza · R`** — it is the default on load, its data
quality scores 100/100, and it is the race where the naive number goes negative.

Every view is deep-linkable if you need to jump: `#/overview`, `#/explain`,
`#/circuit`, `#/tyre`, `#/strategy`, `#/live`, `#/evidence`, `#/ask`, `#/beyond`.

---

## Beat 1 — "Start here" · 60 s · the problem, in one screenshot

**CLICK:** nothing. This is the landing view. Left rail: session **2024 Monza · R**,
view **Start here**.

**POINT AT:** the panel titled **"Why the obvious method fails"**, the two numbers
side by side, per compound.

**WHAT APPEARS:** the left number is **"Lap time vs tyre age"** — the naive fit —
printed in orange with the caption ***"tyre getting faster — impossible"*** when it
goes negative. The right number is **TyreMind**, with `± sd` and a 95% interval
beam. Last rehearsal: naive **−0.004 s/lap** on the hard; TyreMind **+0.074** hard
and **+0.113** medium, correctly ordered. **Read the screen.**

**SAY:**

> "A lap time is a bill with no itemisation. You paid 95 seconds. How much was the
> tyre, how much was the fuel burning off, how much was the track rubbering in?
>
> Here is the first thing anyone tries — fit a straight line through lap time
> against tyre age. On this race it says the hard tyre degraded at **minus** four
> thousandths of a second per lap. Negative. It says the tyre got *faster* the
> longer it ran.
>
> That is not a rounding error. It is the wrong sign. And it is not a one-off — we
> counted it across every dry race in our corpus, **77 of them**. The naive method
> comes out negative in **74% of races**, and in **53% of the 208 individual
> compound-stints**.
>
> Here is why. The car burns fuel, gets lighter, and speeds up by about **0.081
> seconds a lap**. Tyre degradation is **0.03 to 0.08**. The confounder is the same
> size as the signal, so when you don't remove it, the answer flips sign about half
> the time.
>
> Ours, on the right: plus 0.074 for the hard, plus 0.113 for the medium — the
> softer tyre degrading faster, which is the only ordering physics allows — and
> each one with an interval."

**IF A JUDGE ASKS "does it ever go the right way by accident?"** — yes, and say so:
Barcelona is the demo race where the naive fit does not go negative. That is not the
method working. That is the confounders happening not to swamp the signal there.
Whether the standard approach gets the *sign* right is a matter of luck.

**THEN CLICK:** the one button on the screen — **"See where the lap time actually
went"**.

---

## Beat 2 — "Why is the car slow" · 90 s · **the moment**

This is the best 90 seconds in the demo. Do not rush it.

**CLICK:** a run chip at the top (`● DRIVER 41L`) — pick a long stint. Then let the
**"Peel the confounders away"** animation auto-play. It advances every 1.6 s through
four labelled steps; you can also click them in any order.

**WHAT APPEARS, in order:**

1. **"Lap times as driven"** — the grey line. What the stopwatch saw.
2. **"Fuel burn-off removed"**
3. **"Track evolution removed"**
4. **"Traffic removed"** — and the dashed line, the model's tyre estimate, is now
   sitting on the solid de-confounded line.

**SAY, over the animation:**

> "Grey is what the stopwatch saw. Each step removes one cause. Watch the line turn
> over."

**THEN READ THE "In plain language" PANEL OUT LOUD.** It is generated from model
output, not written by us. Last rehearsal it said:

> *"Car RIC was 1.78 seconds FASTER on lap 53 than on lap 13, so a stopwatch says
> the tyre is fine. It is not. Strip out the things that changed around the car and
> the hard tyre has actually lost 2.15 seconds over those laps. The car got faster
> because it burned off fuel, and the fuel gain was bigger than the tyre loss.
> Watching lap times alone would miss a degrading tyre completely."*

**THEN SAY — this is the line the whole pitch is built on:**

> "A car that got a second and a half **faster** while its tyre was dying. Nothing
> that reads lap times can see that. That is the entire problem, in one stint."

**SCROLL TO:** **"Where the time went, lap by lap"** — the stacked bars. Point at
the band labelled **"Unexplained"**.

> "Tyre, fuel, track evolution, traffic — and then that band. That is the residual,
> and we leave it on the chart on purpose. Any decomposition that adds up perfectly
> is hiding something."

---

## Beat 3 — "Where it wears" · 45 s · the 3D, and the physics check

**CLICK:** **Where it wears**. Wait for the WebGL canvas.

**WHAT APPEARS:** the real Monza racing line in 3D, from GPS, with real elevation
exaggerated 6×, a green sphere at start/finish and a car marker moving round it
every 9 s. Header aside shows the lap it is drawn from (e.g. `NOR · 81.432s`).
Legend: `420 telemetry points`.

**CLICK:** the three colour-mode buttons in turn — **"Tyre loading"** (default),
**"Speed"**, **"Cornering force"**.

**SAY:**

> "This is the real racing line, from GPS, with real elevation. Coloured by how hard
> the tyre is being worked.
>
> The reason it is here is not decoration. Left tyres carry **54.9%** of the
> frictional energy at Monza. We never told the model which way Monza runs — it
> worked out **clockwise** from the loading alone, and it gets circuit direction
> right on **7 of 8 circuits**.
>
> That is a physics sanity check that uses **no tyre-wear data at all**. If the
> energy layer were wrong, it would get the direction wrong."

**CONTROL NOTE:** the right-hand button toggles **"Pause" / "Rotate"** for the
orbiting camera. Use it if the motion is distracting on a projector.

**RISK:** do **not** switch sessions while on this screen. Repeated circuit switches
exhaust the browser's WebGL contexts and the canvas goes black with no error.

---

## Beat 4 — "Tyre twin" · 45 s · the metric the customer actually wants

**CLICK:** **Tyre twin**. (Pre-warmed at §1.2, so it is instant.)

**WHAT APPEARS:** an overhead car diagram, four tyres labelled `FL FR RL RR`, each
showing its **% of frictional energy**; beside it **Tyre health `x / 100`** with the
translation *"0.33 s/lap slower than a fresh set"*, plus compound, age, and the
left/right and front/rear splits.

**SCROLL TO:** **"How much life is left"**.

**SAY:**

> "Competitive laps left — **with an interval**, 'between X and Y', never a single
> number. And look at the last column of that table: **applies**. As the projection
> reaches further past what this session actually contains, applicability falls off
> and the model says so.
>
> One thing we deliberately do **not** print: millimetres of tread. No public
> measurement of F1 tread depth exists to anchor a millimetre against, so a number in
> millimetres would be invented precision. We report **percentage of usable life**,
> anchored to the degradation cliff we measured — median **71.9% through a stint**
> across 2,827 real stints. We would rather be right than precise."

**IF TIME:** scroll to **"Should you believe this?"** — the applicability bar, the
risk reasons, and **"What would most reduce the uncertainty"**, which ranks the top
three signals by how much each would cut the error.

> "It does not just report uncertainty. It tells you what to go and measure to
> shrink it."

**RISK:** do not change session on this screen. `/trust` re-fits under three
perturbed priors and it is the slowest call in the product.

---

## Beat 5 — "When to pit" · 60 s · the decision

**CLICK:** **When to pit**. Pick a run chip.

**CONTROL:** the **"Deciding at lap"** range slider — the only slider of its kind in
the app. **Move it in one deliberate drag or click a position once.** Every
intermediate value fires a fresh `/pit-window` sweep of 1,200 simulations per
remaining lap, and there is no debounce.

**WHAT APPEARS:** a recommendation in orange, tagged **"simulated, not observed"**;
Stats **Wins how often** (%) and **By** (s); the **"Cost of stopping 5 laps later
than recommended"** in seconds; a **"Why"** list of reasons; then the pit-window
sweep with an optimum lap marked and a shaded one-second window; then the overlaid
outcome distributions.

**SAY:**

> "Five thousand simulated races per option. And the important detail — we draw a
> **different degradation rate from the posterior for every simulated race**. So this
> spread is real uncertainty about the tyre, not just lap-time noise.
>
> Look where the curves overlap. There, the model is telling you the choice
> genuinely does not matter. A system that can say 'this decision is not load-bearing'
> is as useful as one that says it is.
>
> And here is the number a pit wall actually understands — stopping five laps later
> than recommended costs **[read the screen]** seconds. That is what model accuracy
> is worth, in the only unit that matters."

**THE STRONGEST FOLLOW-ON — say it here:**

> "We tested this against reality. **274 real pit stops**, across 14 sessions, scored
> against what professional strategists actually did on the day. On the stops every
> model answered, we are **5.92 laps** from the real call.
>
> The pooled regression — the model that beats us at forecasting lap times — comes
> **last** at **11.12 laps**. Forecasting and deciding are different jobs."

---

## Beat 6 — "Live monitor" · 60 s · it runs forward

**SET THE PACE SLIDER FIRST.** It is **disabled once streaming starts**. Default is
`50 ms`; `max` (0) finishes instantly.

| Session | Laps | At 50 ms |
|---|---|---|
| Monza **FP2** | 143 | **~7 s** ← use this if you are behind the clock |
| Monza race | 921 | ~46 s |
| Barcelona race | 1,204 | ~60 s |
| Zandvoort race | 1,350 | ~68 s ← avoid |

**CLICK:** **"Start live replay"**.

**WHAT APPEARS:** a status word (`connecting → streaming → complete`), a progress
bar, the aside counting `lap 412 of 921`, a live table of `car / compound / age /
degradation rate / health` with an interval beam on every row, and a
**"Confidence, as laps arrive"** chart showing the uncertainty collapsing.

**SAY, while it runs:**

> "Same model, running forward only, one lap at a time, with **no access to the
> future**. This is not a replay of a result — it is the estimator working.
>
> Watch the confidence chart. The bands are collapsing as laps arrive. And every row
> in that table carries its interval — there is not a bare point estimate anywhere in
> this product."

**WHEN IT COMPLETES,** four Stats appear: **Laps processed**, **Model states**,
**Mean update (ms)**, **Worst update (ms)**. **Read the mean off the screen** — last
rehearsal it was **0.22 ms**.

> "**[read] milliseconds per update**, and the cost is flat in session length — lap
> 60 costs exactly what lap 1 cost.
>
> That is why we chose a Kalman filter over MCMC sampling. A sampler has no
> incremental mode. You cannot re-run one every lap on a pit wall during a race."

---

## Beat 7 — "Does it work" · 90 s · **the beat that wins technical depth**

**CLICK:** **Does it work**. Every panel here reads from committed JSON in
`experiments/results/`.

### 7a — Lead with identifiability (30 s). Not with accuracy.

**POINT AT:** **"What can and cannot be identified"** — three collinearities, each
tagged either **identified from data** (green) or **resolved by assumption** (amber).

**SAY:**

> "Most tools skip this panel. Inside a single stint, tyre age goes up by one every
> lap. Laps of fuel burned also goes up by one every lap. They move together
> **exactly** — we measured it, **520 runs out of 520**, correlation **1.000**, and
> the Fisher information matrix is singular in **11 sessions out of 11**. The
> likelihood surface has a perfectly flat direction. More data does not fix it. It is
> algebra.
>
> There is exactly one escape. A tyre curve **bends** — warm-up early, cliff late.
> Fuel burn-off is a straight line and never bends. That curvature is the only
> data-driven channel, and it carries **6% of the information**. The other **94%
> comes from the physics prior**.
>
> We also computed the **Cramér–Rao bound** — the best precision any unbiased
> estimator could ever achieve here. We sit at **1.38 times** it. So we can tell you
> our answer, *and* how much room is left above us.
>
> **No paper in our 28-paper review states this number.** The closest published model
> to ours has the identical structure and resolves the problem with a prior it never
> quantifies."

### 7b — Then the accuracy (30 s)

**SCROLL TO:** **"Can it recover a degradation rate it was never shown?"** and
**"Against every reasonable alternative"**.

**SAY:**

> "When a true answer does exist, we find it. **[read the screen]**. And on this
> table" — *point at the model ladder* — "read the callout: **the best lap-time
> predictor cannot answer the question**. Three of the models in our full benchmark
> have no degradation parameter at all. They forecast well and they cannot itemise a
> single line of the bill."

### 7c — Then the calibration, and the loss (30 s)

**SCROLL TO:** **"Does a 95% interval contain the answer 95% of the time?"**

**SAY:**

> "Our own model said 95% and delivered **75.5%**. We only know that because we
> checked. We fixed it with conformal prediction — now **94.7%** offline, and live,
> with the interval correcting itself during the race, **95.2% over 69,206 laps**.
> That is **0.2 percentage points from target**.
>
> The honest interval is **2.3 times wider** than the dishonest one. We shipped the
> wide one.
>
> And I will give you the loss on the same screen. On plain lap-time forecasting we
> are **4th of 9**. The pooled regression beats us at 0.4088 to 0.6484. We
> **pre-registered** that we expected to lose that leg before we ran it — the file is
> `experiments/PREREGISTRATION_exp19.md`, committed before the results existed. And
> that same pooled regression comes **last** on pit-stop timing.
>
> An independent published system, Pitwall, reports the same phenomenon and calls it
> *'calibration-optimal is not decision-optimal'*. We found it separately."

### 7d — ⚠ The panel to walk past, or caveat hard

The evidence screen contains **"Does a Friday curve predict Sunday?"** — the exp03
practice-to-race panel, showing 0.0807 vs naive 0.1471. The UI already flags it with
a box reading *"An honest finding, not a clean win"*. **Do not build a claim on it.**

That test is **circular** — the "actual" it scores against is our own model's race
fit. And the pre-registered de-circularised replacement, `exp27`, is **already in the
repo** and currently returns **negative skill for every model including ours** on its
first 2-event partial run (pre-registration specifies 49).

**If a judge stops on that panel, say this and move on:**

> "That one is our weakest evidence and the screen says so. It scores our practice
> estimate against our own race fit, so it shows we agree with ourselves — it is not
> proof. We pre-registered the fix, it is running, and on the first two events it
> comes back negative for **every** model in the ladder, ours included. If that holds,
> the finding is that Friday does not predict Sunday for anyone, which is a bigger
> result than our 45% and we would report it. Nothing else on this screen depends on
> it."

Full handling in `JUDGE_QA.md` Q6.

**SCROLL TO:** **"Is the degradation curve a straight line?"**

> "And the field-standard model is a straight line with two coefficients. We measured
> 2,827 real stints: only **55.9%** are linear. It is the wrong shape for 44% of
> them."

---

## Beat 8 — "Beyond racing" · 45 s · why it is a business

**CLICK:** **Beyond racing**.

**WHAT APPEARS:** NASA C-MAPSS results — **Engines scored**, **Remaining-life
error** (cycles), **Sensors used** (`12 of 21`), **Predicted early** (%); a
per-engine scatter against the diagonal; three asset-profile cards each tagged
**validated** or **architecture only**; and a fleet panel that opens with the words
**"This is not a result."**

**SAY:**

> "Public F1 data has no measured tyre wear, so F1 cannot fully prove this works.
> So we ran the **identical** estimator on NASA's turbofan benchmark, which does have
> run-to-failure ground truth. We changed nothing. Tyre stint becomes engine life,
> tyre age becomes flight cycles.
>
> **22.67 cycles** remaining-life error on all 100 test engines, 44% predicted early
> — the safe side for maintenance.
>
> Purpose-built deep models on that dataset reach **15.98**. We are 30 to 40% worse
> and I am putting that on the slide. **The claim is transfer, not state of the
> art.** A tyre model pointed at a jet engine with no retuning should not work at
> all.
>
> And notice this box" — *point at "This is not a result"* — "the commercial fleet
> arithmetic is labelled illustrative, by us, before anyone asks."

**CLOSE HERE** with the sentence from `STORY.md`:

> "Everyone in this field validates lap-time prediction. Almost nobody checks whether
> the **reason** is right — and the reason is the actual question, because many wrong
> decompositions add up to the same right total.
>
> We are not claiming to be the best tyre model in the world. We are claiming to be
> the only one in this room that can prove **how wrong it is**. That claim is
> provable. 'Best in the world' is not."

---

## 3.9 Extended beat — "Ask the method" · 45 s

**CLICK:** **Ask the method**. A question auto-runs on mount. Seven suggested-question
buttons are on screen; the sharpest to click live is **"Where does a simpler model
beat you?"** or **"Which assumptions carry the result?"**.

**SAY:**

> "Ask it something adversarial. Every answer comes back as a **verbatim passage with
> its source file**, tagged either *measured result* or *documentation*. Nothing is
> generated. There is no language model in this path at all — it is BM25 plus a
> TF-IDF index, deliberately, in a product whose central promise is that it runs with
> the network unplugged.
>
> A chatbot would happily invent a number here. This cannot."

**INVITE A JUDGE TO TYPE.** It is the strongest trust move available and it costs
15 seconds.

## 3.10 Extended beat — prove nothing is hard-coded · 45 s

Drop to a terminal:

```bash
python experiments/exp01_ground_truth_recovery.py --n-seeds 5
```

**SAY:**

> "About a minute. It rewrites `experiments/results/exp01_ground_truth_recovery.json`
> — which is the file the evidence screen you just saw reads from. **No number in
> this product is typed by hand.** Refresh the tab and watch it change."

---

# 4. Screen-versus-deck reconciliation — **read this or get caught**

The dashboard's evidence screen reads **exp01, exp03, exp05, exp16, exp17**. The
deck's headline benchmark is **exp19**, the nine-model pre-registered comparison,
which the shipped UI does **not** display. Both are real, committed, and different
experiments. A judge with both in front of them will notice.

| Quantity | **On screen** (exp01 / exp05) | **In the deck** (exp19) | What to say if challenged |
|---|---|---|---|
| Degradation-rate recovery, ours | **0.0044** (exp01, 25 seeds) | **0.0037** (exp19, 8 seeds, 24 estimates) | "Two experiments. exp01 is the 25-seed recovery check the dashboard reads. exp19 is the nine-model pre-registered benchmark with its own seeds. Different runs, both committed." |
| Degradation-rate recovery, naive | **0.0966** (exp01) | **0.0725** (exp19) | Same answer. |
| Best lap-time CRPS | **0.395** pooled (exp05, 6 models, 20 races) | **0.4088** pooled (exp19, 9 models, 12 races) | "exp05 is the earlier six-model ladder. exp19 is the pre-registered nine-model version with Heilmeier, ARIMA and Cappello & Hoegh added." |
| Our lap-time CRPS | **0.645** (exp05) | **0.6484** (exp19) | Same answer. |
| Our rank on lap time | 3rd of 6 | **4th of 9** | **Always say 4th of 9.** It is the harder number and it is the pre-registered one. |

**The safe universal answer:**

> *"That is exp05, the six-model ladder, which is what this screen reads. The number
> in our deck is exp19 — the nine-model version we pre-registered, which added the
> two published baselines. Both files are in `experiments/results/`. Happy to open
> either."*

---

# 5. Questions that arrive mid-demo

Full set in `docs/pitch/JUDGE_QA.md`. These five interrupt the demo specifically:

| Asked at | Question | One-line answer |
|---|---|---|
| Beat 1 | *"Isn't the naive method a strawman?"* | "The closest published model to ours had to add a positivity prior specifically to stop this happening. It is a real failure in real modelling, not one we built to knock down." |
| Beat 5 | *"Where does the 'real strategist' benchmark come from?"* | "Actual pit laps from 14 real sessions, 274 stops, 93 excluded and listed in the result file. The reference is what the pit wall did, not what our model thinks." |
| Beat 6 | *"Is the live replay actually online, or scripted?"* | "Filtered, not smoothed — the label is on the panel. No future lap reaches any estimate. There is a batch smoother too, and the two agree to 0.0006 s/lap." |
| Beat 7 | *"If 94% is the prior, aren't you just reporting your assumption?"* | "Test it: exp02 moves the prior a full standard deviation both ways and the conclusion does not move. And on NASA the data pulled the rate to 0.00202 against a prior mean of 0.004 — a factor of two, away from the prior." |
| Beat 7 | *"Your practice-to-race test scores you against yourself."* | "Correct, it is circular and the screen says so. The pre-registered fix is in the repo and currently negative for every model including ours, on 2 of 49 events. §7d." |
| Beat 8 | *"So you lose on NASA?"* | "Yes, on point accuracy, by 30 to 40%. We are a tyre model with nothing retuned. The claim is transfer." |

**If asked something not built:** *"We have not tested that yet. It is on the list."*
That costs nothing. A guess costs everything. `docs/research/13_LIMITATIONS_AND_FAILURE_MODES.md`
is the register of what was cut and why.

---

# 6. Never say these on the day

| ❌ Do not say | ✅ Say instead |
|---|---|
| "Best in the world" / "perfect" / "solved" | "The only one here that can prove how wrong it is." |
| "We beat five other models." | "We are best at recovering the wear rate. At plain lap-time forecasting we are 4th of 9." |
| "Our practice-to-race test proves we are right." | "It shows we agree with ourselves. It is a consistency test, not proof." |
| "Track shape is harmful." | "Track shape has no detectable effect at p = 0.22. Only thermal features were harmful." |
| "We measure tread in millimetres." | "Percentage of usable life. Depth is not measurable from public data." |
| "We predict puncture probability." | "A structural exposure index. There are 17 tyre failures in 12 seasons — no probability is calibratable from public data, by anyone." |
| "Drivers differ in tyre care." | "A tiny real effect, teammate-to-teammate only, and it does not improve a forecast." |
| "The community uses the broken method." | "The simple method — the first thing anyone tries." |

---

# 7. The 90-second version, if that is all you get

1. **Start here, Monza race.** Point at the negative number. *"The standard method says the tyre got faster. 74% of 77 real races."* (30 s)
2. **Why is the car slow.** Let the peel-away play. Read the narration. *"A car that got a second and a half faster while its tyre was dying."* (40 s)
3. **Does it work.** Point at identifiability. *"6% of the answer is data, 94% is physics, and we are within 1.38× of the theoretical floor. We are 4th of 9 at forecasting and we say so."* (20 s)

Stop. Offer the rest.
