# TyreMind — what to say, and when

**Seven minutes, ten beats, one screen.** Open the app on **The pitch** and scroll
as you talk. Do not switch screens. Everything you need is on this one page.

Words in **bold** are the lines worth saying close to verbatim. Everything else
is yours to paraphrase. Lines in *italics* are stage directions, not speech.

---

## Before you start

*Have the app open at `#/pitch`, race set to Monza. Scrolled to the top.*

Say nothing about the technology in the first thirty seconds. Say what goes
wrong without it.

---

## 00 · The hook — 30 seconds

*Top of the page. Do not scroll yet.*

> **A Formula 1 race is decided by one number that nobody can measure.**
>
> **How fast the tyre is going away.** Get it right, you win the pit stop. Get it
> wrong, you lose the race in the pit lane, and you lose it to a decision you
> made three laps too early.
>
> Around ten teams in the world can measure it properly. **Everybody else is
> guessing.**

*Pause. Then scroll to beat 1.*

---

## 01 · Why it is hard — 45 seconds

*The stacked chart. Point at the orange band.*

> Here is why it is hard. **Four things move a lap time at the same time.**
>
> The orange is the tyre wearing out, slowing the car down. It grows every lap.
>
> The blue underneath is the fuel burning off. The car is getting lighter, so it
> is getting faster — by almost exactly the same amount, in the opposite
> direction.
>
> *Point at the dashed line.*
>
> **And that is what the timing screen shows you.** The two cancel out. The one
> number everybody watches barely moves, while underneath it the tyre is dying.

---

## 02 · The physics check — 45 seconds

*The car diagram. This is your credibility beat. Slow down here.*

> Before I show you what we built, let me show you how you know the physics
> underneath is real.
>
> These are the four tyres on the car, seen from above. The darker ones did more
> work. **Lean on right-hand corners all lap and the left-hand tyres carry the
> load.**
>
> *Point at the green box.*
>
> So from tyre wear alone, our model says this circuit runs clockwise. It does
> run clockwise. **We never told it. It works that out on seven of the eight
> circuits we tested.**
>
> **If the physics underneath were wrong, it could not do that.**

---

## 03 · What we built — 40 seconds

*The narrowing band.*

> Now the product. We separate the four causes and report only the tyre.
>
> *Trace the band left to right with your finger.*
>
> **Watch the grey band close.** On the left the model has seen almost nothing,
> so it is honest about being unsure. As the laps come in, it narrows.
>
> That is a model learning from evidence rather than asserting a number. **It is
> the difference between a guess and a decision.**

---

## 04 · The forecast — 30 seconds

*The fan.*

> Then it tells you where the car will be in three laps, and in fifteen.
>
> **Notice the band gets wider the further out it looks.** It should. Anyone who
> draws you a confident line twenty laps into the future is lying to you, and a
> strategist finds that out the hard way.

---

## 05 · The decision — 40 seconds

*The pit window curve. This is what they buy.*

> And this is the thing a team actually pays for.
>
> Every lap he could stop on. For each one we simulate the rest of the race over
> and over and count how often that lap turns out to have been right.
>
> *Point at the dip, then the shaded block.*
>
> **The lowest point is the answer. The shaded block is the window a strategist
> keeps open.** Not a number — a window. That is how the decision is actually
> made on a pit wall.

---

## 06 · Why a simpler tool cannot do this — 30 seconds

*The four bars.*

> You might ask why this needs anything clever. Here is why.
>
> We measured the shape of nearly three thousand real stints. **Only just over
> half fade evenly.** The rest warm up, recover, or fall off a cliff — and the
> cliff is the one that ends races.
>
> **Every competing model assumes a straight line.** So it is the wrong shape for
> almost half of real racing. Ours was never told what shape to expect.

---

## 07 · Trust, part one — 30 seconds

*The consensus rows.*

> Two reasons you can trust the number.
>
> First: we do not ask the model once. We re-run it under deliberately different
> assumptions about the things nobody can measure.
>
> *Point at the overlap.*
>
> **When four disagreeing starting points land on the same answer, that answer
> belongs to the race, not to our choices.** When they scatter, we say so instead
> of picking the one we like.

---

## 08 · Trust, part two — 50 seconds

*The calibration diagonal. This is your strongest beat. Do not rush it.*

> Second, and this is the one I would want to hear if I were you.
>
> The dashed diagonal is a tool being exactly as confident as it deserves to be.
>
> *Point at the orange lines sagging below it.*
>
> **Every competing model sits below that line. They all claim more certainty
> than they have earned.** That is the failure that gets a strategist hurt.
>
> **Ours did too.** It told us it was right ninety-five times in a hundred. We
> checked it against sixty-nine thousand real laps. It was right seventy-five.
>
> So we rebuilt the confidence from measured outcomes instead of theory. *Point
> at the green pair on the diagonal.* **That is us now.**
>
> **Almost nobody runs this test. That is why almost everything on the market
> overstates how sure it is.**

---

## 09 · Bigger than racing — 30 seconds

*The scatter.*

> One more thing. Strip the motorsport words away and this is a general problem.
> **Something wears out while it works, you cannot measure it directly, and other
> things move the only signal you can see.**
>
> We pointed the identical code at NASA's jet engine benchmark without changing a
> line. Each dot is an engine. **It is not a Formula 1 trick.**

---

## 10 · The business — 60 seconds

*The four pillars, then the closing line.*

> So who buys it.
>
> **Roughly ten teams in the world can do this today** — they have departments,
> licensed data and budgets. **Every series below them makes the same pit calls,
> on the same compounds, from the same public timing feed, with nothing.**
>
> F2, F3, F4, regional single-seaters, GT, endurance privateers, club racing.
>
> **The capability is not new. It has simply never been cheap enough to reach
> them.**
>
> *Point at the pillars.*
>
> It runs on a laptop a team already owns. No GPU, no cloud bill, and the data it
> needs is public. It needs four fields — lap time, tyre type, tyre age, lap
> number — and **every series already publishes all four.**
>
> **And nothing leaves the garage.** No upload, no vendor sees their data. In
> motorsport that is not a feature. It is the condition of sale.

---

## The close — 20 seconds

*Stop scrolling. Look up.*

> Everyone can build a dashboard that shows a tyre wear number.
>
> **We built one that is right, that tells you how confident it is and is
> genuinely right that often, and that we then tried very hard to break
> ourselves.**
>
> **Thirteen of our own ideas failed. We published all thirteen.** Including a
> far more advanced physics model we built specifically to try to beat our own
> system. It lost, so we did not ship it.
>
> **We would rather you checked us than believed us.**

---

# If they interrupt

Short answers. Do not over-explain. Offer the screen.

**"Is this live or a recording?"**
> Live. *Change the race in the sidebar.* Every chart on the page just refitted.

**"How do you know you are right, if nobody publishes real tyre wear?"**
> We do not need the right answer to prove a method is wrong. A method that says
> tyres get faster as they wear has disproved itself, and that happens in about
> three races out of four. That result needs no trust at all.

**"What is your accuracy?"**
> Four point three times better than the closest published research model at
> recovering the wear rate. But the number I would rather you asked about is the
> confidence one — it is on screen, and we are the only ones who measured it.

**"Where are you weakest?"**
> A simpler model forecasts raw lap times better than we do, and we send that
> question to it rather than pretending otherwise. On real pit stops we tie with
> the closest published research rather than beating it. Both are on the evidence
> screen.

**"Who is your competition?"**
> Inside Formula 1, the teams' own departments, and we are not trying to displace
> those. Below Formula 1 there is nothing, which is the whole business.

**"How long did this take / is it finished?"**
> It runs offline from a clean clone today. Six hundred and fifteen automated
> tests, thirty-two recorded experiments, every result file in the repository.

**"What would you do with investment?"**
> Two things. Get it into one junior championship as a pilot, and build the
> hierarchical layer that lets a compound parameter carry across a season
> properly — we tested one version of that and it failed, and the result file
> says so.

---

# Things not to say

- Do not say **Kalman filter**, **state-space**, **conformal** or **posterior**
  unless someone asks a technical question first. They are correct and they end
  a non-technical listener's attention.
- Do not claim to beat a professional strategist. The measured margin is small
  and the honest claim is against the standard method.
- Do not oversell the market. You have no revenue figures and inventing one is
  the fastest way to lose a room that has been checking everything else.
- Do not skip the failures. Thirteen refuted ideas is your credibility, not your
  weakness.

---

# Timing

| Beat | Target |
|---|---|
| Hook | 0:30 |
| 01 Why it is hard | 0:45 |
| 02 Physics check | 0:45 |
| 03 What we built | 0:40 |
| 04 The forecast | 0:30 |
| 05 The decision | 0:40 |
| 06 Why simpler fails | 0:30 |
| 07 Trust, consensus | 0:30 |
| 08 Trust, calibration | 0:50 |
| 09 Bigger than racing | 0:30 |
| 10 The business | 1:00 |
| Close | 0:20 |
| **Total** | **7:30** |

If you are cut to five minutes, drop beats 04, 06 and 09. Never drop 02, 08 or
10.
