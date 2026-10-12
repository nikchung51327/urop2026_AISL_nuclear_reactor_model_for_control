# Validation and constraints notes

What the model reproduces, what it deviates from, what has been settled, and
what it must not be used for. Two source papers:

- **[V]** Vajpayee, Becerra, Bausch, Deng, Shimjith, Arul (2020).
  *Dynamic modelling, simulation and control design of a pressurized water-type
  nuclear power plant.* **Nucl. Eng. Des. 370**, 110901.
- **[L]** Li, G. (2014). *Modeling and LQG/LTR control for power and axial power
  difference of load-follow PWR core.* **Ann. Nucl. Energy 68**, 193–203.

The model takes **[V]'s plant whole** and replaces his single-point core with
**[L]'s two-point axial core**. So the two registers below ask different
questions: for [V], *did the plant survive the transplant?*; for [L], *is the
core actually his?*

**How to read this file.** Section 2 is everything that is **closed** — stated
once, concisely, with the evidence that closed it. Section 3 is what is still
**open**. Sections 4–5 are deviations that remain by choice. Section 6 is the
**constraints on use**, and is the section to read before quoting any result
from this model in a safety or licensing context.

---

## 1. Validation result

`validate_vajpayee.py` reproduces [V] §7: a single input perturbed at t = 20 s,
16 variables logged over 200 s, four cases. Run with
`Config(two_point=False, xenon=False)` — the axial split collapsed and the
poison removed — because [V]'s core is single-point with no xenon, and that is
the honest comparison.

| | perturbation | [V] section |
|---|---|---|
| Case A | control rod, reactivity step (80 pcm) | 7.1 |
| Case B | governor valve coefficient, +5 % step | 7.2 |
| Case C | pressurizer heater, +50 % pulse 20–100 s | 7.3 |
| Case D | feedwater temperature, ramp +5 % 20–100 s | 7.4 |

**Result: 40 / 40 direction agreements** with the shipped defaults.

| variant | flag | score |
|---|---|---|
| **shipped** (surge an input, `pzr_partial_scale = 1e3`) | — | **40/40** |
| eq. 32 surge additionally computed | `--surge` | 37/40 |
| Table 1 *and* eq. 34 exactly as printed | `--printed` | 30/40 |
| fitted pressurizer scale | `--scale 700` | 40/40, and matches Fig. 11e to 4 dp |

Magnitudes, not just directions:

| variable | case | model end | paper end |
|---|---|---|---|
| `Pn` | A | 1.018 | 1.018 |
| `Thot` | A | 329.25 | 329.2 |
| `Tcold` | A | 298.36 | 298.3 |
| `Ts` | A | 289.304 | 289.3 |
| `ps` | A | 7.411 | 7.410 |
| `_Tf` (core mean) | A / B / C / D | 634.04 / 635.08 / 626.71 / 619.18 | 634.0 / 634.0 / 626.7 / 619.5 |
| `P_tur` | all four | exact | — |

Fuel temperature needs one note. [V] has a single fuel node; the two-point core
has one per half, and with `two_point=False` the halves are still thermally in
**series**, so they straddle his value (619.06 and 634.23 against his 626.66).
The arithmetic mean is 626.64 — a 0.02 °C miss — and that is the like-for-like
comparison, so the panel plots the mean.

Figures: `validation_fig10.png`, `validation_fig11.png`, plus `_printed`,
`_surge` and `_eq34printed` variants.

---

## 2. Resolved

Seven threads that were open during development and are now closed. Each is
stated once here; the long narratives they replace have been removed.

### 2.1 Surge is an input, and [V]'s Fig. 11e was made with eq. 32 disabled

**The problem.** [V] prints eq. 32, which *computes* surge flow from ten coolant
nodes' temperature rates. But his §8.3.1 *injects* a −1 kg/s surge disturbance
from outside, and his eq. 61 makes surge the *output of the level controller*.
A quantity the model calculates cannot also be injected and commanded.

**The evidence.** His §7 text and his own Fig. 11e disagree:

| case | what §7.1–7.4 **says** happens to `pp` | Fig. 11e **shows** | this model |
|---|---|---|---|
| A | "surge flow into the pressurizer **thereby increasing the pressure**" | flat | increases |
| B | "a **decrement in the pressurizer pressure**" | flat | decreases |
| C | "**leads to a rise** in the pressurizer pressure" | rises | rises |
| D | no direction given | flat | increases |

For all three cases where the paper states a direction, this model agrees with
the **text** and disagrees with the **figure**. Rendering Fig. 11e at 160 dpi
rules out a plotting artefact: the y-axis runs 15.40–15.48, so a 0.04 MPa move
would be half the panel height; A, B and D sit *exactly* on 15.41 for all 200 s
while C moves 0.0625.

Disabling eq. 32 and changing nothing else reproduces that exactly — three cases
flat to four decimal places, one still moving.

**The resolution.** `mdot_sur` (input 7) is the surge flow, and
`Config(pzr_surge=False)` is the default. This is the only reading under which
his equations, his §8 and his Fig. 11e are mutually consistent, and it scores
40/40 against 37/40 with eq. 32 added. `pzr_surge=True` additionally computes
expansion-driven surge: physically more complete, matches his §7.1–7.2 text, but
then a controller acting on `mdot_sur` is superimposed on a quantity the plant
also calculates.

**How much the surge path actually matters.** Worst case over the four cases it
moves `pp` by 0.04 MPa, normalised power by 3 parts in 10 000, and hot-leg
temperature by 0.015 °C — a reactivity worth of +0.62 / −0.19 / +0.01 / +0.17 pcm
against `beta` = 650 pcm. That is why all 36 non-`pp` comparisons are unaffected
by the switch. The surge *channel*, by contrast, is a strong actuator: +1 kg/s
held for 100 s moves `pp` by +0.32 MPa, eight times anything eq. 32 produces.
Weak as a feedback path, strong as an input — which is what makes it usable as
a control channel.

### 2.2 The default makes the pressurizer level loop mandatory

A consequence of 2.1, found four days after the default was flipped.

With `pzr_surge=False` and nothing commanding `mdot_sur`, **the pressurizer has
no mass path at all** — eq. 32 was the only thing moving water in or out. Gentle
transients are fine, but an MPC run at the 72 spm drive limit takes the level
state `lw` non-physical and the integration dies with `array must not contain
infs or NaNs`, raised by `scipy.linalg.lu_factor` with no mention of the
pressurizer anywhere in the traceback.

This is not an argument against the default; it is an argument that the level
loop is now mandatory rather than optional. Closing `PressurizerLevelPI` (whose
actuator is `mdot_sur`, [V] eq. 61) fixes it, which is what a real plant's CVCS
does. `Config(pressurizer=False)` and `pzr_surge=True` both also survive,
confirming the diagnosis.

### 2.3 A single pressurizer multiplier does fit

Sweeping `pzr_partial_scale` with the current defaults:

| scale | score | `pp` end, cases A / B / C / D |
|---|---|---|
| 1 (literal) | 31/40 | 15.410 / 15.410 / **58.817** / 15.410 |
| 1e2 | 31/40 | 15.410 / 15.410 / 15.847 / 15.410 |
| **700** (fitted) | **40/40** | 15.410 / 15.410 / **15.472** / 15.410 |
| 1e3 (shipped) | 40/40 | 15.410 / 15.410 / 15.454 / 15.410 |
| 1e4 | 40/40 | 15.410 / 15.410 / 15.414 / 15.410 |
| **paper** | | **15.410 / 15.410 / 15.473 / 15.410** |

With surge as an input, cases A, B and D sit at *exactly* [V]'s 15.410 at every
scale — the scale only ever affects the one case whose pressure moves. That
turns a two-sided fit into a one-parameter fit against a single number, and it
is well conditioned. `1e4` is **too large**, damping the heater response away.

Correcting the other suspect, `Vtheta[5]` from 2.776 to 0.2776, changes the
total surge coefficient by only 1.55× and moves the score by at most one
comparison. **The `Vtheta` OCR artefact is real but minor** — not the cause.

### 2.4 Eq. 34 is missing a `tau_hp`

[V] prints the HP turbine's first-derivative coefficient as
`(O_rv + tau_ip)/(tau_hp·tau_ip)`, but the IP and LP stages (eqs. 35, 36) —
identical in structure — both print `(O_rv·tau_hp + tau_ip)/(...)`.

| form | ζ | behaviour |
|---|---|---|
| eq. 34 as printed | 0.3162 | underdamped, 21 s ringing |
| with `tau_hp`, as eqs. 35/36 | 1.7322 | overdamped, smooth |

With `tau_hp` the quadratic factors **exactly** into
`(s + 1/tau_hp)(s + 1/tau_ip)` — residual 0.00e+00. The corrected equation is two
first-order lags in series, which is what an HP stage feeding an IP stage
physically is, and which cannot ring. The printed form's complex pole pair
corresponds to no mechanism in the plant.

Three further arguments: his Fig. 11f shows no ringing; a reversal count across
all 13 validated panels isolates the ringing to `P_tur` alone (case B shows 6
direction reversals where every thermal and pressure panel settles
monotonically), and the flag takes that 6 to 0 while touching nothing else.

**`HP_TAU = True` is the default.** `--no-hp-tau` restores the print. Not a
solver artefact: `rtol` 1e-7 → 1e-11, `atol` 1e-9 → 1e-14, and LSODA / BDF /
Radau agree to 3.3e-8.

### 2.5 Eq. 38 — the turbine speed loop was inert (our bug, not the paper's)

`TurbineSpeedPI` was implemented, wired correctly, and did nothing, because the
signal it reads could not move. Eq. 38's constants only close if its power terms
are in **watts**, and they were being fed per-unit:

| reading | implied H | dω/dt at 10 % mismatch |
|---|---|---|
| watts | 23.7 s | 0.127 Hz/s |
| per unit (as implemented) | — | 3.9e-11 Hz/s |

The ratio is exactly `Pth0 = 3.233e9`, identifying a units mismatch rather than
a modelling choice. **This was a bug in this implementation, not a defect in the
paper**, so the corrected form is the default; `Params(eq38_power_base=1.0)`
restores the old behaviour. Before: a 10 % load rejection moved speed 9.4e-8 Hz
in 50 minutes. After: an ungoverned rejection reaches 108.6 Hz and
`TurbineSpeedPI` holds the peak to 60.13 Hz.

**A trap this created.** `GridDemand + TurbineSpeedPI + HoldPower` through a
load-follow tracked demand to 0.0024 and looked correct. It was not the turbine
doing it — the rods pulled reactor power down, steam pressure fell, and since
`mdot_so = Ctg·cv·ps` the steam flow fell with the valve frozen. The manoeuvre
silently became reactor-leading, and the only visible tell was `utg` never
moving.

**Eq. 62's gains do not transfer either**, for an unrelated reason. `KP = 643`
implies 0.16 % governor droop where real machines run 4–5 %; at that gain a
0.13 Hz error commands 80 units of valve travel against a range of 0–1.5, the
valve slams shut and the set undershoots to 57.7 Hz. Shipped defaults are
droop-based, `KP = 0.333` per Hz.

### 2.6 Panel (h) — three channels match, one is not comparable

| channel | [V] Fig. 11h | this model |
|---|---|---|
| case B, `utg` | 5 % | **5.00 %** |
| case C, `iheat` | 50 % | **50.00 %** |
| case D, `T_fw` | 5 % ramp | **5.00 % ramp** |
| case A, rods | ~5.5 % | **80 pcm** (different quantity) |

Case A is not a mismatch. Rods here are a **velocity** input, so "% variation in
actuator output" has no counterpart. What is comparable is the reactivity
injected, and [V]'s own panel (g) pins that at 0.8 ΔmK/K = **80 pcm**, which this
model reproduces at 79.98 pcm — independently confirming the step size chosen for
case A. The panel is labelled to say so.

The heater channel previously read 65 % because it was normalised by its ~0 trim
value rather than the 1.3 mA installed capacity. That was a genuine plotting bug
and is fixed.

### 2.7 `rod_map` — the default was the transpose of [L] Case 2

**Resolved 2026-10-11.** The default is now [L]'s Case 2, `[[1,0],[0,1]]`:
Mrod drives the bottom half, AOrod the top.

[L] §6 fixes the geometry: core 366 cm, point 1 / point 2 boundary at 183 cm, AO
band [236, 316] cm — inside the top half, so AOrod sits in point 2 in *both* his
cases. A rod model in differentials tracks where the rod **tip** is, so the cases
differ in where Mrod *starts*:

| configuration | `rod_map` | Mrod `dPn / dDI` | AOrod `dPn / dDI` |
|---|---|---|---|
| **[L] Case 2** (MIMO, eq. 12) — the default | `[[1,0],[0,1]]` | +2.22e-3 / −2.21e-2 | +5.07e-3 / +1.92e-2 |
| **[L] Case 1** (SISO, eq. 11) | `[[0,0],[1,1]]` | +5.07e-3 / +1.92e-2 | identical |
| former default (withdrawn) | `[[0,1],[1,0]]` | +5.07e-3 / +1.92e-2 | +2.22e-3 / −2.21e-2 |

**Case 1 is not Case 2 with AOrod frozen.** Freezing AOrod in Case 2 leaves Mrod
on the bottom half (`dDI = −0.082` for a 20-step move); Case 1 has it on the top
(`dDI = +0.072`). Opposite halves, opposite sign — a different matrix, not a
different command. Verified numerically, not assumed.

**What the correction did and did not change.** The two maps are the *same
plant*: `A` identical, the two columns of `B` merely exchanged, so no previously
published number was wrong — the score is 40/40 before and after. What was wrong
was the labelling, which matters the moment a controller is written against
`vrod_M` or `vrod_AO` by name. The differential pairing in `AxialOffsetBand` and
`SlidingAO` was flipped to match (`vrod_AO += vd`, `vrod_M -= vd`); under the old
signs with the corrected map those loops would have driven `DI` away from target.

### 2.8 Retractions

Three explanations in earlier revisions of this file were constructed after the
fact and are **withdrawn**. They are listed because a retraction that is not
recorded is indistinguishable from never having made the claim.

| withdrawn claim | what is true |
|---|---|
| The old `rod_map` "was chosen so that Mrod has the larger power authority, which is what 'power regulating rod' implies." | **Origin unknown.** It is the transpose of [L] eq. 12; the probable cause is a transposition error when eqs. 11–12 were entered. No record of a choice exists. |
| `w_tur` is "uncontrollable" and `Pdem` is "a disturbance, not a control input". | Both were consequences of the eq. 38 units bug (2.5). Neither is a structural property. |
| "The flat cases want a large scale and the heater case wants a small one", so no single `pzr_partial_scale` fits. | An artefact of measuring with the surge path on. With surge as an input a single multiplier fits well (2.3). |

A fourth, smaller correction: panel (h)'s case A normalisation was described as
"a deliberate choice". It was a mislabelled axis, now fixed (2.6).

Writing a plausible rationale for an unexplained value, rather than recording
that its origin is unknown, converts a defect into an apparent design decision
and prevents it from ever being found. The `implementation-provenance` skill was
written in response to this.

---

## 3. Open items

### 3.1 `pzr_partial_scale` — origin unknown, largest open uncertainty

Table 1 prints `K1p`–`K4p` with unit labels that do not parse (`K3p` as
J/m³·MPa; `K2p` and `K4p` identical). Literally they give ≈3.2 MPa per kg/s of
surge — the pressurizer runs to −57 MPa in 20 s.

**Origin of the shipped `1e3`: unknown.** It is a round decade and the obvious
first guess at a kPa/MPa slip, but no record of how it was picked exists, and
`700` was only measured later. It is *kept* because the open-loop score is
identical (40/40 either way), so changing the default would churn every published
number for no measured gain — a decision taken now, not the reason it was
written. Either value is a reconstruction of a table that does not parse. **A
legible copy of [V] Table 1 is what would settle it.**

### 3.2 A ~0.4 % speed difference in the SG secondary

[V]'s Fig. 11f shows case B overshooting to ~1.0315 at t ≈ 45 s before settling
to ~1.0285. Neither form of eq. 34 reproduces that; this model rises
monotonically to 1.0287. The cause is not the turbine but the steam flow feeding
it: `mdot_so` peaks at 1.0448 within about a second and has decayed to 1.0282 by
t = 45, and the turbine's dominant lag is `tau_hp = 10 s`, so it never sees the
peak. For [V] to show a hump at t ≈ 45, his `ps` must still be near 0.9835 of
nominal there where this model is at 0.97994 — a difference of **0.36 %**, which
multiplied into the steam flow is exactly the ~0.003 overshoot in question.

It is **not** `Ks`: the SG pressure node has a 14.7 ms time constant, so `ps` is
slaved to the metal-tube temperatures, and scaling `Ks` by 8× moves the settling
time only from 2.4 s to 3.0 s. The difference lives in the SG metal/primary
coupling constants (`tau_ms`, `tau_mp`, `tau_p`) and **has not been isolated
further. Bounded at ~0.4 % in `ps` and ~0.3 % in turbine output.**

### 3.3 [V] never names a solver

*(checked 2026-10-12, full text search.)* The paper says only "MATLAB/Simulink
environment", twice. No integrator, no tolerances, no fixed or maximum step size,
no output sample rate. The only times quoted anywhere are event times.

That is a reproducibility gap in a paper whose stated contribution is that the
model is "readily implementable", and it matters more here than for most:
Simulink's `VariableStepAuto` picks between an explicit and a stiff method by its
own heuristic, and his eq. 45 log amplifier carries `tau_1 = 5e-8 s` against
thermal modes near 1e-3 /s — a stiffness ratio above 1e10, where that choice is
not cosmetic. **Which solver he used is unknown and cannot be recovered from the
paper; do not infer one from Simulink's defaults.**

It does not affect anything in this file. Every conclusion here was tested across
three methods and four decades of tolerance.

### 3.4 [L]'s Tables 2 and 3 have never been compared against

His 13th-order numerator / 14th-order denominator SISO and MIMO transfer
functions are **the sharpest available check on the two-point core**, and they
are not done. Only meaningful after 5.6 below, since they belong to a different
plant.

### 3.5 `F_couple = 5.6` 1/s is the one genuinely free parameter

[L] gives `F = 4·d·v_s/h²` symbolically and never prints `d`, `v_s` or `F`.
Calibrated so the axial xenon oscillation period lands in the 20–30 h band.
Any `DI` result depends on it, and on `rho_xe_full` (3.6). **Run a sensitivity
study before quoting one.**

### 3.6 Xenon worth is from literature, not either paper

[L]'s Table 1 *names* `alpha_X1` but never prints a value. Used −0.028 Δk/k
(2800 pcm), the standard PWR figure.

---

## 4. Deviations from [V] as printed

Each is a place where the paper as printed is self-inconsistent or dimensionally
wrong, and the model makes a documented choice.

| # | What | Why | What the model does |
|---|---|---|---|
| 1 | **Rod drive gain, eq. 50** | `G` in cents/step, `v_rod` in steps per **minute**, while every other derivative is per **second**. Taken literally, full rod speed is prompt critical in under a second. | Divided by 60 → 1.16 cents/s at the 72 spm limit, consistent with the ~12 cents in his Fig. 12b. **A residual ~5 orders of magnitude is unrecoverable from the text**, so his Table 2 rod gains do not transfer. |
| 2 | **Pressurizer partials K1p–K4p** | Unit labels do not parse. | `pzr_partial_scale = 1e3`. See 3.1 — still the largest open uncertainty, though it no longer costs any validation comparisons. |
| 3 | **`Vtheta[5] = 2.776`** | Every other entry is O(0.1–1.3); this one is 10× larger and the next is 0.2776. | Left as printed, flagged as a likely OCR artefact. Correcting it moves the score by ≤1 (2.3). |
| 4 | **`Ctg = 2.0481`** | Does not close the SG heat balance at the tabulated initial conditions. | Derived from the heat balance; see `consistency_report()`. |
| 5 | **`lw0 = 28.06` m** | Exceeds the pressurizer length `l = 14.2524` m. Physically impossible. | Used 8.5527 m. |
| 6 | **`h_cow`, `h_lp`** | Not enthalpies — off by ~4 and ~3 decades. | Scaled. Condenser is off by default anyway. |
| 7 | **Level loop, eq. 61** | Manipulates a quantity eq. 32 computes. | Resolved by making surge an input (2.1). |
| 8 | **`tau_1 = 5e-8` s log amplifier** | Adds a −2e7 /s eigenvalue to a system whose slowest mode is −2.1e-5 /s. | Off by default (`fast_log_amp=False`). |
| 9 | **`F_ip = 0`** | Table 1 zeroes the IP turbine fraction, making its 2 states inert. | Excluded by default (`include_ip=False`). |
| 10 | **Condenser, eqs. 39–42** | Implemented, but a pure sink — nothing feeds back into the plant. | Off by default (`condenser=False`). |

### Which of these are load-bearing

`Params.as_printed()` restores all six flagged values; flipping **one at a time**
isolates each:

```
ABLATION -- shipped defaults with ONE value restored to the paper
  (shipped)        40/40
  pzr partials     31/40   <-- -9
  lw0 = 28.06      39/40   <-- -1
  condenser h      40/40
  Ctg = 2.0481     40/40
  eq 50 literal    40/40
```

**Five of the six are free.** The condenser enthalpies, the tabulated `Ctg` and
eq. 50's literal reading change the score by nothing at all. `lw0 = 28.06` costs
a point, and the geometry objection to it stands independently.

**One carries everything: the pressurizer partials, at −9.** As printed they send
Case C's end-state pressure to 9.09 MPa against [V]'s 15.473, or to 58.8 MPa with
the shipped `lw0` — about 4× operating pressure. The model still integrates; it
just produces a physically meaningless number in that one variable.

---

## 5. Deviations from [L]

### 5.1 Implemented faithfully

- eqs. 1–6: two-point kinetics, 6 delayed groups per point, iodine/xenon per
  point, Mann thermal stack per point, feedback on fuel, coolant and xenon.
- eqs. 7–10: diffusion coupling `Pr12`, `Pr21` with the `Pf20/Pf10` ratios kept
  explicit.
- eqs. 11–12: rod model, configurable to both of [L]'s cases (2.7).
- eqs. 13–14: `Pr = (Pr1+Pr2)/2`, `DI = (Pr2−Pr1)/2`.
- `T2Cin = TC12`: the halves are in series, bottom → top.
- No boron, and his reason is stated in his introduction: controlling by boron
  *"must produce lots of liquid waste and increase the control difficulty"*, so
  his plants are built *"of ignoring boron concentration changes and selecting
  the power regulating rod and AO rod as the input mechanism."*

### 5.2 The rod model is a velocity model, not [L]'s position model

[L] eqs. 11–12 are `drho_rod = a_Mrod · dMrod` with `dMrod` in fractions of core
length — a position-to-reactivity map. The model uses [V] eq. 50,
`drho_rod/dt = G · v_rod` — a rate. Consequences:

- **There is no rod position state.** [L]'s Figs. 9b and 11b (rod position
  trajectories in cm) cannot be plotted, and the AO band [236, 316] cm cannot be
  enforced as a hard constraint.
- [L] never prints numerical values for `a_Mrod` or `a_AOrod` anyway, so his
  model could not have been implemented as stated.

### 5.3 Coolant reactivity convention follows [V], not [L]

[L] eq. 3 puts `alpha_C/2` on each coolant node; [V] eq. 44 puts the full
`alpha_c` on each. Default follows [V] so the model reduces to his; flip
`half_coolant_coeff` for [L]'s convention.

### 5.4 `Pf10 ≠ Pf20` is not exercised

`r12 = r21 = 1.0` is hard-coded (flagged as a one-line change). An asymmetric
nominal axial shape is not available.

### 5.5 `AO` itself is not exposed

`signals()` returns `DI`; [L] eq. 13's `AO = DI/Pn` is not a named signal.
Trivial to add.

### 5.6 No parameter in the core is [L]'s

**[L]'s Table 1 contains no numbers at all** — it is a symbol glossary. Every
thermal and kinetic parameter comes from [V]. [L] cites IAEA (1971) for a
2200 MW plant at 285/317 °C, 15.5 MPa, 12861.1 kg/s; [V]'s plant is a different
one. **So no parameter set in this model is [L]'s**, and his Tables 2/3 transfer
functions were computed for a plant this model does not represent.

### 5.7 Linearization differences

- Uses all 6 delayed groups; [L] collapses to one group. More faithful to the
  nonlinear plant, *less* comparable to his Tables 2/3.
- Coolant inlet temperature is a state, not a constant. [L] holds it constant
  because his core is isolated; here the plant is attached, which is the point of
  the exercise — but it makes the linear model structurally different from his.

### 5.8 [L]'s control design is not implemented

- **LTR.** There is an `LQG` controller ([V] eqs. 63–70). There is **no loop
  transfer recovery**: no recovery gain sweep, no `Gamma = [CH CL]ᵀ`
  construction ([L] §3.3), no Nyquist/Bode or singular-value comparison.
- **Integrator augmentation** ([L] eq. 20, Fig. 2). [L] augments with `1/s`
  *before* designing, which is what gives his loop zero steady-state error to a
  step. The shipped `LQG` has no augmentation, which is why it has no such
  guarantee.
- **The stabilising shift in `LQG` is a workaround, not a design.** With a single
  actuator several integrators are uncontrollable, so `(A,B)` is unstabilisable
  and the ARE has no finite solution; the code designs against `A − shift·I`.
  [L] avoids this because his core is isolated and his augmented pair is
  genuinely controllable.
- **Stability analysis** — Theorems 1 and 2, the LMI solve for `N` (his Tables 4,
  5), Criterion 1's controllability determinant, the inverse Nyquist array with
  Gershgorin bands. None implemented.
- **His scenarios** — the 24 h ramp load-follow at ±0.0556 %/min with the DI band
  (his Figs. 9, 11) and the Case 2 DI target trajectory. Not run.

---

## 6. Constraints on use

**This is the section to read before quoting any result from this model.**

### 6.1 Actuators that do not exist

- **Feedwater flow / SG level control.** `T_fw` is an input; `mdot_fw` is not —
  it is implicit in the SG energy balance. There is no SG downcomer level state,
  so no three-element feedwater controller is possible.
- **Boron / CVCS boration–dilution.** No boron concentration state. ([L] excludes
  boron deliberately, so this is a shared, intentional omission.)
- **Steam dump / turbine bypass.** No path for steam around the turbine, so the
  model cannot do a load rejection or turbine trip without a reactor trip.
- **Reactor coolant pump speed.** Primary flow is constant; no RCP coastdown.

### 6.2 Physics that does not exist

- **Decay heat.** Power follows kinetics to zero. Any shutdown or large downpower
  transient is wrong in the tail.
- **Peaking factors, DNBR, linear heat rate.** No thermal limits of any kind.
- **Pressure drop and pump head.** No primary hydraulics.
- **SG shrink/swell, U-tube level.** The secondary is a 2-node lump.
- **Spatial anything** in [V] — which is exactly why [L]'s core is grafted in.

### 6.3 Protection and limits that do not exist

- No reactor trip logic, no overpower-ΔT or overtemperature-ΔT setpoints.
- No rod insertion limits, no withdrawal interlocks, no rod position state at
  all (5.2).
- No actuator rate limits beyond the per-controller `v_max` clips.

### 6.4 Numerical and operational constraints

- **The axial mode is unstable.** `DI` runs away in either direction, reaching
  |DI| ≈ 1 within about six hours of an uncorrected tilt, and never settles. Any
  run longer than a few hours needs an axial offset controller.
- **The pressurizer level loop is mandatory** under aggressive control (2.2).
- **Linear validity** is about a 12 % step for 5 % error in `Pn`. Beyond that,
  schedule.
- **`cond(A) ≈ 1e25`.** If you integrate the linear model yourself, pass the
  Jacobian explicitly; an implicit solver finite-differencing it produces
  non-finite values.
- **The `rod_map` is constant**, so it encodes one fixed assumption about where
  both rod tips sit. Nothing detects a tip crossing the midplane. Fine for
  excursions of tens of steps out of a ~231-step bank; not for a full withdrawal.

### 6.5 The bottom line

The model is honest for **load-following within the normal band** — power
manoeuvres, temperature programmes, axial offset control. It is **not** valid for
accident, trip or protection studies, and **it cannot be used to argue that a
controller respects a safety limit, because it contains no safety limits.**

---

## 7. Summary

| | status |
|---|---|
| [V] plant, eqs. 1–52 | implemented; 10 documented deviations, all from defects in the printed paper |
| [V] control, eqs. 53–70 | implemented; **rod-loop and governor gains in his Table 2 do not transfer** (eq. 50 unit error; 0.16 % droop) |
| [V] open-loop validation, Figs. 10–11 | **40/40** shipped; 37/40 `--surge`; 30/40 `--printed` |
| [L] core model, eqs. 1–14 | implemented; **no numerical parameter in it is [L]'s** |
| [L] rod cases 1 and 2 | default is Case 2; Case 1 via `Config(rod_map=[[0,0],[1,1]])`. Case 1 is *not* Case 2 with AOrod frozen (2.7) |
| [L] LQG/**LTR** design | **not implemented** — LQG only, no recovery, no augmentation |
| [L] Tables 2/3 transfer functions | **not compared** — the sharpest unused validation available |
| [L] stability theorems, INA | not implemented |
| [L] 24 h load-follow scenarios | not run |

**Biggest single open item:** `pzr_partial_scale`. A reconstruction of a table
that does not parse, origin of the shipped value unknown. A legible copy of [V]
Table 1 would settle it.

**Biggest unused opportunity:** [L]'s Tables 2 and 3. Reproducing those transfer
functions from the linearized two-point core — after rebuilding it on his IAEA
(1971) parameters — is the one test that would validate the core itself rather
than the plant around it.
