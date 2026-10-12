# API reference

`pwr_two_point_plant_v2` — a non-linear pressurised water reactor plant model
with a control testbed.

```python
from pwr_two_point_plant_v2 import *
```

Everything below is importable from the top level. New users should read
`QUICKSTART.md` first; this page is the complete reference.

---

## Contents

**[Building a plant](#building-a-plant)**
[`Plant`](#plant) · [`Params`](#params) · [`Config`](#config) ·
[`Config.pzr_surge`](#pzr_surge) ·
[`Config.rod_map`](#rod_map-and-lis-two-cases)

**[Finding equilibria](#finding-equilibria)**
[`trim`](#trim) · [`trim_sweep`](#trim_sweep)

**[Running simulations](#running-simulations)**
[`ClosedLoop`](#closedloop) · [`Result`](#result)

**[Writing controllers](#writing-controllers)**
[`Controller`](#controller) · [`LoopContext`](#loopcontext)

**[Built-in controllers](#built-in-controllers)**
[`Manual`](#manual) · [`GridDemand`](#griddemand) · [`ValveOnDemand`](#valveondemand) ·
[`RodsOnPower`](#rodsonpower) · [`RodsOnTavg`](#rodsontavg) ·
[`AxialOffsetBand`](#axialoffsetband) · [`SGPressurePI`](#sgpressurepi) ·
[`PressurizerHeaterPI`](#pressurizerheaterpi) · [`DiscretePI`](#discretepi) ·
[`SlidingAO`](#slidingao)

**[Vajpayee's control loops](#vajpayees-control-loops-paper-section-5)**
[`CoreNeutronicsPI`](#coreneutronicspi) · [`TemperaturePI`](#temperaturepi) ·
[`CombinedPowerPI`](#combinedpowerpi) · [`PressurizerSprayPI`](#pressurizerspraypi) ·
[`PressurizerLevelPI`](#pressurizerlevelpi) · [`TurbineSpeedPI`](#turbinespeedpi) ·
[`Params.eq38_power_base`](#paramseq38_power_base) ·
[`LQG`](#lqg) · [`VAJPAYEE_TABLE2`](#vajpayee_table2)

**[Reproducing the paper](#reproducing-the-paper)**
[`Params.as_printed`](#paramsas_printed) · [`HP_TAU`](#hp_tau)

**[Load profiles](#load-profiles)**
[`step`](#step) · [`ramp`](#ramp) · [`tavg_program`](#tavg_program)

**[Preset configurations](#preset-configurations)**
[`turbine_leading`](#turbine_leading) · [`reactor_leading`](#reactor_leading) ·
[`open_loop`](#open_loop)

**[Linear analysis](#linear-analysis)**
[`linearize`](#linearize) · [`dc_gain`](#dc_gain) · [`freq_response`](#freq_response) ·
[`typical_scales`](#typical_scales)

**[Name tables](#name-tables)** · **[Low-level](#low-level)** ·
**[Troubleshooting](#troubleshooting)**

---

# Building a plant

## `Plant`

```python
Plant(p=None, cfg=None)
```

The non-linear plant. Holds the parameter set, the state layout, and the
derivative function.

**Parameters**

| | |
|---|---|
| `p` : [`Params`](#params), optional | Physical constants. Default `Params()`. |
| `cfg` : [`Config`](#config), optional | Which subsystems are present. Default `Config()`. |

**Attributes**

| | |
|---|---|
| `n` : int | Number of states (53 with default `Config`). |
| `m` : int | Number of inputs (always 7). |
| `names` : list of str | State names in vector order. |
| `ix` : dict | `name -> index` into the state vector. |
| `x0` : ndarray, shape (n,) | Nominal state. **Not** a converged equilibrium — use [`trim`](#trim). |
| `u0` : ndarray, shape (7,) | Nominal input. Also not trimmed. |
| `p`, `cfg` | The objects passed in. |

### `Plant.deriv`

```python
deriv(t, x, u) -> ndarray
```

State derivative. The whole model is this one function; everything else is
bookkeeping around it.

**Parameters** — `t` : float, seconds (the model is autonomous; `t` is accepted
for solver compatibility). `x` : ndarray, shape (n,). `u` : ndarray, shape (7,).

**Returns** — ndarray, shape (n,), the time derivative of `x`.

### `Plant.signals`

```python
signals(x, u=None) -> dict
```

Human-readable quantities derived from the state. **A function of the state
alone** — `u` is accepted for backward compatibility and ignored. That is what
lets a controller read signals without creating an algebraic loop.

**Returns** — dict with these keys:

| key | meaning | at 100 % |
|---|---|---|
| `Pn` | normalised core power, 1.0 = full | 1.0000 |
| `DI` | axial power difference, `(top − bottom)/2` | 0.0000 |
| `Pr1`, `Pr2` | normalised power, bottom and top half | 1.0000 |
| `Tavg` | mean of the four coolant node temperatures, °C | 315.9221 |
| `Thot`, `Tcold` | hot-leg and cold-leg temperature, °C | 327.2997 / 296.9595 |
| `Ts` | secondary saturation temperature, °C | 288.0600 |
| `ps` | steam pressure, MPa | 7.2800 |
| `pp` | pressuriser pressure, MPa | 15.4100 |
| `lw` | pressuriser water level, m | 8.5527 |
| `mdot_so` | steam flow, kg/s | 2164.1 |
| `P_tur` | turbine mechanical power, per unit | 1.0000 |
| `P_th` | core thermal power, W | 3.233e9 |
| `w_tur` | turbine speed, Hz | 60.0000 |
| `ilo` | ex-core log amplifier current, mA | 19.6554 |
| `irtd` | RTD current, mA | 14.6669 |

> **Note.** `Tavg` is the mean of four well-mixed lumps, which is **not** the
> plant convention `(Thot + Tcold)/2` — that is 312.13 °C at full power, about
> 3.8 °C lower. Use the model's `Tavg` for control (it is internally
> consistent); use `(Thot + Tcold)/2` when comparing against plant data.

### `Plant.simulate`

```python
simulate(x0, t_end, u_of_t, n_out=2000, method="BDF", rtol=1e-8, atol=1e-10)
```

Low-level integration with an arbitrary input policy `u_of_t(t, x) -> u`.
Returns a raw `scipy` solution object. Prefer [`ClosedLoop`](#closedloop)
unless you need this.

---

## `Params`

```python
Params(**overrides)
```

Every physical constant, as a dataclass. All SI: W, °C, MPa, kg, s, J.
Override any field by keyword:

```python
p = Params(G_cents_per_step=1.17, alpha_c=-2.2e-4)
plant = Plant(p)
```

**Selected fields** (there are ~90; see the source for the full table):

| field | default | meaning |
|---|---|---|
| `Lam` | 3e-5 | neutron generation time, s |
| `beta_i` | 6 values | delayed neutron fractions, sum 6.502e-3 |
| `lam` | 6 values | delayed group decay constants, 1/s |
| `alpha_f` | −2.16e-5 | fuel temperature coefficient, Δk/k per °C |
| `alpha_c` | −1.80e-4 | coolant temperature coefficient, Δk/k per °C |
| `alpha_p` | +1.5664e-4 | pressure coefficient, Δk/k per MPa |
| `G_cents_per_step` | 0.9679 | control rod worth, cents per step |
| `F_couple` | 5.6 | axial neutron coupling, 1/s **(see note)** |
| `rho_xe_full` | −0.028 | equilibrium xenon worth at 100 %, Δk/k **(see note)** |
| `tau_f`, `tau_c`, `tau_r` | 4.376, 7.166, 0.674 | core thermal time constants, s |
| `ps0`, `pp0`, `Ts0` | 7.28, 15.41, 288.06 | nominal steam/pressuriser pressure, sat. temp |
| `Tf0`, `Tc10`, `Tc20` | 626.66, 312.13, 327.30 | nominal fuel and coolant temperatures, °C |

**Derived at construction** — `beta` (6.502e-3), `tau_r_half`, `Pth0`
(3.233 GW), `Ctg0`, `G_dkk_per_step`, `eq38_power_base`.

**Rod worth chain.** `G_cents_per_step` is in cents of ρ, i.e. hundredths of a
dollar, where one dollar is β. `G_dkk_per_step` converts it to the Δk/k the
reactivity equation actually wants:

```
G_dkk_per_step = G_cents_per_step / 100 × beta
               = 0.9679 / 100 × 6.502e-3
               = 6.2933e-5   Δk/k per step
```

A Westinghouse bank travels 231 steps (228 plus overlap, from the R-104P manual
Vajpayee cites in his section 5.1.3), so the integral worth of one bank is
`6.2933e-5 × 231 = 0.01454` Δk/k — $2.24, or 1454 pcm. That 231 is what
`validate_vajpayee.py` uses as `ROD_FULL_TRAVEL` to turn rod reactivity into the
percent-of-travel axis of Vajpayee's Fig. 11h.

> **Two parameters are not from either source paper** and both materially
> affect axial behaviour: `F_couple` (derived from `4dν/h²`) and
> `rho_xe_full` (standard PWR figure). Run a sensitivity study before quoting
> any `DI` result. Six further values are flagged transcription corrections —
> run `consistency_report(plant)` for the audit.

---

## `Config`

```python
Config(two_point=True, xenon=True, pressurizer=True, sensors=True,
       condenser=False, include_ip=False, fast_log_amp=False,
       pzr_surge=False, rod_map=...)
```

Which subsystems are present. Changing these changes `plant.n` and the state
ordering, which is why `plant.ix` is per-instance.

| field | default | effect |
|---|---|---|
| `two_point` | `True` | `False` forces both axial halves identical (collapses to a single-point core). |
| `xenon` | `True` | `False` removes iodine/xenon feedback. `DI` then returns to exactly zero after a tilt. |
| `pressurizer` | `True` | `False` drops the `pp` and `lw` states and holds primary pressure constant. |
| `sensors` | `True` | Ex-core amplifiers and RTDs. |
| `condenser` | `False` | Passive appendage; off by default. |
| `include_ip` | `False` | IP turbine states. `F_ip = 0`, so they carry no power. |
| `fast_log_amp` | `False` | Keeps `tau_1 = 5e-8 s`. Extremely stiff; rarely worth it. |
| `pzr_surge` | `False` | `True` *adds* Vajpayee eq 32 — surge computed from coolant expansion — on top of the `mdot_sur` input. See below. |
| `rod_map` | `[[1,0],[0,1]]` | Rows are halves, columns are `(Mrod, AOrod)`. Default is Li's Case 2: Mrod drives the **bottom** half (half 1), AOrod the **top** (half 2). |

### `pzr_surge`

Surge flow is an **input** (`mdot_sur`, index 7), not a computed quantity, and
`pzr_surge=False` is the standard configuration.

Vajpayee gives surge an equation (eq 32) but also injects it as a disturbance in
his section 8.3.1 and *commands* it from his level controller (eq 61). A quantity
eq 32 calculates cannot also be injected and commanded. Treating it as an input
is the only reading under which his equations, his section 8 and his Fig. 11e are
mutually consistent — and it scores 40/40 against his published figures, against
37/40 with `pzr_surge=True`.

`pzr_surge=True` additionally computes expansion-driven surge. Physically more
complete, and it matches his section 7.1–7.2 text, but a controller acting on
`mdot_sur` is then superimposed on a quantity the plant also calculates.

The effect is confined to the pressurizer: worst case it moves `pp` by 0.04 MPa,
normalised power by 3 parts in 10 000, and hot-leg temperature by 0.015 °C.

### `rod_map` and Li's two cases

`rod_map @ vrod` distributes rod reactivity over the two halves. Rows are halves
(row 0 = half 1 = bottom), columns are the two rod groups `(Mrod, AOrod)`.

```python
Config(rod_map=np.array([[1., 0.], [0., 1.]]))   # Li Case 2 (MIMO) — THE DEFAULT
Config(rod_map=np.array([[0., 0.], [1., 1.]]))   # Li Case 1 (SISO)
```

Li's section 6 fixes the geometry: core height 366 cm, the point-1/point-2
boundary at 183 cm, AO band 236–316 cm. A rod model written in **differentials**
tracks where the rod **tip** is, not where its body sits, which is what makes
each case a different matrix:

- **Case 2 (MIMO, the default).** Mrod starts at 183 cm — tip at the midplane —
  so moving it sweeps the **bottom** half. AOrod sits in the **top**. Two
  independent inputs: full rank, so `Pn` and `DI` are separately controllable.
- **Case 1 (SISO).** Mrod starts at 366 cm, fully withdrawn, so it sweeps the
  **top** half, and AOrod is held motionless. Both columns are then identical,
  the matrix is rank-1, and the core is single-input — which is exactly Li's
  statement about Case 1.

**Case 1 is not Case 2 with AOrod frozen.** This is the easy mistake, and it is
wrong in sign. Freezing AOrod in Case 2 leaves Mrod on the **bottom** half
(`dDI = −0.082` for a 20-step move); Case 1 has it on the **top**
(`dDI = +0.072`). Opposite halves, opposite sign. The two cases differ in where
Mrod *starts*, which is a different matrix — not a different command. To run
Case 1, pass its `rod_map`; do not simply stop commanding `vrod_AO`.

**Provenance.** Through 2026-10-11 the shipped default was `[[0,1],[1,0]]` — the
transpose of Case 2, with the rod labels exchanged. The origin of that matrix is
unknown and it was most likely a transposition error when Li's equations 11 and
12 were entered. An earlier revision of this document justified it as a choice
made "so that Mrod has the larger power authority"; that explanation was
constructed after the fact and has been retracted. The transpose is the *same
plant* — identical `A`, with the two columns of `B` merely exchanged — so no
result produced under it was numerically wrong. Only the labels were, which
matters the moment a controller is written against them.

**What the map does not know.** It is constant, so it encodes one fixed
assumption about where both tips sit. It is a linearisation about a nominal
insertion and is wrong if a tip crosses the midplane — and nothing detects that,
because there is no rod-position state (see [the inputs](#the-inputs); rods are
velocity inputs). Adequate for load-following excursions of tens of steps out of
a ~231-step bank; not for a full withdrawal.

Measured single-rod authority at 100 % power, +5 steps/min held for 280 s:

| input | `ΔPn` | `ΔDI` |
|---|---|---|
| `vrod_M` (bottom) | +0.0124 | **−0.0917** |
| `vrod_AO` (top) | +0.0248 | **+0.0793** |

Same sign on power, opposite sign on offset. This is why the differential
pairing in `AxialOffsetBand` and `SlidingAO` is `vrod_AO += vd`,
`vrod_M -= vd`.

---

# Finding equilibria

## `trim`

```python
trim(plant, power, di_target=0.0, ps_target=None, guess=None, verbose=False)
    -> (x0, u0, residual)
```

Solve for a steady state at the requested power: a state where every derivative
is zero.

**Parameters**

| | |
|---|---|
| `plant` : [`Plant`](#plant) | |
| `power` : float | Target normalised power. Verified 0.2 to 1.0. |
| `di_target` : float, default 0.0 | Target axial offset. |
| `ps_target` : float, optional | Steam pressure setpoint. Default `plant.p.ps0`. |
| `guess` : ndarray, optional | Initial guess for the 19 unknowns, from a previous solve. See [`trim_sweep`](#trim_sweep). |
| `verbose` : bool, default False | Print the residual and converged inputs. |

**Returns**

| | |
|---|---|
| `x0` : ndarray, shape (n,) | Equilibrium state. |
| `u0` : ndarray, shape (7,) | Matching inputs. |
| `residual` : float | ‖dx/dt‖ at the solution. |

**Notes**

Expect `residual ≈ 1e-13`. **Above about 1e-10 the result is not an
equilibrium** — do not proceed, because any later drift will be
indistinguishable from your controller's effect.

Precursor, iodine and xenon states are eliminated analytically before the
solve, reducing the numerical problem from 53 unknowns to 19. Three
integrators are handled separately: rod reactivity is treated as a free
algebraic variable, turbine speed by imposing `Pdem = P_tur`, and pressuriser
level is left to settle.

**Examples**

```python
x0, u0, res = trim(plant, 1.0)
print(f"residual {res:.2e}, Tavg {plant.signals(x0)['Tavg']:.2f} °C")
```

## `trim_sweep`

```python
trim_sweep(plant, powers, di_target=0.0, verbose=True)
    -> list of (power, x0, u0, residual)
```

Trim at a sequence of power levels using continuation — each solve is seeded
with the previous answer.

**Always use this for a family of operating points.** A cold solve far from
100 % may not converge.

```python
family = trim_sweep(plant, np.arange(1.0, 0.19, -0.05))
for pw, x, u, r in family:
    print(pw, plant.signals(x)["Tavg"])
```

---

# Running simulations

## `ClosedLoop`

```python
ClosedLoop(plant, controllers, demand=None, power=1.0, di_target=0.0,
           x0=None, u0=None, measurement_noise=None)
```

Plant plus a list of controllers, integrated together as one augmented system.

**Parameters**

| | |
|---|---|
| `plant` : [`Plant`](#plant) | |
| `controllers` : sequence of [`Controller`](#controller) | Run in list order. An empty list leaves every actuator at its trim value. |
| `demand` : callable, optional | `demand(t) -> float`, the load profile. Default: constant at `power`. See [load profiles](#load-profiles). |
| `power` : float, default 1.0 | Power to trim at, if `x0`/`u0` are not given. |
| `di_target` : float, default 0.0 | Axial offset to trim at. |
| `x0`, `u0` : ndarray, optional | A trim point from [`trim`](#trim). Supplying both **skips the solve**, which saves ~0.6 s per loop — do this when sweeping controller gains. |
| `measurement_noise` : callable, optional | `f(t, sig_dict) -> sig_dict`, applied **only at sample instants** of discrete controllers. Injecting noise into a continuous controller would corrupt the solver's Jacobian. |

**Attributes**

| | |
|---|---|
| `x0`, `u0` : ndarray | The trim point actually used. Note `u0[IU["utg"]] = 0.99963`, not 1.0. |
| `trim_residual` : float | |
| `n` : int | Total states: plant plus all controller states. |
| `ctx` : [`LoopContext`](#loopcontext) | Passed to every controller. |
| `controllers` : list | |
| `discrete`, `continuous` : list of int | Indices, split by whether `Ts` is set. |

### `ClosedLoop.run`

```python
run(t_end, n_out=1500, method=None, rtol=1e-7, atol=1e-9, x0=None) -> Result
```

Integrate from 0 to `t_end` seconds.

| | |
|---|---|
| `t_end` : float | Simulated duration, seconds. |
| `n_out` : int, default 1500 | Number of output samples. |
| `method` : str, optional | `"LSODA"` (default, continuous) or `"BDF"` (default, sampled). `"Radau"` also available. |
| `rtol`, `atol` : float | Solver tolerances. |
| `x0` : ndarray, optional | Override the initial state (inputs still come from the trim). |

**Notes**

If any controller sets `Ts`, the *sampled path* is used: the run is chopped at
every sample instant, discrete controllers fire once each, and their outputs
are held constant between. This costs roughly 80 ms per restart, so it scales
with `t_end / Ts` and is about an order of magnitude slower than the continuous
path. Sample only the loop that needs it.

The model is stiff by roughly seven orders of magnitude (prompt neutrons at
−217 /s, xenon at −2.1e−5 /s). Explicit solvers are not an option.

### `ClosedLoop.inputs`

```python
inputs(t, x, xc_all) -> (u, dxc)
```

Evaluate the control law without integrating. Useful for checking a controller
returns the trim inputs at the trim point.

### `ClosedLoop.sample`

```python
sample(t, x, xc_all, which=None) -> None
```

Fire the sampled controllers once, in place. Called automatically by `run`.

---

## `Result`

Returned by [`ClosedLoop.run`](#closedlooprun).

**Attributes**

| | |
|---|---|
| `df` : `pandas.DataFrame` | One row per output sample. ~73 named columns — see below. |
| `t` : ndarray | Time, seconds. |
| `X` : ndarray, shape (n, N) | Plant states. |
| `XC` : ndarray | Controller states. |
| `U` : ndarray, shape (7, N) | Inputs actually applied. |
| `loop` : [`ClosedLoop`](#closedloop) | |

**Column naming**

| prefix | contents | example |
|---|---|---|
| — | time and demand | `t`, `t_h`, `demand` |
| — | every state, by layout name | `Pr_h1`, `Tc2_h2`, `cv_1`, `rrod_1` |
| — | every signal | `Pn`, `DI`, `Tavg`, `ps`, `pp` |
| `u_` | every actuator | `u_utg`, `u_vrod_M`, `u_iheat` |
| `xc_` | controller states | `xc_valve_pi.I` |

So after a run you have states, derived signals, and what each controller
commanded, aligned on one time axis. You never need `IU` or `plant.ix` to read
results — only to write controllers.

### `Result.__getitem__`

```python
res["Pn"]   # -> ndarray
```

Shorthand for `res.df["Pn"].to_numpy()`.

### `Result.plot`

```python
plot(cols, x=None, ax=None, subplots=False, label=None, figsize=None, **kw)
```

Plot columns against time.

| | |
|---|---|
| `cols` : list of str | Any columns of `df`. |
| `x` : {`"t"`, `"t_h"`}, optional | Default picks seconds for runs under an hour, hours above. |
| `ax` : Axes, optional | Draw onto existing axes — pass the return of an earlier call to overlay runs. |
| `subplots` : bool, default False | One panel per column. **Use whenever units differ**: `Pn ≈ 1` and `Tavg ≈ 316` on one axis makes the first a flat line. |
| `label` : str, optional | Legend label. With one column it replaces the column name. |

**Returns** — an `Axes`, or an array of `Axes` when `subplots=True`.

Colours come from `Result.COLORS`, a validated categorical palette assigned in
fixed order, so a series keeps its colour when you add or remove one.

```python
res.plot(["Pn", "DI"])                      # one axis
res.plot(["Pn", "Tavg"], subplots=True)     # different units
ax = res_p.plot(["ps"], label="P")          # overlay two runs
res_pi.plot(["ps"], ax=ax, label="PI")
```

### `Result.setpoint`

```python
setpoint(ax, y, label=None) -> ax
```

Draw a horizontal reference line in neutral grey, so colour always means "a
controller" and never "a target".

### `Result.summary`

```python
summary(cols=("Pn", "DI", "Tavg", "ps", "pp", "P_tur")) -> str
```

Start / min / max / end table as text.

---

# Writing controllers

## `Controller`

Base class. Override `update`; declare `states` only if you need memory.

```python
class MyController(Controller):
    name   = "my_controller"      # used in xc_ column names
    states = ()                   # names of your own integrator states
    Ts     = None                 # None = continuous; a number = sampled

    def initialize(self, ctx):
        return np.zeros(len(self.states))

    def update(self, t, sig, xc, u, ctx):
        ...
        return EMPTY              # or np.array([...])
```

### `update`

| argument | |
|---|---|
| `t` : float | Simulation time, seconds. |
| `sig` : dict | `plant.signals(x)` at this instant. |
| `xc` : ndarray | This controller's own states. |
| `u` : ndarray, shape (7,) | The input vector — **mutate in place**. |
| `ctx` : [`LoopContext`](#loopcontext) | Trim point, demand, live state. |

**Returns** — the derivative of `xc` (continuous) or its next value
(sampled); `EMPTY` if stateless.

### Rules

**Rod commands accumulate, every other actuator is assigned.**

```python
u[IU["vrod_M"]] += v      # correct — lets two loops share the rod groups
u[IU["utg"]]     = cmd    # correct
```

Using `=` on a rod silently deletes whatever another controller wrote.

**Integrators must be declared states, never Python attributes.** In continuous
mode the solver calls `update` at times it chooses, including on steps it later
rejects and re-takes. A `self.I += e*dt` would count discarded work. Declaring
`states` hands the job to the solver.

**In sampled mode (`Ts` set) that restriction lifts.** `update` is called
exactly once per sample with `t` increasing, so `self.u_prev`, a ring buffer, a
covariance matrix or a mode flag all work. This is what makes MPC, discrete
observers, `sign()` switching, rate limits and latching logic possible.

**Continuous controllers must be continuous.** `sign()`, relays and hysteresis
collapse the step size. Use `tanh(s/φ)`, or set `Ts` and sample.

### Signs

A wrong sign does not raise an error — the run grinds and only reports when the
solver gives up. Check against:

- A valve step of **+0.05** moves `ps` by **−0.144 MPa**.
- Positive `vrod` **raises** power; `vrod_AO` **raises** `DI`, `vrod_M` **lowers** it.

---

## `LoopContext`

Passed to every controller as `ctx`.

| attribute | |
|---|---|
| `plant` : [`Plant`](#plant) | |
| `x0` : ndarray | Trim state. |
| `u0` : ndarray | Trim inputs — the baseline your controller should offset from. |
| `sig0` : dict | Signals at the trim point. Use for default setpoints: `ctx.sig0["ps"]`. |
| `demand` : callable | `ctx.demand(t)` — the load profile. |
| `x` : ndarray | **Live state**, refreshed before every call. Read-only. |

### `ctx.state`, `ctx.states`

```python
ctx.state("Tc2_h2")                 # one state by name -> float
ctx.states("Pr_h1", "Pr_h2")        # several -> ndarray
```

Full-state access, for state feedback, pole placement, sliding surfaces or MPC.

---

# Built-in controllers

## `Manual`

```python
Manual(input_name, fn, accumulate=False)
```

Drive any input from a function of `(t, sig)`. The quickest way to apply a step
or probe an actuator.

```python
base = u0[IU["utg"]]
Manual("utg", lambda t, sig: base + (0.05 if t > 20 else 0))
Manual("utg", lambda t, sig: base + 0.01*np.sin(0.1*t))
Manual("utg", lambda t, sig: base - 0.1*(sig["Pn"] - 1))   # proportional
```

Set `accumulate=True` to use `+=` (required for rod inputs).

## `GridDemand`

```python
GridDemand()
```

Sets `Pdem` from the load profile. Feedforward; no feedback.

## `ValveOnDemand`

```python
ValveOnDemand()
```

Governor valve tracks load demand directly. This is what makes a run
*turbine-leading*: the secondary moves first and the reactor follows.

## `RodsOnPower`

```python
RodsOnPower(kp=3000.0, v_max=72.0, deadband=0.0)
```

Common-mode rod speed tracks the power demand. No integrator state is needed —
the rod drive already integrates, so a proportional law on rod *speed* is
integral action on power.

## `RodsOnTavg`

```python
RodsOnTavg(program=None, kp=6.0, v_max=72.0, deadband=0.55)
```

Common-mode rod speed holds `Tavg` on its load program. The reactor half of
realistic load following: the turbine takes the load, primary temperature
moves, the rods bring `Tavg` back. Power is never commanded.

`program` defaults to [`tavg_program(plant)`](#tavg_program).

## `AxialOffsetBand`

```python
AxialOffsetBand(band=0.03, kp=2000.0, v_max=72.0, target=0.0)
```

Differential rod speed holds `DI` inside a dead band. With the default
parameters the axial xenon mode is unstable, so without this `DI` saturates on
any transient longer than a few hours.

The differential is `vrod_AO += vd`, `vrod_M -= vd`, which raises `DI` for
`vd > 0` under the default `rod_map`. **Both lines must be flipped if you pass a
`rod_map` that swaps the halves** — otherwise the loop drives `DI` away from the
band instead of into it, and nothing raises an error.

## `SGPressurePI`

```python
SGPressurePI(kp=0.537, ki=0.117, ref=None, u_min=0.0, u_max=1.5, t_track=10.0)
```

Valve holds steam pressure, with back-calculation anti-windup. Used in
*reactor-leading* mode. `ref` defaults to `ctx.sig0["ps"]`.

## `PressurizerHeaterPI`

```python
PressurizerHeaterPI(kp=13.0, ki=1.3, ref=None, i_max=1.3, t_track=60.0)
```

Heater holds pressuriser pressure **from below only** — output is clipped at
zero, so it cannot bring pressure down. Spray is deliberately not shipped: in
the transcribed equation 31 spray *raises* pressure, which is a flagged
inconsistency.

## `DiscretePI`

```python
DiscretePI(signal, actuator, kp, ki, Ts=1.0, ref=None, lo=-1e9, hi=1e9,
           sign=1.0, rate_limit=None, t_track=30.0, name=None)
```

Zero-order-hold PI from any signal to any actuator.

| | |
|---|---|
| `signal` : str | A key of `plant.signals`, e.g. `"ps"`. |
| `actuator` : str | A key of `IU`, e.g. `"utg"`. |
| `Ts` : float | Sample period, seconds. |
| `sign` : float | `+1` for `error = ref − y`; `-1` when the process gain is negative (as for valve → `ps`). |
| `rate_limit` : float, optional | Max change per second. Needs the previous command, which is only legal because the controller is sampled. |

```python
DiscretePI("ps", "utg", kp=0.537, ki=0.117, Ts=2.0, sign=-1.0, lo=0., hi=1.5)
```

## `SlidingAO`

```python
SlidingAO(K=2.0, lam=0.2, Ts=5.0, target=0.0, v_max=72.0, boundary=0.0)
```

Relay sliding-mode control of axial offset, on surface
`s = d(DI)/dt + λ·(DI − target)`, with differential rod speed `K·sign(s)`.

Uses both v2 features: `sign()` needs sampling to be integrable, and `d(DI)/dt`
is computed from `ctx.x` because it is not in `sig`. Set `boundary > 0` to
replace `sign` with `tanh(s/boundary)`.

Same differential convention and same caveat as
[`AxialOffsetBand`](#axialoffsetband).

Relative degree from rod speed to `DI` is 2 (speed → reactivity → power), which
is why the surface carries a derivative term.

---

# Vajpayee's control loops (paper section 5)

The seven loops from Vajpayee Table 2, implemented as shipped controllers. Each
names the paper equation it comes from.

> **The rod-loop gains in his Table 2 do not transfer.** Eq 50 reads
> `dρrod/dt = G·vrod` with `G` in cents/step and `vrod` in steps per **minute**,
> while every other derivative in the paper is per **second**. Taken literally,
> full rod speed injects ~70 cents/s — prompt critical in 1.4 s. This model
> divides by 60, which recovers a factor of 60; the remaining ~5 orders are
> unrecoverable from the text. The defaults below are retuned for this
> implementation. Read `VAJPAYEE_GAIN_NOTE` before using his numbers.

## `CoreNeutronicsPI`

```python
CoreNeutronicsPI(kp=2.0e3, ki=2.0e2)
```

Eq 55. Reads `ilo` (ex-core detector current), drives both rods in common mode.
Retuned — see the note above.

## `TemperaturePI`

```python
TemperaturePI(kp=1.25e2, ki=5.0e-1)
```

Eq 56. Reads `irtd` (RTD current), drives both rods. Retuned.

## `CombinedPowerPI`

```python
CombinedPowerPI(kp_lo=1.0e3, ki_lo=1.0e2, kp_rtd=60.0, ki_rtd=0.25,
                v_max=72.0, t_track=30.0)
```

Eq 57. Sums the neutronics and temperature loops onto the same rod channel.

## `PressurizerSprayPI`

```python
PressurizerSprayPI(kp=2.935e5, ki=1.695e5)
```

Eq 60, acting on `mdot_spr`. **Ships disabled.** In eq 31 as transcribed the
spray mass-addition term outweighs the condensation term, so spray *raises*
pressure and the loop is positive feedback. See `VAJPAYEE_SIGN_NOTES`.

## `PressurizerLevelPI`

```python
PressurizerLevelPI(kp=1.275e3, ki=7.366e2)
```

Eq 61, acting on `mdot_sur`. His equation commands a quantity his eq 32
computes; with surge as an input (the default) that contradiction disappears.

## `TurbineSpeedPI`

```python
TurbineSpeedPI(kp=0.333, ki=4.3e-3, u_min=0.0, u_max=1.5)
```

Eq 62. Reads `w_tur`, drives `utg` — the loop a real governor runs.

**Defaults are droop-based, not Table 2.** A real governor runs 4–5 % droop, so
`dutg/dw = 1/(0.05 × 60) = 0.333` per Hz. Vajpayee prints `KP = 643`, ~1900×
that, which implies 0.16 % droop; at that gain a 0.13 Hz error commands 80 units
of valve travel against a range of 0–1.5, the valve slams shut and the set
undershoots to 57.7 Hz. The shipped gains hold a 10 % load rejection to a 0.66 Hz
peak and return to 60.003 Hz with no saturation.

**This loop is inert unless `eq38_power_base` is correct.** It is by default —
see below.

## `Params.eq38_power_base`

```python
Params(eq38_power_base=None)    # None -> Pth0, the correct value
```

Power base for eq 38, in watts. Eq 38 reads

```
dw/dt = (P_tur - Pdem) / ((2*pi)^2 * J_tur * w * I_tg)
```

`P_tur` and `Pdem` are normalised everywhere else in this model, but eq 38's
constants only make sense if its power terms are in **watts**:

| reading | implied H | dw/dt at 10 % mismatch |
|---|---|---|
| watts | 23.7 s | 0.127 Hz/s |
| per unit | — | 3.9e-11 Hz/s |

The ratio between those is exactly `Pth0`, which identifies it as a units
mismatch rather than a modelling choice. Feeding per-unit power was a bug in
*this* implementation, not a faithful transcription, so the corrected form is
the default. `Params(eq38_power_base=1.0)` restores the old behaviour, in which
a 10 % load rejection moves turbine speed by 9.4e-8 Hz in 50 minutes and the
governor has nothing to act on.

H = 23.7 s is still ~4× high for a real turbine-generator. To pick an H:

```python
Params(I_tg=1.99642e5 * H_target / 23.7)
```

With the default gains and `H = 23.7 s`, an ungoverned 10 % load rejection runs
the set to 108.6 Hz; with `TurbineSpeedPI` it peaks at 60.13 Hz.

## `LQG`

```python
LQG(plant, actuator, measurement, q=1e-3, r=1e5, xi=5.0, theta=1.0,
    drop=('w_tur',), u_lo=-1e12, u_hi=1e12, accumulate=None, name=None,
    shift=1e-3)
```

Eqs 63–70: Kalman filter plus LQR, designed on the linearised plant.

**The stabilising shift is a workaround, not a design choice.** With a single
actuator several integrators — the other rod group, pressuriser level — are
uncontrollable, so `(A, B)` is unstabilisable and the ARE has no finite solution.
The design is done against `A − shift·I`. There is **no integrator augmentation**
and therefore no guarantee of zero steady-state error.

## `VAJPAYEE_TABLE2`

The paper's gains verbatim, keyed by loop:

```python
CoreNeutronicsPI(**VAJPAYEE_TABLE2["core_neutronics"])   # his numbers, not ours
```

Keys: `core_neutronics`, `temperature`, `sg_pressure`, `pzr_heater`,
`pzr_spray`, `pzr_level`, `turbine_speed`.

**Only two shipped controllers use his gains verbatim**, and provenance differs
loop by loop:

| controller | shipped | [V] Table 2 | |
|---|---|---|---|
| `PressurizerSprayPI` | 2.935e5 / 1.695e5 | same | his — but **ships disabled**, eq 60 is positive feedback as transcribed |
| `PressurizerLevelPI` | 1275 / 736.6 | same | his |
| `SGPressurePI` | 0.537 / 0.117 | 0.5368 / 0.1169 | his magnitudes, rounded to 3 s.f., **sign flipped** for stability |
| `CoreNeutronicsPI` | 2000 / 200 | 0.03087 / 0.003947 | **retuned**, ~6e4x — eq 50 unit error |
| `TemperaturePI` | 125 / 0.5 | 0.00193 / 1.006e-5 | **retuned**, ~6e4x — eq 50 unit error |
| `PressurizerHeaterPI` | 13 / 1.3 | 4092 / 286.1 | **retuned**, ~300x — his gain saturates the 1.3 mA heater on a 0.0003 MPa error |
| `TurbineSpeedPI` | 0.333 / 4.3e-3 | 643 / 8.426 | **retuned** — his implies 0.16 % droop where real machines run 4-5 % |

`VAJPAYEE_GAIN_NOTE` and `VAJPAYEE_SIGN_NOTES` are printable strings explaining
the two classes of problem.

---

# Reproducing the paper

## `Params.as_printed`

```python
Params.as_printed(**overrides) -> Params
```

Every flagged value restored to Vajpayee Table 1 / eq 50 **exactly as printed**.
The shipped defaults deviate in six places because the printed value is
dimensionally impossible or self-inconsistent:

| quantity | shipped | as printed |
|---|---|---|
| `K1p`, `K2p` | × 1e3 | literal |
| `lw0` | 8.5527 m | 28.06 m (exceeds `l` = 14.2524) |
| `h_cow` | 6.974e5 J/kg | 69.74 J/kg |
| `h_lp` | 1.036e6 J/kg | 1036 J/kg |
| `Ctg0` | derived, 297.38 | 2.0481 (×145 smaller) |
| `rod_spm_to_per_s` | 1/60 | 1.0 |

Scores 25/40 against his figures, against 37/40 shipped. **Only the pressurizer
partials are load-bearing** — flipping each of the other five individually
changes the score by at most one comparison. `compare_printed.py` runs the
ablation.

Related fields: `use_printed_Ctg` / `Ctg_printed`, `pzr_partial_scale`.

## `HP_TAU`

```python
import pwr_two_point_plant_v2 as M
M.HP_TAU = False         # module-level; set before constructing a Plant
```

**`True` is the default.** Set it to `False` to restore [V] exactly as printed.

Vajpayee prints the HP turbine's first-derivative coefficient (eq 34) as
`(O_rv + tau_ip)/(tau_hp·tau_ip)`, but the IP and LP stages (eqs 35, 36) — which
have identical structure — both print `(O_rv·tau_hp + tau_ip)/(...)`. The missing
`tau_hp` decides the damping:

| form | ζ | behaviour |
|---|---|---|
| as printed (`False`) | 0.3162 | underdamped, 21 s ringing |
| with `tau_hp` (`True`, **default**) | 1.7322 | overdamped, smooth |

With `tau_hp` the quadratic factors **exactly** into
`(s + 1/tau_hp)(s + 1/tau_ip)` — the residual is 0.00e+00, not merely small. The
corrected equation is therefore two first-order lags in series, which is what an
HP stage feeding an IP stage physically is, and which cannot ring. The printed
form's complex pole pair corresponds to no mechanism in the plant.

Three further arguments put the default on the corrected form: eq 34 as printed is
structurally inconsistent with eqs 35–36, which describe the same kind of stage
and both carry the `tau_hp`; his Fig. 11f shows no ringing; and a reversal count
across all 13 validated panels isolates the ringing to `P_tur` alone — case B
shows 6 direction reversals where every thermal and pressure panel settles
monotonically. The flag takes that 6 to 0 and leaves the others untouched.

Not a solver artefact: `rtol` 1e-7 → 1e-11, `atol` 1e-9 → 1e-14, and
LSODA / BDF / Radau all agree to 3.3e-8.

Module-level because `linearize`'s free function needs it too.

---

# Load profiles

Any `f(t) -> float` in fractional full power works. Two are provided.

## `step`

```python
step(start=1.0, target=0.9, t_step=1800.0) -> callable
```

## `ramp`

```python
ramp(start=1.0, target=0.5, t_start=3600.0, ramp_time=7200.0,
     hold=21600.0, t_return=None) -> callable
```

Down-ramp, hold, up-ramp.

## `tavg_program`

```python
tavg_program(plant, full_power_tavg=None) -> callable
```

Programmed average coolant temperature against load, `prog(load) -> °C`.

**Derived, not assumed.** Trim this model at constant steam pressure and `Tavg`
comes out exactly linear in load, extrapolating to 288.06 °C at zero load —
which is `Ts0`, the secondary saturation temperature. At no load the primary
sits at secondary saturation, which is the physically right answer and good
evidence the loop model is coherent. Span is about 27.9 °C.

---

# Preset configurations

## `turbine_leading`

```python
turbine_leading(plant=None, demand=None, ao_band=0.03, pzr=True, **kw)
    -> ClosedLoop
```

**How a grid-connected plant actually behaves.** Valve tracks load demand,
steam flow changes, primary temperature moves, rods bring `Tavg` back to
program. Reactor power is never commanded — it is the outcome. Steam pressure
is not regulated and stays within about ±1 % on its own.

Controllers: `GridDemand`, `ValveOnDemand`, `RodsOnTavg`, `AxialOffsetBand`,
and `PressurizerHeaterPI` if `pzr`.

## `reactor_leading`

```python
reactor_leading(plant=None, demand=None, ao_band=0.03, pzr=True, **kw)
    -> ClosedLoop
```

Rods command power; the valve regulates steam pressure. A useful contrast, but
not how a grid-connected plant behaves.

Controllers: `GridDemand`, `RodsOnPower`, `AxialOffsetBand`, `SGPressurePI`,
and `PressurizerHeaterPI` if `pzr`.

## `open_loop`

```python
open_loop(plant=None, power=1.0, **kw) -> ClosedLoop
```

No controllers. Every actuator sits at its trim value.

---

# Linear analysis

## `linearize`

```python
linearize(plant, x0, u0, scaled=True) -> (A, B, C, D, S)
```

Numerical Jacobians by central finite differences with relative perturbations.

**Returns** — `A` (n×n), `B` (n×7), `C` (8×n), `D` (8×7), and `S` (n,), the
diagonal scaling applied: `A_scaled = S⁻¹ A S`, `B_scaled = S⁻¹ B`,
`C_scaled = C S`.

Outputs are [`OUTPUT_NAMES`](#name-tables); inputs are [`U_NAMES`](#name-tables).

**Always keep `scaled=True`.** The states span seven decades; eigenvalues of
the unscaled matrix are unreliable. Even scaled, `cond(A) ≈ 1.2e26`.

**Notes**

The model has 5 eigenvalues at the origin (rod reactivity, turbine speed,
pressuriser level) and an unstable complex pair near +5.7e−5 (the axial xenon
mode). `w_tur` is an *uncontrollable* integrator — remove that state before any
Riccati solve, or LQR returns a marginally stable loop.

Linear validity: peak relative error in `Pn` against the non-linear plant is
0.4 % at a 1 % valve step, 4.2 % at 10 %, 8.4 % at 20 %. The 5 % crossing is
around a 12 % step.

## `dc_gain`

```python
dc_gain(A, B, C, D) -> ndarray, shape (8, 8)
```

Steady-state gain, `−C A⁻¹ B + D`, via pseudo-inverse.

> **Not valid for the rod inputs.** Rod inputs are *speeds*, so they integrate;
> `A` is singular in those directions and the pseudo-inverse returns a
> confident-looking number that means nothing. Measure rod gains with step
> responses instead.

## `freq_response`

```python
freq_response(A, B, C, D, w, iu, iy) -> ndarray, complex
```

Frequency response from input index `iu` to output index `iy` over the angular
frequency array `w`.

```python
w = np.logspace(-4, 1, 240)
g = freq_response(A, B, C, D, w, IU["vrod_M"], OUTPUT_NAMES.index("Pn"))
```

## `typical_scales`

```python
typical_scales(plant) -> ndarray, shape (n,)
```

Hand-set typical magnitude per state; what `linearize` uses for scaling.

---

# Name tables

| constant | contents |
|---|---|
| `U_NAMES` | `['vrod_M', 'vrod_AO', 'utg', 'iheat', 'mdot_spr', 'Pdem', 'T_fw', 'mdot_sur']` |
| `IU` | `{name: index}` into the input vector |
| `ROD_IDX` | `(0, 1)` — the two accumulating channels |
| `OUTPUT_NAMES` | `['Pn', 'DI', 'Tavg', 'ps', 'pp', 'P_tur', 'w_tur', 'lw']` |
| `CORE_HALF` | the 12 state names repeated per axial half |
| `LOOP` | the 10 primary loop temperature states |
| `TRIM_UNKNOWNS` | the 19 states the trim solver iterates on |
| `EMPTY` | `np.zeros(0)` — return this from a stateless controller |

## The inputs

| input | units | notes |
|---|---|---|
| `vrod_M` | steps/min | Mrod **speed**. Drives the **bottom** half (half 1) under the default `rod_map` (Li Case 2). Positive = withdraw = power up, `DI` down. Limit ±72. |
| `vrod_AO` | steps/min | AOrod speed. Drives the **top** half (half 2). Positive raises both `Pn` and `DI`. Limit ±72. |
| `utg` | — | Governor valve command. 1.0 ≈ wide open. |
| `iheat` | mA | Pressuriser heater current. Saturates near 1.3. |
| `mdot_spr` | kg/s | Pressuriser spray. **Sign is suspect** — see `PressurizerHeaterPI`. |
| `Pdem` | per unit | Grid load demand. A *disturbance*, not a control input. |
| `T_fw` | °C | Feedwater temperature. A disturbance. |
| `mdot_sur` | kg/s | Pressuriser **surge flow**, positive *into* the pressuriser. Also serves as CVCS charging/letdown, which enters the mass balance identically. `mdot_ch` is a backwards-compatible alias for the same index. |

`mdot_sur` is the actuator for `PressurizerLevelPI` and the channel Vajpayee
perturbs in his section 8.3.1. It is a strong input: **+1 kg/s held for 100 s
moves `pp` by +0.32 MPa**, about eight times anything eq 32 generates on its own.
See [`Config.pzr_surge`](#pzr_surge).

Rods are **velocity** inputs: `dρrod/dt = G·vrod/60`, so holding +5 steps/min
keeps adding reactivity indefinitely. To set a rod *position*, write
`x[plant.ix["rrod_1"]]` in the initial condition.

## The state vector (default `Config`, 53 states)

| indices | states |
|---|---|
| 0–11 | half 1 (bottom): `Pr_h1`, `C1_h1`…`C6_h1`, `I_h1`, `X_h1`, `Tf_h1`, `Tc1_h1`, `Tc2_h1` |
| 12–23 | half 2 (top): same names with `_h2` |
| 24–33 | loop: `Trxu`, `Thot`, `Tsgi`, `Tp1`, `Tp2`, `Tm1`, `Tm2`, `Tsgu`, `Tcold`, `Trxi` |
| 34–36 | `ps`, `pp`, `lw` |
| 37–42 | turbine: `Php_1`, `Php_2`, `Plp_1`, `Plp_2`, `Plp_3`, `w_tur` |
| 43–47 | `cv_1`, `cv_2` (valve), `Qheat`, `rrod_1`, `rrod_2` |
| 48–52 | sensors: `ilo`, `ilr_1`, `ilr_2`, `Trtd1`, `Trtd2` |

---

# Low-level

## `run`

```python
run(dfun, x0, t_end, u_of_t, n_out=1500, method="BDF", rtol=1e-8, atol=1e-10)
```

Module-level integration with any derivative function and any
`u_of_t(t, x) -> u`. For when `ClosedLoop` is in the way.

## `const_u`

```python
const_u(u) -> callable
```

A constant input policy, for use with `run`.

## `freeze`

```python
freeze(plant, keep) -> callable
```

Returns a derivative wrapper that zeroes every state whose name does not start
with one of the `keep` prefixes. Used to exercise one subsystem at a time.

```python
d = freeze(plant, ["ps"])          # only the steam pressure state evolves
```

## `consistency_report`

```python
consistency_report(plant) -> None
```

Prints the parameter audit: internal balance checks, and the six flagged
transcription corrections with the reasoning for each.

## `build_layout`

```python
build_layout(cfg) -> (ix, names)
```

The state layout for a given `Config`. Called by `Plant.__init__`.

---

# Troubleshooting

| symptom | cause |
|---|---|
| Run hangs, or "step size less than spacing between numbers" | Sign error in a controller. Check the two anchors under [Signs](#signs). |
| `KeyError: 'pp'` from `plant.ix` | Built with `Config(pressurizer=False)`. |
| Trim residual above 1e-10 | Not an equilibrium. Do not proceed. |
| Plot is a flat line at the bottom | Mixed scales — pass `subplots=True`. |
| x-axis spans 0 to 0.08 | Plotted against `t_h` on a short run. |
| No plot appears | Script rather than notebook — needs `plt.show()`. |
| Integrator winds up, slow recovery | Clipping without anti-windup. |
| One rod loop has no effect | Used `=` instead of `+=` on a rod input. |
| Controller state never changes | Returned `EMPTY` instead of a derivative. |
| Power diverges in under a second | Sign error on a temperature coefficient, or reactivity in cents fed to an equation expecting Δk/k. |
| Sampled run very slow | Cost scales with `t_end / Ts`. Sample only the loop that needs it. |

## What the model cannot do

No actuator exists for feedwater flow, boron, charging flow, steam dump or
coolant flow, so no control strategy can manipulate them. There is no steam
generator level state (feedwater is assumed to match steam flow instantly), no
peaking factors, no rod insertion limits, no decay heat, and no trip or scram
logic. Adding any of these means a model change, not a controller.
