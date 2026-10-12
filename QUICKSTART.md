# Quickstart

A guided tour of the PWR plant simulator, for someone who knows process control
but not nuclear engineering. By the end you will have written a P, a PI and a
PID controller, seen what each one does, and built the complete control loop a
real plant actually runs.

That loop has three parts, and it is worth having the shape in mind from the
start, because every section after §8 is building one of them:

```
   grid takes less load
        │
        ▼
   TURBINE GOVERNOR ──── measures shaft SPEED ───▶ moves the valve
        │                                            (utg)
        │  steam flow changes, primary temperature moves
        ▼
   RODS ON Tavg ──────── measures average TEMPERATURE ──▶ both rods together
        │                                                  (common mode)
        │  reactor power follows -- it is never commanded
        ▼
   AXIAL OFFSET BAND ─── measures DI ─────────▶ the two rods in opposition
                                                 (differential mode)
```

Read it as three loops that each measure something different and move something
different. The governor never hears about your demand profile; the rods never
hear about power. **Reactor power is an outcome, not a setpoint** — that is the
single most counterintuitive thing about the architecture, and §14 is devoted to
it.

The eventual goal of this model is **axial offset control**, the third loop, so
the tour builds toward it. You need no nuclear background — everything is
explained in terms of heat exchangers, residence times and feedback.

You need Python with `numpy`, `scipy`, `pandas` and `matplotlib`, and the file
`pwr_two_point_plant_v2.py` next to your script or notebook. Nothing else.

---

## 1. The plant, in five minutes

A pressurised water reactor is **two water loops with a heat exchanger between
them**.

```
            ┌─────────── hot leg, 327 °C ──────────┐
            │                                      ▼
     ┌──────┴──────┐                       ┌───────┴───────┐
     │    CORE     │                       │     STEAM     │ ══ steam ══▶ [valve] ══▶ TURBINE
     │ heat source │                       │   GENERATOR   │
     └──────┬──────┘                       └───────┬───────┘
            ▲                                      │
            └─────────── cold leg, 297 °C ─────────┘
```

Follow the arrows: that is **one** circuit. Water leaves the core hot, gives up
its heat in the steam generator, and comes back cold to be heated again. The
second loop is the secondary side, on the right — it is a separate body of
water that boils, and the two never mix.

**The primary loop** runs at 15.4 MPa — high enough that it never boils, which
is what the "pressurised" in the name refers to. The two pipes have names you
will see constantly: the *hot leg* out of the core at 327 °C, and the *cold leg*
back into it at 297 °C. One full lap takes about **10 seconds**. Remember that
number; it is the dead time in every loop you will close.

**The steam generator** is a shell-and-tube heat exchanger. Primary water in the
tubes, secondary water in the shell. The secondary side boils at 7.28 MPa and
288 °C, and the steam goes to the turbine.

**The valve** (`utg`) meters steam to the turbine. This is where the outside
world touches the plant: opening it lets more steam out, which pulls more heat
out of the primary, which cools the core.

Flow rate is constant. Power moves by changing *temperatures*, not flows.

### The core is a heat source that regulates itself

This is the one piece of nuclear physics you need.

The chain reaction is governed by a single dimensionless number, **reactivity**
(ρ). At ρ = 0 the reaction exactly sustains itself and power is steady. Push ρ
positive and power climbs; negative and it falls. Reactivity is the knob.

What makes a reactor controllable is that **ρ depends on temperature, and
negatively**:

> hotter coolant → ρ falls → power falls → coolant cools → ρ rises

That is a proportional feedback loop built into the physics, with a gain set by
the *moderator temperature coefficient* (−1.8 × 10⁻⁴ per °C in this model). If
you are a chemical engineer: it behaves like an exothermic reaction whose rate
constant *decreases* with temperature. Inherently stable, self-regulating.

The practical consequence is large: **open the turbine valve and reactor power
rises on its own**, with no operator action. You will see this in §6. Your
controller's job is mostly to trim this natural response, not to create it.

### The core is tall, and the top and bottom can disagree

The core is about 3.7 m tall. This model splits it into a **bottom half** and a
**top half**, each with its own power. Two numbers describe the result:

| | definition | meaning |
|---|---|---|
| `Pn` | (top + bottom) / 2 | total power, 1.0 = full power |
| `DI` | (top − bottom) / 2 | **axial offset** — how lopsided the core is |

At a healthy steady state `Pn = 1.0` and `DI = 0.0`.

**Why `DI` matters.** If the top half makes more power than the bottom, the fuel
there runs hotter. Local power peaking is what sets the safety margin, so plants
hold `DI` inside a band — typically a few percent.

**Why it is hard.** A fission product called **xenon-135** absorbs neutrons
strongly, and it builds up and decays over *hours*. Where power is high, xenon
accumulates and suppresses power there; meanwhile the other half, with less
xenon, picks up. Half a day later the situation reverses. The result is a slow
oscillation in `DI` with a period of roughly 30 hours, and **in this model it is
unstable** — left alone, `DI` grows until one half of the core has effectively
shut down.

So axial offset control is: *hold a weakly-observable variable inside a narrow
band, against an unstable mode with a ten-hour timescale, using actuators that
also move the thing you are mainly trying to control.* That is the interesting
problem this model exists for.

---

## 2. What you can move

Eight inputs, but only three matter for most work:

| input | units | what it does |
|---|---|---|
| `vrod_M` | steps/min | **Mrod** — control rod group acting on the **bottom** half |
| `vrod_AO` | steps/min | **AOrod** — control rod group acting on the **top** half |
| `utg` | — | governor valve, 1.0 ≈ wide open |

(The others — `iheat`, `mdot_spr` and `mdot_sur` for the pressuriser, and `Pdem`, `T_fw` which
are disturbances rather than controls — are in the API reference.)

> **Which rod sits where.** The default follows Li's Case 2: Mrod's tip is at the
> midplane so moving it sweeps the **bottom** half, while AOrod stays in the
> **top**. His Case 1 instead starts Mrod fully withdrawn so it sweeps the top,
> and holds AOrod still — a different `rod_map`, not just a frozen rod. Both are
> one line: see `Config.rod_map` in the API reference.

**Control rods are speed inputs, not positions.** This surprises everyone. You
command *how fast the rod moves*, in steps per minute, and the reactivity is the
integral of that:

```
d(reactivity)/dt = G · vrod / 60
```

So holding `vrod_M = +5` does not park the rod somewhere — it keeps adding
reactivity for as long as you hold it. Positive means withdrawing the rod, which
raises power. The drive saturates at ±72 steps/min.

This has a consequence you will meet in §10: **a proportional controller on rod
speed is integral action on power.** You get offset-free tracking for free.

**Two rod groups, two jobs.** Because one rod acts on the top and the other on
the bottom, moving them *together* changes total power, while moving them
*oppositely* changes axial offset. That is the natural pairing, and §9 shows it.

---

## 3. First contact

```python
import numpy as np
import matplotlib.pyplot as plt
from pwr_two_point_plant_v2 import *

plant = Plant()
print(f"{plant.n} states, {plant.m} inputs")
```

```
53 states, 8 inputs
```

You will touch about six of those 53. The rest — delayed neutron precursors,
xenon, pipe temperatures, turbine stages — run themselves.

Three ways to refer to things, and you only need the third for now:

```python
plant.ix["Tc2_h2"]     # a state, by name -> an index
IU["utg"]              # an input, by name -> an index
plant.signals(x)       # all the readable quantities -> a dict
```

---

## 4. Finding a steady state

Before you can test a controller you need a starting point where **nothing is
changing**. Otherwise you cannot tell your controller's effect from the drift
that was already happening.

```python
x0, u0, residual = trim(plant, 1.0)      # 1.0 = 100 % power
print(f"residual: {residual:.2e}")

sig = plant.signals(x0)
for k in ("Pn", "DI", "Tavg", "Thot", "Tcold", "ps"):
    print(f"  {k:6s} {sig[k]:10.4f}")
```

```
residual: 5.29e-13
  Pn         1.0000
  DI         0.0000
  Tavg     315.9221
  Thot     327.2997
  Tcold    296.9595
  ps         7.2800
```

`trim` solves for the state where every derivative is zero. The **residual** is
‖dx/dt‖ at that solution — about 1e-13, which is zero to machine precision.

> **Check the residual every time.** If it comes back above about 1e-10, the
> solver found something that is not an equilibrium, and everything downstream
> will drift for reasons you will waste hours chasing.

---

## 5. Simulating, and seeing steady state

The simulator is `ClosedLoop`. With an **empty controller list** it just runs
the plant with every actuator parked at its trim value:

```python
res = ClosedLoop(plant, [], x0=x0, u0=u0).run(600)     # 600 seconds

axs = res.plot(["Pn", "DI", "Tavg", "ps"], subplots=True)
axs[0].set_ylim(0.99, 1.01)
axs[1].set_ylim(-0.01, 0.01)
plt.show()
```

![Steady state](qs/01_steady.png)

Four flat lines. `Pn` goes from 1.000000 to 1.000000 and `ps` holds 7.2800 for
the full ten minutes. **This is your baseline** — every controlled run is judged
against it, and if this drifts, nothing else you do is interpretable.

Two things about plotting, both of which you just used:

- **`subplots=True` whenever units differ.** `Pn ≈ 1` and `Tavg ≈ 316`; on a
  shared axis the first becomes a flat line squashed against the bottom.
- **The time axis picks itself.** Runs under an hour get seconds, longer runs
  get hours. Override with `x="t"` or `x="t_h"`.

Everything from the run is in `res.df`, a pandas DataFrame with about 73 named
columns: every state, every signal, and every actuator prefixed `u_`.

```python
res.df[["t", "Pn", "DI", "ps", "u_utg"]].tail(3)
```

---

## 6. A step input, and the chain it sets off

Now disturb it — but first, how to name an actuator. The eight inputs live in
one array, and `IU` maps a name to its position:

```python
print(U_NAMES)
# ['vrod_M', 'vrod_AO', 'utg', 'iheat', 'mdot_spr', 'Pdem', 'T_fw', 'mdot_sur']
print(IU["utg"])        # 2        <- 'utg' is the third slot
print(u0)               # [0.  0.  0.99963  0.  0.  0.99996  232.2]
print(u0[IU["utg"]])    # 0.99963  <- the valve, at the trim point
```

So `u0[IU["utg"]]` reads "the valve entry of the trimmed input vector". You
could write `u0[2]`, but the name survives you forgetting the order.

`Manual` then drives that one input from a function of time. Here it holds the
valve 5 % further open, starting at t = 20 s:

```python
base = u0[IU["utg"]]      # the trimmed valve position, 0.99963 (not 1.0)

loop = ClosedLoop(
    plant,
    [Manual("utg", lambda t, sig: base + (0.05 if t > 20 else 0))],
    x0=x0, u0=u0,
)
res = loop.run(300)

res.plot(["u_utg", "ps", "Tcold", "Pn"], subplots=True)
plt.show()
```

![Valve step](qs/02_valve_step.png)

Read those four panels left to right — they are a causal chain:

1. **The valve opens** 5 % at t = 20 s.
2. **Steam pressure falls**, 7.280 → 7.135 MPa. More steam is leaving than the
   core is making.
3. **The cold leg cools**, 296.96 → 295.85 °C. A lower secondary pressure means
   a lower boiling temperature, so the steam generator pulls harder on the
   primary.
4. **Reactor power rises**, 1.000 → 1.029.

Step 4 is the self-regulation from §1. Colder water returned to the core raises
reactivity, which raises power — **and nobody moved a control rod.**

> **Use the trimmed baseline, not 1.0.** `u0[IU["utg"]]` is 0.99963. If you
> write `1.0 + 0.05`, you apply a small step at t = 0 before the real one.

### The dead time

Zoom in on the moment the valve opens:

```python
res = loop.run(60, n_out=1200)
ax = res.plot(["Pn"])
ax.set_xlim(18, 30)
plt.show()
```

![Dead time](qs/03_deadtime.png)

Power does not move for about **1 second** after the valve does. That is the
transport delay — colder water has to physically travel from the steam generator
back to the core before the reactivity feedback can act.

This is why aggressive derivative action does not help on this plant (§12), and
why too much integral gain will make a loop oscillate.

---

## 7. A step on the rods

Same tool, different actuator. Watch both outputs:

```python
loop = ClosedLoop(
    plant,
    [Manual("vrod_M", lambda t, sig: 5.0 if t > 20 else 0.0)],
    x0=x0, u0=u0,
)
res = loop.run(300)
res.plot(["Pn", "DI"], subplots=True)
plt.show()
```

![Rod step](qs/04_rod_step.png)

Mrod at +5 steps/min for 280 s moves `Pn` by +0.012 **and `DI` by −0.092**.

Two things worth noticing:

- **`DI` moves more than `Pn` does.** Mrod acts on the bottom half, so it tilts
  the core more than it raises total power — and it tilts it *downward*, making
  `DI` negative. Any power controller using one rod group will quietly distort
  the axial shape.
- **Neither trace is settling.** The rod is still moving, so reactivity is still
  being added. This is the velocity-input behaviour from §2.

Run the same experiment on `vrod_AO` and you get `Pn` +0.025 and `DI` **+0.079**
— same sign on power, opposite sign on offset, because AOrod acts on the top
half. That opposition is what lets two rod groups control two things.

---

## 8. Closing the loop

So far every input was a prescribed function of time — **open loop**. A
controller instead decides what to do based on what the plant is doing.

In this simulator a controller is a class with one method:

```python
class HoldPower(Controller):
    name = "hold_power"

    def update(self, t, sig, xc, u, ctx):
        error = ctx.demand(t) - sig["Pn"]              # where we want - where we are
        speed = float(np.clip(3000.0 * error, -72.0, 72.0))
        u[IU["vrod_M"]]  += speed                      # note +=
        u[IU["vrod_AO"]] += speed
        return EMPTY                                   # no internal state
```

The four arguments you will actually use:

| | |
|---|---|
| `t` | time in seconds |
| `sig` | dict of plant readings right now — `sig["Pn"]`, `sig["DI"]`, `sig["ps"]`… |
| `u` | the input vector. **Write into it**; it is mutated in place |
| `ctx` | the trim point and the demand profile: `ctx.u0`, `ctx.sig0`, `ctx.demand(t)` |

And two rules:

- **Rod commands use `+=`, everything else uses `=`.** Rods accumulate so that a
  power loop and an offset loop can share them. Use `=` on a rod and you
  silently erase the other controller.
- **Clip to the actuator limit.** Rods saturate at ±72 steps/min.

Now put it in a loop alongside the same valve disturbance as §6:

```python
loop = ClosedLoop(
    plant,
    [HoldPower(),
     Manual("utg", lambda t, sig: base + (0.05 if t > 20 else 0))],
    demand=lambda t: 1.0,          # hold 100 % power
    x0=x0, u0=u0,
)
res = loop.run(300)
```

Compare it against the uncontrolled run:

```python
ol = ClosedLoop(plant,
                [Manual("utg", lambda t, sig: base + (0.05 if t > 20 else 0))],
                x0=x0, u0=u0).run(300)

ax = ol.plot(["Pn"], label="no controller")
res.plot(["Pn"], ax=ax, label="rods holding power")
ax.set_ylabel("Pn")
plt.show()
```

![Closed loop vs open loop](qs/06_loop_vs_open.png)

Without control, power settles 2.9 % high and stays there. With control it peaks
at 1.009 and returns to **1.00002** — offset-free, from a purely proportional
law, because the rod drive integrates.

Note `ax = ...` then passing `ax=ax` — that is how you overlay two runs.

---

## 9. Following a demand profile, with a real governor

Real load following means tracking a *schedule*. `step` and `ramp` build one.
And the moment a schedule exists, you need the piece that has been missing: the
**turbine governor**.

```python
demand = step(1.0, 0.9, t_step=60.0)      # grid drops to 90 % at t = 60 s

loop = ClosedLoop(plant,
                  [GridDemand(),       # the grid changes what it takes
                   TurbineSpeedPI(),   # governor: measures SPEED, moves the valve
                   HoldPower()],       # rods hold reactor power (for now)
                  demand=demand, x0=x0, u0=u0)
res = loop.run(600, max_step=2.0)

ax = res.plot(["demand", "Pn"])
ax.set_ylabel("normalised power")
plt.show()
```

![Demand step](qs/07_demand_step.png)

Power tracks to 0.89996, and turbine speed returns to 60.0002 Hz after a peak
excursion of 0.83 Hz. Three controllers are cooperating:

| controller | measures | moves |
|---|---|---|
| `GridDemand` | the profile | `Pdem`, the load the grid takes |
| `TurbineSpeedPI` | **shaft speed** | `utg`, the governor valve |
| `HoldPower` | reactor power | both rods together |

**Why a speed controller, when nothing asked about speed?** Because that is what
a governor physically is. It sits on the shaft and measures one thing. If the
shaft runs fast it closes the valve; if slow, it opens. It has never heard of
your demand profile.

And speed is not a side-effect you can ignore — it is the plant's integrator.
Equation 38 of the model is the shaft torque balance:

```
dω/dt = (P_tur − Pdem) / (inertia)
```

Speed **integrates** the mismatch between what the turbine makes and what the
grid takes. Leave it unwatched and any imbalance accumulates without limit. §16
shows exactly how badly, by taking the governor back out.

Note `max_step=2.0`. The governor is a fast loop on a slow plant; without a step
bound the solver can stride over the valve's response. It costs little here.

`ctx.demand(t)` inside your controller is this profile. Swap in
`ramp(1.0, 0.5, ...)` for a slow load reduction.

---

## 10. A proportional controller

Now build P, PI and PID properly, on a loop that shows textbook behaviour: the
**valve holding steam pressure** while the rods move power around.

Why this loop and not the rods? Because of §2 — rod controllers get integral
action free from the drive, so they never show steady-state offset. The valve is
a normal actuator, so it behaves the way a control course says it should.

```python
class ValveP(Controller):
    name = "valve_p"

    def __init__(self, kp=0.5):
        self.kp = kp

    def update(self, t, sig, xc, u, ctx):
        error = sig["ps"] - ctx.sig0["ps"]             # see the sign note below
        u[IU["utg"]] = ctx.u0[IU["utg"]] + self.kp * error
        return EMPTY
```

> **The sign.** Textbooks write `error = setpoint − measurement`. Here it is the
> other way round, because **opening the valve lowers pressure**. If pressure is
> too high we want the valve to open further, so a *positive* error must produce
> a *positive* valve command. Get this backwards and the simulation will hang
> rather than raise an error — see §15.

`ctx.sig0["ps"]` is steam pressure at the trim point, which makes "hold where we
started" the default setpoint.

```python
demand = step(1.0, 0.9, t_step=60.0)
res_p = ClosedLoop(plant, [GridDemand(), HoldPower(), ValveP()],
                   demand=demand, x0=x0, u0=u0).run(400)

ax = res_p.plot(["ps"], label="P")
res_p.setpoint(ax, plant.p.ps0, "setpoint")
ax.set_ylabel("steam pressure (MPa)")
plt.show()
```

![P control offset](qs/08_p_offset.png)

Pressure drops 0.16 MPa and **never comes back**. That is steady-state offset,
and the reason is structural: a proportional controller only produces output
when there *is* an error, and holding the valve at its new position requires a
permanent one.

`setpoint()` draws the target in neutral grey, so colour always means "a
controller" and never "a target".

---

## 11. Adding integral action

To remove the offset the controller needs **memory**. In this simulator memory
means declaring a state:

```python
class ValvePI(Controller):
    name   = "valve_pi"
    states = ("I",)                    # one internal state, named I

    def __init__(self, kp=0.5, ki=0.12):
        self.kp, self.ki = kp, ki

    def update(self, t, sig, xc, u, ctx):
        error = sig["ps"] - ctx.sig0["ps"]
        u[IU["utg"]] = ctx.u0[IU["utg"]] + self.kp*error + self.ki*xc[0]
        return np.array([error])       # d(I)/dt = error
```

Two things changed. `states = ("I",)` declares the integrator, which then
appears as `xc[0]`. And the return value is now the **derivative** of that
state — so returning `error` makes `xc[0]` the integral of the error.

> **Never write `self.I += error * dt`.** The solver is adaptive: it tries a
> step, decides it was too large, throws it away and retries. A hand-rolled
> integrator would count the discarded steps. Declaring the state hands the job
> to the solver, which knows which steps were real.

```python
res_pi = ClosedLoop(plant, [GridDemand(), HoldPower(), ValvePI()],
                    demand=demand, x0=x0, u0=u0).run(400)

ax = res_p.plot(["ps"], label="P")
res_pi.plot(["ps"], ax=ax, label="PI")
res_p.setpoint(ax, plant.p.ps0, "setpoint")
ax.set_ylabel("steam pressure (MPa)")
plt.show()
```

![P vs PI](qs/09_p_vs_pi.png)

| | peak error | final error | IAE |
|---|---|---|---|
| P | 0.1597 MPa | −0.1597 MPa | 49.33 |
| PI | 0.0283 MPa | 0.0000 MPa | 0.84 |

Offset gone, and a 58× improvement on integrated error. The integral term keeps
pushing the valve until the error is exactly zero.

---

## 12. Adding derivative action

Derivative action needs the *rate of change* of the measurement — and you do not
have it, because `sig` gives you values, not slopes. The standard fix is a
**filtered derivative**: pass the measurement through a first-order lag and use
the lag's own derivative. That costs one more state.

```python
class ValvePID(Controller):
    name   = "valve_pid"
    states = ("I", "ps_filt")

    def __init__(self, kp=0.5, ki=0.12, kd=0.5, tau_f=1.0):
        self.kp, self.ki, self.kd, self.tau_f = kp, ki, kd, tau_f

    def initialize(self, ctx):
        return np.array([0.0, ctx.sig0["ps"]])        # start the filter at ps

    def update(self, t, sig, xc, u, ctx):
        error = sig["ps"] - ctx.sig0["ps"]
        d_ps  = (sig["ps"] - xc[1]) / self.tau_f      # filtered derivative
        u[IU["utg"]] = (ctx.u0[IU["utg"]] + self.kp*error
                        + self.ki*xc[0] + self.kd*d_ps)
        return np.array([error, d_ps])
```

`initialize` sets the starting value of each state — the filter must start at
the current pressure, not zero, or it will produce an enormous spurious
derivative in the first instant.

`d_ps` is used twice: once in the control law, and once as the derivative of the
filter state. That is what makes `xc[1]` track `ps` with time constant `tau_f`.

![PI vs PID](qs/10_pi_vs_pid.png)

| | peak error | final error | IAE |
|---|---|---|---|
| PI | 0.02828 MPa | 0.00000 | 0.8446 |
| PID | 0.02875 MPa | 0.00000 | 0.8455 |

**Slightly worse.** Not a tuning failure — a structural result, and the cause is
the dead time from §6. Derivative action works by predicting where the output is
heading, and on a loop with roughly 10 s of transport delay that prediction is
stale by the time it matters. Push `kd` higher and it degrades monotonically.

Reporting the PI and explaining why you stopped there is the correct outcome.

---

## 13. A load-following manoeuvre

Everything so far has been a single disturbance. Load following is the real job:
the grid asks for less power, you give it less power, and later you give it back.

The standard manoeuvre is a **ramp down, a hold, and a ramp back up**. Ramp rates
matter — a real plant moves at a few percent of full power per minute, not in a
step — and `ramp()` builds exactly this shape:

```python
demand = ramp(start=1.0,        # begin at full power
              target=0.9,       # go to 90 %
              t_start=600.0,    # hold full power for the first 10 minutes
              ramp_time=600.0,  # take 10 minutes to get there
              hold=1800.0)      # sit at 90 % for 30 minutes, then ramp back
```

That is 1 % of full power per minute, which is a realistic rate.

Now three controllers, the same arrangement as §9. Nothing new is needed —
`HoldPower` from §8 is reused unchanged:

```python
loop = ClosedLoop(plant,
                  [GridDemand(),      # tell the turbine what the grid wants
                   TurbineSpeedPI(),  # governor holds the shaft at 60 Hz
                   HoldPower()],      # rods bring reactor power to demand
                  demand=demand, x0=x0, u0=u0)

res = loop.run(4800, n_out=1600, max_step=2.0)      # 80 minutes
```

![Load following](qs/11_load_follow.png)

```python
import numpy as np
err = np.abs(res.df["Pn"] - res.df["demand"])
print(f"worst tracking error {err.max():.5f}")       # 0.00058
print(f"power at the end     {res.df['Pn'].iloc[-1]:.5f}")   # 1.00001
print(f"worst speed error    {np.abs(res.df['w_tur'] - 60).max():.4f}")  # 0.0673
```

**Read the three panels in order, because each one tells you something the one
above it hides.**

**Top — it works.** Power tracks the demand ramp to within 0.06 %, and the
turbine follows. Rod speed peaks at 1.7 spm out of the 72 available, so the
manoeuvre is nowhere near an actuator limit. If this were the only plot you
looked at, you would call the job done.

**Middle — the plant moved, and did not quite come home.** `Tavg` falls about
3 °C on the way down and recovers to 315.57 against the 315.92 it started from.
That 0.35 °C is not an error; the core genuinely is in a slightly different
state. Nothing in this loop was watching temperature — `HoldPower` watches
power. This is why real plants run a *programmed* `Tavg` that slides with load,
and §14 replaces `HoldPower` with exactly that.

**Bottom — the part that matters, and the reason for this whole model.** Axial
offset starts at zero, climbs to +0.054 during the hold, drops back when power
returns, and settles at **+0.0279**. It never comes home.

> The next two sections add one loop each, and §15 puts all three runs on the
> same axes so you can see exactly which variable each controller is responsible
> for. If you want the punchline first, skip to that figure.

Nothing controlled it. `HoldPower` drives both rods together, which moves total
power and says nothing about where in the core that power sits. Meanwhile xenon
has been redistributing on its own timescale — hours, not minutes — and the
30-minute hold was long enough for it to start.

So the manoeuvre succeeded on the metric you were watching and left the plant
leaning. Run the same profile twice back to back and the lean accumulates.

**That is the load-following control problem.** Power control alone is not
enough; you need a second loop on the differential rod channel, which is exactly
what §16 points you at.

> Try it: change `hold` to `7200.0` (two hours) and watch how much further `DI`
> goes. Then try `target=0.5` — a deeper manoeuvre moves xenon harder.

---

## 14. Power is an outcome, not a setpoint

`HoldPower` has carried us this far, and it is about to be retired. Look at what
it does:

```python
error = ctx.demand(t) - sig["Pn"]      # commands reactor power directly
```

It reads the demand profile and drives reactor power to match. That works, and
**it is not how a plant runs.**

In a real plant the grid moves the turbine, the turbine changes how much heat
leaves the primary, the primary temperature moves, and the rods bring
temperature back to a programmed value. Reactor power is whatever comes out of
that chain. Nobody ever tells the reactor what power to make.

### The Tavg program

The setpoint the rods actually track is **average coolant temperature**, and it
is not constant — it slides with load:

```python
prog = tavg_program(plant)
prog(1.0)    # 315.92 °C  at full power
prog(0.5)    # 302.0  °C  at half power
prog(0.0)    # 288.06 °C  at zero load
```

That line is **derived, not assumed**. Trim the model at constant steam pressure
and `Tavg` comes out exactly linear in load, extrapolating to 288.06 °C at zero
— which is `Ts0`, the secondary saturation temperature. At no load the primary
sits at secondary saturation, which is physically the right answer and a good
check that the loop model hangs together. The span is about 27.9 °C.

`RodsOnTavg` is that loop, and it is shipped:

```python
loop = ClosedLoop(plant,
                  [GridDemand(),       # grid takes less
                   TurbineSpeedPI(),   # governor opens/closes the valve on SPEED
                   RodsOnTavg()],      # rods hold Tavg on PROGRAM
                  demand=demand, x0=x0, u0=u0)
res = loop.run(4800, n_out=1600, max_step=2.0)
```

Note what is gone: nothing in that list mentions power.

### What it costs, and what it buys

Same 80-minute manoeuvre as §13, the only change being the reactor-side loop:

| | worst `Pn` error | `Tavg` at end | peak speed error |
|---|---|---|---|
| `HoldPower` (power commanded) | **0.00058** | 315.57 °C | 0.0673 Hz |
| `RodsOnTavg` (temperature programmed) | 0.00479 | 315.52 °C | 0.0507 Hz |

Power tracking gets **8× worse**, and that is not a defect — it is the
architecture being honest. `RodsOnTavg` never promised to track power. It
promised to hold temperature on program, and the power that results is within
0.5 % of demand *as a consequence*, with no power measurement anywhere in the
loop.

What you buy is a loop that matches the plant. The rods respond to the thing
they can actually influence quickly, the deadband (±0.55 °C, from Vajpayee's rod
speed program) keeps them still during small excursions instead of hunting, and
the reactor is never asked to chase a number it reaches only through a 10-second
transport lag.

> If you need tight power tracking for a study, `HoldPower` or `RodsOnPower` is
> the right tool and there is nothing wrong with using it. Just do not call the
> result load following — name the architecture you actually ran.

**What §13 was quietly getting wrong.** Look at the middle panel of the figure
in §15. With `HoldPower`, `Tavg` peaks at **318.1 °C**, which is **5.0 °C above its
program** at that moment, and stays high for the whole hold. Nothing in that loop was watching
temperature, so nothing objected. `RodsOnTavg` keeps it inside the dead band the
entire manoeuvre. That is the trade being made: 8× worse on a number nobody
measures in a real plant, in exchange for holding the one they do.

---

## 15. The complete loop

Three loops, three measurements, three actuator modes. This is the whole thing:

```python
demand = ramp(start=1.0, target=0.9, t_start=600.0,
              ramp_time=600.0, hold=1800.0)

loop = ClosedLoop(plant,
                  [GridDemand(),               # the grid's load profile
                   TurbineSpeedPI(),           # SPEED  -> valve
                   RodsOnTavg(),               # Tavg   -> both rods together
                   AxialOffsetBand(band=0.03)],# DI     -> the rods in opposition
                  demand=demand, x0=x0, u0=u0)

res = loop.run(4800, n_out=1600, max_step=2.0)
```

![The complete loop](qs/13_full_loop.png)

### The three architectures side by side

This is the figure to spend a minute on. Same manoeuvre, same governor, same
plant — the only difference between the three traces is which rod loop is
closed.

```python
import matplotlib.pyplot as plt

prog = tavg_program(plant)

# §14 coincides with §13 on DI AND with §15 on Pn and Tavg, so it is drawn
# LAST, thin and dashed, on top of both.  Colour alone would hide one of them.
#            label                        controllers        colour  style  lw  zorder
CASES = [("§13  power held, Tavg and DI free",
          [GridDemand(), TurbineSpeedPI(), HoldPower()],
          Result.COLORS[0], "-", 2.8, 2),
         ("§14  Tavg on program, DI free",
          [GridDemand(), TurbineSpeedPI(), RodsOnTavg()],
          Result.COLORS[1], (0, (5, 3)), 1.6, 4),
         ("§15  + axial offset band",
          [GridDemand(), TurbineSpeedPI(), RodsOnTavg(), AxialOffsetBand(band=0.03)],
          Result.COLORS[3], "-", 2.2, 3)]

runs = {lab: ClosedLoop(plant, c, demand=demand, x0=x0, u0=u0)
                 .run(4800, n_out=1600, max_step=2.0).df
        for lab, c, *_ in CASES}

fig, axs = plt.subplots(3, 1, figsize=(9.5, 9.6), sharex=True)
d0 = runs[CASES[0][0]]
t0 = d0["t"] / 60.0
tgt = np.array([prog(v) for v in d0["demand"]])

axs[0].plot(t0, d0["demand"], color="#8a8984", lw=1.3, ls=(0, (4, 3)), label="demand")
axs[1].fill_between(t0, tgt - 0.55, tgt + 0.55, color="#8a8984", alpha=0.16, lw=0,
                    label="program ±0.55 °C dead band")        # <- the dead band
axs[1].plot(t0, tgt, color="#8a8984", lw=1.3, ls=(0, (4, 3)), label="Tavg program")
for y in (0.03, -0.03):                                        # <- the AO band
    axs[2].axhline(y, color="#8a8984", lw=1.0, ls=(0, (2, 3)), zorder=0)

for lab, _, col, ls, lw, z in CASES:
    d = runs[lab]
    axs[0].plot(d["t"]/60.0, d["Pn"],   color=col, lw=lw, ls=ls, zorder=z, label=lab)
    axs[1].plot(d["t"]/60.0, d["Tavg"], color=col, lw=lw, ls=ls, zorder=z)
    axs[2].plot(d["t"]/60.0, d["DI"],   color=col, lw=lw, ls=ls, zorder=z)

axs[0].set_title("power — commanded in §13, an OUTCOME in §14 and §15",
                 fontsize=9.5, loc="left")
axs[1].set_title("average coolant temperature — uncontrolled in §13, on program from §14",
                 fontsize=9.5, loc="left")
axs[2].set_title("axial offset — free to drift until §15 closes the loop on it",
                 fontsize=9.5, loc="left")
for ax, yl in zip(axs, ("Pn  (reactor power)", "Tavg  (°C)", "DI  (axial offset)")):
    ax.set_ylabel(yl); ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
axs[2].set_xlabel("time (minutes)")
axs[0].legend(fontsize=8, frameon=False, loc="lower right")
axs[1].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("Adding one loop at a time: what each controller is responsible for",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.975])
plt.show()
```

![Adding one loop at a time](qs/14_three_architectures.png)

Each case is a different linestyle as well as a different colour, because §13
and §14 lie on top of each other in the `DI` panel and §14 and §15 lie on top of
each other in the `Pn` panel. Colour alone would not tell you which is which.

Read it one panel at a time, and notice that **each controller fixes its own
panel and ignores the others**:

| | `Pn` | `Tavg` | `DI` |
|---|---|---|---|
| **§13** power held | tracks to 0.00058 | **peaks 5.0 °C above program** | drifts to +0.0279 |
| **§14** + Tavg on program | 0.00479, an outcome | **within 1.0 °C throughout** | drifts to +0.0280 |
| **§15** + axial band | 0.00498 | inside the dead band | **held, ends −0.0071** |

The shaded strip in the middle panel is the **±0.55 °C dead band** around the
program. Inside it `RodsOnTavg` commands zero rod speed — which is why the trace
sits near the lower edge rather than on the line, and why it is not a tuning
fault. Its worst excursion is 1.02 °C, during the ramp, where the dead band and
the rod speed limit together mean it cannot keep up instantaneously.

The bottom panel is the one that justifies the whole model. §13 and §14 are
indistinguishable there: adding a temperature loop did **nothing** for axial
offset, because a common-mode rod command cannot change a tilt. Only §15, which
uses the differential mode, moves that curve at all.

### How two controllers share one pair of rods

This is the part worth understanding properly, because the easy answer is wrong.

Two rods are **two degrees of freedom**, and the loops use different combinations
of them:

```
RodsOnTavg        common mode     vrod_M += v     vrod_AO += v      (together)
AxialOffsetBand   differential    vrod_M -= vd    vrod_AO += vd     (opposed)
```

The `+=` from §8 is what physically lets them share: both write into the same two
inputs and the writes add. Use `=` and the second controller erases the first.

Step each mode and measure *both* outputs:

| mode | `ΔPn` | `ΔDI` |
|---|---|---|
| common (+5, +5) | **+0.0369** | −0.0125 |
| differential (−5, +5) | +0.0129 | **+0.1706** |

At a glance that looks poorly decoupled: moving both rods together to chase
temperature tilts the core by a third as much as it moves power. **But that
ratio is meaningless**, because `Pn` and `DI` are different quantities and
nothing says a unit of one is comparable to a unit of the other. Any claim about
interaction has to survive a change of units.

The measure that does is the **relative gain array**:

```python
G   = np.array([[0.03687, 0.01292],      # rows: Pn, DI
                [-0.01248, 0.17055]])    # cols: common, differential
RGA = G * np.linalg.inv(G).T
# [[0.975 0.025]
#  [0.025 0.975]]
```

**λ₁₁ = 0.975**, where 1.0 is no interaction at all. The common/differential
pairing is almost perfectly decoupled, and rescaling `DI` by any factor leaves
that number unchanged — which is the whole point of using it. (A condition
number does *not* survive that test: rescale `DI` by 10 and `cond(G)` goes from
4.55 to 45.3. Do not quote it as a decoupling measure.)

So the loops do not fight because the pairing is genuinely right: each mode
drives its own variable and leaks ~2.5 % into the other.

**What makes it robust is that the small leakage is also lopsided.** Measured on
the 80-minute manoeuvre:

| | effect |
|---|---|
| the AO loop's effect on `Tavg` | **0.0003 °C** — worst-case error goes 1.0156 → 1.0153 |
| the Tavg loop's effect on `DI` | **0.035** of transient tilt |

The corrective loop is a hundred times gentler on the authoritative one than the
other way round. That is the right direction for a hierarchy:
`AxialOffsetBand` sees the temperature loop's tilt as one more disturbance to
reject, which is its entire job, while the temperature loop barely notices it is
sharing the actuators. Reverse the asymmetry — build a `DI` loop with strong
authority over power — and the two *would* fight, RGA or no RGA.

Two other things keep them apart. **Timescale**: `RodsOnTavg` acts over minutes,
while the axial problem lives in hours, so they rarely pull at once. **The dead
band**: inside ±0.03 the AO loop commands nothing at all, so during ordinary
transients it is not even in the loop.

> **Which part of §14's drift is the leakage?** Run it with
> `Config(xenon=False)` and `DI` peaks at **0.0349** — that transient tilt is the
> common-mode leakage, exactly as the table above predicts — then returns to
> **exactly 0.0000**. With xenon on it peaks at 0.0543 and ends at **+0.0280**.
> So the leakage is real but self-correcting; the permanent offset is entirely
> xenon. Only one of those two needs a controller.

### Why adding the band costs nothing

Over 80 minutes, adding it changes almost nothing except the thing it controls:

| | `DI` peak | `DI` at end | worst `Pn` error | peak speed error |
|---|---|---|---|---|
| no band | 0.0543 | **+0.0280** | 0.00479 | 0.0507 Hz |
| `AxialOffsetBand(0.03)` | **0.0303** | **−0.0071** | 0.00498 | 0.0507 Hz |

Speed error is *identical*. Power tracking degrades in the fifth decimal. The
offset peak is halved and, more importantly, the core comes home instead of
being left leaning at +0.028.

**Reading the third panel.** `Tavg` visibly trails its program and ends about
0.4 °C below it. That is not sloppy tuning — it is the **±0.55 °C dead band**
from §14 doing its job. Inside the band the controller commands zero rod speed,
so the last half-degree is deliberately left uncorrected rather than chased with
constant rod motion. Real rod control programs work this way; a plant that
hunted the last tenth of a degree would wear out its drives for nothing. Set
`RodsOnTavg(deadband=0.0)` and the gap closes to 0.008 °C — but total rod travel
over the manoeuvre rises from **49.0 to 59.8 steps**, 22 % more drive wear to
recover four tenths of a degree that nothing cares about.

### Why the band is not optional

Eighty minutes is too short for the real problem to show. Run the same
architecture over a day — down to 90 %, hold ten hours, back up:

```python
demand = ramp(start=1.0, target=0.9, t_start=3600.0,
              ramp_time=3600.0, hold=36000.0)
res = loop.run(24 * 3600, n_out=2000, max_step=5.0)
```

| `DI` at | 6 h | 12 h | 18 h | 24 h | peak |
|---|---|---|---|---|---|
| no band | +0.44 | +0.88 | +0.94 | **−0.98** | 0.978 |
| with band | +0.030 | −0.030 | +0.007 | +0.029 | **0.030** |

```python
runs24 = {}
for lab, extra in (("no band", []), ("±0.03 band", [AxialOffsetBand(band=0.03)])):
    runs24[lab] = ClosedLoop(plant, [GridDemand(), TurbineSpeedPI(), RodsOnTavg()] + extra,
                             demand=demand, x0=x0, u0=u0
                             ).run(24*3600, n_out=2000, max_step=5.0).df

fig, axs = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
h = runs24["no band"]["t"] / 3600.0

axs[0].plot(h, runs24["no band"]["demand"], color="#8a8984", lw=1.3, ls=(0, (4, 3)))
axs[0].set_ylabel("demand"); axs[0].set_ylim(0.88, 1.02)

axs[1].plot(h, runs24["no band"]["DI"],    color="#8a8984", lw=2.6, ls=(0, (5, 3)),
            label="no band")
axs[1].plot(h, runs24["±0.03 band"]["DI"], color=Result.COLORS[3], lw=1.9,
            label="±0.03 band")
axs[1].set_ylabel("DI  (axial offset)")
axs[1].legend(fontsize=8.5, frameon=False, loc="lower left")

# the two halves' xenon is what drives the panel above
for lab, (c_bot, c_top, ls) in (("no band",    ("#8a8984", "#c0a86a", (0, (5, 3)))),
                                ("±0.03 band", (Result.COLORS[0], Result.COLORS[1], "-"))):
    d = runs24[lab]
    axs[2].plot(h, d["X_h1"], color=c_bot, lw=2.0, ls=ls, label=f"X bottom — {lab}")
    axs[2].plot(h, d["X_h2"], color=c_top, lw=2.0, ls=ls, label=f"X top — {lab}")
axs[2].set_ylabel("xenon, normalised"); axs[2].set_xlabel("time (hours)")
axs[2].legend(fontsize=8, frameon=False, ncol=2, loc="upper left")

for ax in axs:
    ax.axvline(22.3, color="#d8d7d2", lw=1.0, zorder=0)        # where they cross
    ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Twenty-four hours: why the axial band is not optional", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.975])
plt.show()
```

(`demand` here is the 24-hour profile defined just above, not the 80-minute one.)

![Twenty-four hours, with xenon](qs/15_xenon_24h.png)

Without the band the core goes fully lopsided, and look at the sign at 24 h: it
has **crossed over**. The bottom panel is why, and it is worth tracing through,
because it is the mechanism §1 described in words.

**Follow the two grey curves.** Power tilts toward the bottom half early on, so
xenon accumulates there — bottom xenon climbs to **1.76** while the top falls to
**0.78**. Xenon absorbs neutrons, so the bottom half is now being poisoned
exactly where the power is, and it starts to lose out to the clean top half.
Around **hour 22 the two curves cross**, and `DI` flips from +0.94 to −0.98 in
about two hours. That is one half-cycle of the axial xenon oscillation, and in
this model the mode is **unstable**, so it does not decay — the next swing is
bigger.

Note the lag. Xenon peaks hours after the power tilt that created it, because it
is bred from iodine with a ~9 hour half-life. That delay is the whole reason the
mode oscillates rather than simply settling.

**With the band (coloured curves) xenon barely moves at all** — peak 1.057 and
1.039 against 1.76 and 0.78. The controller never lets the tilt get large enough
to build an asymmetric xenon inventory, so the instability has nothing to feed
on. It is cheaper to prevent than to correct.

Peak speed error over the whole day is **0.0106 Hz either way**. The band costs
nothing in the other two loops; it simply uses a channel they were not using.

> Try it: raise `band` to 0.10 and the controller stops acting almost entirely —
> the dead band is wider than the excursion, so `DI` wanders free and diverges
> anyway. Then try `SlidingAO()` in place of the band, which holds `DI` about
> ten times tighter for similar rod motion.

---

## 16. Why not just tell the valve what to do?

Worth seeing what the governor saved you from, because the failure is silent.

There is a tempting shortcut called `ValveOnDemand`:

```python
u[IU["utg"]] = ctx.u0[IU["utg"]] * ctx.demand(t)      # just read the profile
```

It sets the valve to match the demand schedule. Swap it in for `TurbineSpeedPI`
and rerun §13:

![Feedforward valve vs governor](qs/12_governor.png)

**The top panel is the trap.** Power tracking is indistinguishable — 0.00099 for
the feedforward, 0.00058 for the governor. If power were the only thing you
plotted you would conclude the two are equivalent.

**The middle panel is why they are not.** With `ValveOnDemand`, turbine speed
climbs to **63.5 Hz**. With the governor it stays within 0.067 Hz and returns to
60.0001.

A 3.5 Hz excursion on a 60 Hz grid is not a tuning detail, it is a blackout.
Every generator on the network would trip. And it gets worse with the realistic
reactor loop: run `ValveOnDemand` with `RodsOnTavg` instead of `HoldPower` and
speed reaches **77 Hz**, because nothing is forcing power to track demand
quickly any more.

Why *up*, when the grid asked for *less* power? The shaft is a balance of
torques. The grid taking less load is like easing the brake — the steam is still
pushing just as hard, so the shaft accelerates. That is eq 38 from §9 doing its
work.

**The bottom panel holds the deepest lesson.** `ValveOnDemand` puts the valve at
exactly 0.900 of its trim opening — it was told the answer. The governor finds
**0.834**.

The governor is right. Steam flow is `ṁ = Ctg·cv·ps`, and `ps` falls during the
manoeuvre, so holding 90 % power does not need 90 % valve. The feedforward
version "knew" the answer and the answer was wrong; it got away with it only
because the rods quietly absorbed the difference.

> **The general point.** Feedforward needs you to know the correct actuator
> position in advance. Feedback only needs you to measure the consequence. When
> they disagree, feedback is right — it is looking at the plant and feedforward
> is looking at your assumptions.

`ValveOnDemand` still has a legitimate use: when you are studying the **reactor**
and want the secondary to be a clean, predictable boundary condition rather than
a dynamic system of its own. Just never use it to make a claim about grid
behaviour or frequency.

> Try it: `TurbineSpeedPI(**VAJPAYEE_TABLE2["turbine_speed"])` uses the paper's
> published gains instead of the droop-based defaults. The valve slams shut and
> the set undershoots to 57.7 Hz — his `KP = 643` implies 0.16 % droop where real
> machines run 4–5 %.

---

## 17. A model predictive controller, in thirty lines

Everything so far reacts to an error that has already happened. MPC instead
*predicts* what the plant will do and picks the input sequence that works out
best. It is less exotic than it sounds — three ingredients, all of which you
already have.

```python
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import cont2discrete

class SimpleMPC(Controller):
    name = "simple_mpc"
    Ts   = 5.0                      # MPC must be SAMPLED, never continuous

    def __init__(self, N=20, r=1e-5, v_max=72.0, preview=False):
        self.N, self.r, self.v_max, self.preview = N, r, v_max, preview

    def initialize(self, ctx):
        # 1. a linear prediction model, from the plant itself
        A, B, C, D, S = linearize(ctx.plant, ctx.x0, ctx.u0)
        self.S = S
        b = (B[:, IU["vrod_M"]] + B[:, IU["vrod_AO"]]).reshape(-1, 1)   # common mode
        c = C[[OUTPUT_NAMES.index("Pn")], :]
        Ad, Bd, Cd, _, _ = cont2discrete((A, b, c, 0), self.Ts, method="zoh")

        # 2. a condensed prediction:  y = Psi @ z  +  Theta @ v
        N = self.N
        self.Psi = np.vstack([Cd @ np.linalg.matrix_power(Ad, k + 1)
                              for k in range(N)])
        self.Theta = np.zeros((N, N))
        for k in range(N):
            for j in range(k + 1):                       # lower triangular:
                self.Theta[k, j] = (Cd @ np.linalg.matrix_power(Ad, k - j)
                                    @ Bd).item()         # causality
        # stack tracking rows on sqrt(r)*I so ONE least squares does both jobs
        self.M = np.vstack([self.Theta, np.sqrt(self.r) * np.eye(N)])
        return EMPTY

    def update(self, t, sig, xc, u, ctx):
        z = (ctx.x - ctx.x0) / self.S                    # scaled deviation state
        if self.preview:          # MPC may read the FUTURE of the profile
            ref = np.array([ctx.demand(t + (k + 1) * self.Ts) - ctx.sig0["Pn"]
                            for k in range(self.N)])
        else:
            ref = np.full(self.N, ctx.demand(t) - ctx.sig0["Pn"])
        rhs = np.concatenate([ref - self.Psi @ z, np.zeros(self.N)])
        # 3. a box-constrained least squares, re-solved every Ts seconds
        sol = lsq_linear(self.M, rhs, bounds=(-self.v_max, self.v_max),
                         lsmr_tol="auto", max_iter=30)
        v = float(sol.x[0])                              # receding horizon
        u[IU["vrod_M"]]  += v
        u[IU["vrod_AO"]] += v
        return EMPTY
```

Four things in there are worth pausing on.

**`Ts = 5.0` is not optional.** MPC solves an optimisation and uses its own
previous result. A continuous controller is called at times the solver chooses,
including on steps it later rejects and re-takes — see §8. Sampling is what makes
this legal.

**`Theta` is lower triangular, and that triangle is the causality of the model.**
An input at step *j* cannot affect an output before step *j*. Nothing enforces
that; it falls out of writing the prediction honestly.

**Only `sol.x[0]` is used.** The optimiser returns a whole plan, and all but its
first move is thrown away, then the problem is re-solved at the next sample. That
is the *receding horizon*, and it is what makes MPC feedback rather than a
precomputed schedule — the plan is recomputed from where the plant actually is.

**`r` is the move penalty.** Raise it and the controller becomes gentle; lower it
and it drives hard. It is the one knob you will actually turn.

### Seeing the receding horizon

This is the picture that makes MPC click. Each dashed fan is a complete plan the
controller made at one instant — twenty predicted outputs and the twenty rod
moves meant to achieve them. The filled dot is the only move ever applied.

To draw it you need the plan the controller *discards*, so subclass it and keep
a copy:

```python
class RecordingMPC(SimpleMPC):
    """Same controller, but keeps the plan it throws away at each sample."""
    def initialize(self, ctx):
        self.log = []
        return super().initialize(ctx)

    def update(self, t, sig, xc, u, ctx):
        z   = (ctx.x - ctx.x0) / self.S
        ref = np.full(self.N, ctx.demand(t) - ctx.sig0["Pn"])
        rhs = np.concatenate([ref - self.Psi @ z, np.zeros(self.N)])
        sol = lsq_linear(self.M, rhs, bounds=(-self.v_max, self.v_max),
                         lsmr_tol="auto", max_iter=30)
        y_pred = self.Psi @ z + self.Theta @ sol.x + ctx.sig0["Pn"]
        self.log.append((t, sol.x.copy(), y_pred))      # <- the whole plan
        v = float(sol.x[0])
        u[IU["vrod_M"]] += v; u[IU["vrod_AO"]] += v
        return EMPTY

rec = RecordingMPC(N=20, r=1e-5)
d = ClosedLoop(plant, [GridDemand(), TurbineSpeedPI(), rec],
               demand=step(1.0, 0.9, t_step=60.0),
               x0=x0, u0=u0).run(400, n_out=800, max_step=2.0).df

SHOW = [60.0, 75.0, 95.0, 130.0]                        # four sample instants
FAN  = ["#9ec5ea", "#5f9fd8", "#2a78d6", "#17508f"]     # pale -> dark, in time order
plans = [L for L in rec.log if L[0] in SHOW]

fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
axs[0].plot(d["t"], d["demand"], color="#8a8984", lw=1.3, ls=(0, (4, 3)), label="demand")
axs[0].plot(d["t"], d["Pn"], color="#0b0b0b", lw=2.2, label="Pn, what actually happened")
axs[1].step(d["t"], d["u_vrod_M"], color="#0b0b0b", lw=2.2, where="post",
            label="rod speed actually applied")

for (t_k, v_plan, y_pred), col in zip(plans, FAN):
    t_y = t_k + np.arange(1, rec.N + 1) * rec.Ts        # predicted outputs
    t_u = t_k + np.arange(0, rec.N) * rec.Ts            # planned inputs
    axs[0].plot(t_y, y_pred, color=col, lw=1.5, ls=(0, (3, 2)), marker="o", ms=2.6,
                label=f"plan made at t = {t_k:.0f} s")
    axs[1].step(t_u, v_plan, color=col, lw=1.5, ls=(0, (3, 2)), where="post")
    # the filled dot is the ONLY move that is ever applied
    axs[0].plot([t_k], [np.interp(t_k, d["t"], d["Pn"])], "o", ms=5, color=col, zorder=5)
    axs[1].plot([t_k], [v_plan[0]], "o", ms=5, color=col, zorder=5)

axs[0].set_ylabel("Pn"); axs[0].set_xlim(40, 260)
axs[1].set_ylabel("common rod speed (spm)"); axs[1].set_xlabel("time (s)")
axs[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
axs[1].legend(fontsize=8, frameon=False, loc="lower right")
for ax in axs:
    ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("What an MPC is actually doing between samples", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.975])
plt.show()
```

![The receding horizon](qs/16_mpc_horizon.png)

Two things to take from it.

**The plan is thrown away twenty times over.** At t = 60 s the controller lays
out a full trajectory to 160 s; five seconds later it discards it and plans
again from where the plant actually went. Nothing is ever executed as scheduled.
That discarding *is* the feedback.

**The plans are visibly wrong, and that is the real lesson.** Every fan predicts
a gentle settle to ~0.96; the plant undershoots to 0.81. The prediction model
contains **only the rods** — and the governor is moving the valve the entire
time, which is an input the model has never heard of. Freezing the valve cuts the
mean horizon-end prediction error **4×, from 0.064 to 0.016**, which is how we
know that is the cause rather than a guess. The residual 0.016 is linearisation
error: the run leaves the trim point by about 9 %, close to the 12 % validity
edge.

> **MPC is only as good as its model, and "its model" means every input that
> moves the plant — not only the one you are commanding.** A real design fixes
> this with a disturbance model and an offset-free formulation. `SimpleMPC` has
> neither, deliberately, because leaving them out is what makes the failure
> visible.

### The one knob

```python
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
BLUES = ["#8ab8e8", "#4a94dd", "#2a78d6", "#17508f"]   # sequential: r is ordered

for r, col in zip((1e-6, 1e-5, 1e-4, 1e-3), BLUES):
    dd = ClosedLoop(plant, [GridDemand(), TurbineSpeedPI(), SimpleMPC(N=20, r=r)],
                    demand=step(1.0, 0.9, t_step=60.0),
                    x0=x0, u0=u0).run(400, n_out=800, max_step=2.0).df
    axs[0].plot(dd["t"], dd["Pn"], color=col, lw=1.8, label=f"r = {r:g}")
    axs[1].step(dd["t"], dd["u_vrod_M"], color=col, lw=1.8, where="post")
    e = dd["Pn"].to_numpy() - dd["demand"].to_numpy()
    print(f"r = {r:<7g} IAE {np.trapezoid(np.abs(e), dd['t']):7.4f}   "
          f"peak rod {np.abs(dd['u_vrod_M']).max():5.1f} spm")

axs[0].plot(dd["t"], dd["demand"], color="#8a8984", lw=1.3, ls=(0, (4, 3)))
axs[0].set_ylabel("Pn"); axs[0].legend(fontsize=8, frameon=False)
axs[1].set_ylabel("rod speed (spm)")
for ax in axs:
    ax.set_xlabel("time (s)"); ax.set_xlim(40, 300); ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("`r` trades tracking against actuator effort", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
```

![Move penalty](qs/17_mpc_tuning.png)

| `r` | IAE power | peak rod speed |
|---|---|---|
| 1e-6 | 6.0777 | **72.0 spm** (on the stop) |
| 1e-5 | 5.4578 | 22.8 spm |
| 1e-4 | 2.9538 | 3.0 spm |
| 1e-3 | **2.2541** | 0.3 spm |

Lower `r` buys aggression and here it buys nothing: at 1e-6 the controller slams
into the ±72 spm limit and tracking gets *worse*, because a saturated actuator is
an open loop. On this plant the gentle end is better on both counts at once,
which should make you suspicious. The comparison at the end of this section
shows what that suspicion is about.

### Running it, and the result you should not expect

```python
demand = ramp(start=1.0, target=0.9, t_start=600.0,
              ramp_time=600.0, hold=1800.0)
res = ClosedLoop(plant, [GridDemand(), TurbineSpeedPI(), SimpleMPC()],
                 demand=demand, x0=x0, u0=u0).run(4800, n_out=1600, max_step=2.0)
```

| controller | IAE power | peak error | end |
|---|---|---|---|
| **`HoldPower`** from §8 | **0.2444** | **0.00058** | 1.00001 |
| `SimpleMPC`, N=20 | 10.6000 | 0.00939 | 1.00004 |
| `SimpleMPC`, N=20, preview | 10.5376 | 0.00809 | 1.00003 |
| `SimpleMPC`, N=60, preview | 10.5026 | 0.00811 | 1.00000 |

**The six-line controller from §8 beats it by 43×**, and that is not a tuning
failure. It is §2 coming back: the rod drive integrates, so a proportional law on
rod *speed* is already a high-gain **integral** controller on power — offset-free
by construction, with no model and nothing to tune. MPC has no structural edge to
bring against that. It also loses under a tight 5 spm rate limit, normally MPC's
home ground, because the simple law is already driving into the limit in the
right direction.

`preview=True` feeds the **future** of the demand profile into the reference.
No feedback controller can do that — it is feedforward, optimally weighted — and
here it buys almost nothing, for a specific reason: the core responds in seconds
and a realistic load ramp takes ten minutes, so there is nothing to anticipate.
Preview pays when the plant is slow *relative to the reference changes*.

> **None of this says MPC is a bad method.** It says this loop is a bad advert
> for it. MPC earns its keep where the actuator is not already an integrator,
> where constraints bind in a direction a simple law handles badly, or where
> genuine multivariable coupling needs coordinating. `CONTROL_EXAMPLES.md` takes
> the two-input version against a modal 2×2 PID and finds the same answer — which
> is itself worth knowing before you build one.

---

## 18. When something goes wrong

**A wrong sign does not raise an error. The run grinds.** If a simulation takes
more than about 30 seconds, kill it and check signs before anything else. Two
anchors to check against:

- A valve step of **+0.05** moves `ps` by **−0.144 MPa**.
- Positive `vrod` **raises** power; `vrod_AO` **raises** `DI`, `vrod_M` **lowers** it.

Other common ones:

| symptom | cause |
|---|---|
| plot is a flat line at the bottom | mixed units — use `subplots=True` |
| x-axis runs 0 to 0.08 | plotted hours on a short run |
| trim residual above 1e-10 | not an equilibrium; do not proceed |
| one rod controller has no effect | used `=` instead of `+=` |
| controller state never changes | returned `EMPTY` instead of a derivative |
| no plot appears | script not notebook — add `plt.show()` |

---

## 19. Where to go next

You now have everything needed to start on axial offset control. The pieces
already in front of you:

- `DI` is a signal like any other — `sig["DI"]`.
- The rods oppose each other on `DI` (§7), so a **differential** command
  controls it. AOrod drives the top half, so this **raises** `DI`:
  `u[IU["vrod_AO"]] += v` and `u[IU["vrod_M"]] -= v`.
- Xenon makes the mode unstable on a timescale of hours, so your test runs need
  to be long — `run(14*3600)`, not `run(400)`.

Three things to try, in order:

1. **Write your own `DI` controller** on the differential channel and run it
   against `AxialOffsetBand` from §15. A dead band is the crudest thing that
   works; almost anything continuous beats it.
2. **Compare pairings.** Controlling `(Pn, DI)` with `(common, differential)`
   rather than `(Mrod, AOrod)` is measurably tighter, because the modal pairing
   is nearly decoupled — which is the same fact that lets `RodsOnTavg` and
   `AxialOffsetBand` share the rods without fighting.
3. **Go long.** Everything interesting about xenon happens over hours. §15's
   24-hour run is the shortest horizon that shows the instability; a 400-second
   test will tell you your controller is perfect.

The stack from §15 is the thing a real plant runs:

```python
ClosedLoop(plant,
           [GridDemand(),               # the grid asks for a power
            TurbineSpeedPI(),           # SPEED -> valve
            RodsOnTavg(),               # Tavg  -> rods, common mode
            AxialOffsetBand(band=0.03)],# DI    -> rods, differential mode
           demand=demand, x0=x0, u0=u0)
```

Four controllers, two of them sharing the rods — which is why rod commands use
`+=` (§8). Swap the fourth for your own and you are doing axial offset research.

Two more worth knowing about once you are past the basics:

- **`SlidingAO()`** is a sampled sliding-mode controller on the same channel. It
  holds `DI` about ten times tighter than the dead band for similar rod motion,
  and it is the worked example of why `Ts` exists.
- **`Config(pzr_surge=...)` and `Params.as_printed()`** switch the model between
  "reproduces the published figures" and "follows the published equations". They
  are not the same model. `VALIDATION_AND_CONSTRAINTS.md` explains where and why they differ.

For every function, argument and column name, see **`API_REFERENCE.md`**.
