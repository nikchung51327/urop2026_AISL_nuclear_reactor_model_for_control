# Control examples

Worked examples for comparing control algorithms on `pwr_two_point_plant_v2.py`.
Every number below is measured, not asserted — paste the snippet and you get it.

**Start with `QUICKSTART.md`.** It covers the plant, trim, step responses, and
P / PI / PID from nothing, and this page assumes all of it. What follows is the
multivariable half of the problem — the part a single loop cannot reach.

| # | example | what it teaches |
|---|---|---|
| 1 | Mrod alone | one group, and what it leaves behind |
| 2 | AOrod alone | the second controlled variable |
| 3 | 2×2 PID on both rods | pairing, and a second integrator |
| 4 | constrained MPC on both rods | prediction and limits |
| 5 | PID vs MPC | a fair comparison, and where each breaks |
| 6 | feedforward and cascade | two structures beyond a single loop |

Common preamble for every example:

```python
import numpy as np
from pwr_two_point_plant_v2 import *

plant = Plant()
x0, u0, res = trim(plant, 1.0)
```

---
## Plotting setup

Every figure below uses this style module. Save it as `viz.py` beside your
script. The three series colours are a validated categorical palette — worst
all-pairs colour-blind separation ΔE 9.2, normal-vision ΔE 24.0 — assigned in
fixed order so a series keeps its colour when you add or drop one.

```python
# viz.py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                        # blue, orange, aqua
RAMP   = ["#bdd7f2", "#8ab8e8", "#4a94dd", "#2a78d6", "#17508f"]  # sequential blue
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#d8d7d2", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2.0, "savefig.dpi": 130,
    "figure.constrained_layout.use": True,
})

def setpoint(ax, y, label=None):
    """Reference line in neutral ink — never a series colour."""
    ax.axhline(y, color=REF, lw=1.0, ls=(0, (4, 3)), zorder=0, label=label)

def finish(ax, ylabel, xlabel="time (s)", legend=True):
    ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
```

Four rules these encode, worth keeping whatever you plot:

- **Never two y-scales on one axis.** `Pn` ≈ 1 and `Tavg` ≈ 316 do not belong on
  the same plot. Use separate panels.
- **Setpoints and demand are not series.** They go in neutral grey, dashed, so
  colour means "controller" and nothing else.
- **Fixed colour order.** Blue is always the first controller, orange the second.
  Dropping a case must not repaint the survivors.
- **Where two traces coincide, change the linestyle.** Identity is never carried
  by colour alone — you will see this in example 5.

---

## Reference: the gain matrix

The measured step gain of each actuator, which the examples below refer back to.
`QUICKSTART.md` sections 6 and 7 derive these; they are repeated here because
every pairing argument on this page rests on them.

```
input        dPn       dDI     dTavg       dps
vrod_M   +0.0131   -0.0986    +1.863   +0.0838
vrod_AO  +0.0264   +0.0854    +1.639   +0.1693
utg      +0.0292   -0.0107    -0.473   -0.1444
```

- **Both rods raise power; they move `DI` in opposite directions.** That is the
  common/differential split, and it is why two loops can share one pair of rods.
- **`utg` moves `ps` hard and `DI` barely.** The valve's natural partner is steam
  pressure, not power.

Do not use `dc_gain(A, B, C, D)` for the rods. Rod inputs are *speeds*, so they
integrate; `A` is singular in those directions and the pseudo-inverse returns a
number that looks meaningful and isn't. Measured step responses are the honest
version.

---

## 1. Mrod alone

```python
class MrodP(Controller):
    name = "mrod_p"
    def __init__(self, kp=3000., v_max=72.):
        self.kp, self.v_max = kp, v_max
    def update(self, t, sig, xc, u, ctx):
        v = float(np.clip(self.kp*(ctx.demand(t) - sig["Pn"]), -self.v_max, self.v_max))
        u[IU["vrod_M"]] += v
        return EMPTY

r = ClosedLoop(plant, [GridDemand(), MrodP(), ValvePI()],
               demand=dem, x0=x0, u0=u0).run(900, n_out=600)
```

```
Pn_end 0.9002  (demand 0.90)      |DI|max 0.45368   DI_end +0.45368
```

Power tracks to four figures with **no integrator state**. The rod drive already
integrates — `dρrod/dt = G·vrod/60` — so a proportional law on rod *speed* is
integral action on power. That is why none of the rod controllers here declare
states for the proportional path.

But `DI` has run to **+0.45** and is still climbing. Mrod acts on the bottom
half only, so holding power with one group tilts the core — and the axial xenon
mode then amplifies the tilt rather than decaying it. Left alone the run
saturates near |DI| ≈ 0.94; it diverges in *either* direction, so the sign here
is set by which half the rod is in, not by the instability. That is what
example 2 is for.

![One rod group at a time](figures/ex07_rod_groups.png)

Plot both outputs or you will not see the problem. The left panel says the
controller works; the right panel says it has tilted the core by 0.45 and is
not finished.

Note `+=`, not `=`. Rod commands accumulate so two loops can share the groups.
Use `=` and you silently delete the other controller.

---

## 2. AOrod alone

```python
class AOrodP(Controller):
    name = "aorod_p"
    def __init__(self, kp=2000., v_max=72., target=0.0):
        self.kp, self.v_max, self.target = kp, v_max, target
    def update(self, t, sig, xc, u, ctx):
        v = float(np.clip(self.kp*(sig["DI"] - self.target), -self.v_max, self.v_max))
        u[IU["vrod_AO"]] += v
        return EMPTY
```

```
AOrod only  -> DI       Pn_end 1.0000   |DI|max 0.00000   DI_end +0.00000
naive both  -> Pn, DI   Pn_end 0.8999   |DI|max 0.02329   DI_end +0.00009
```

**Sign again:** the error is `DI − target`, positive. From the table in example
3, `vrod_AO` moves `DI` *negative*, so positive gain on positive error is
negative feedback. I had this backwards on the first attempt and power ran to
2.16 before the integrator gave up.

Run both groups together and you get power tracking *and* `DI` held — the naive
pairing works. Example 9 asks whether it's the best one.

---

## 3. 2×2 PID on both rods

Two pairings are available. Direct: Mrod→power, AOrod→offset. Modal: common
mode→power, differential→offset.

```python
class RodPID2x2(Controller):
    name = "rod_pid"; states = ("Ip", "Id", "yp", "yd")
    def __init__(self, kp=(3000., 2000.), ki=(0., 0.), kd=(0., 0.),
                 Tf=5.0, modal=True, v_max=72.):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.Tf, self.modal, self.v_max = Tf, modal, v_max
    def initialize(self, ctx):
        return np.array([0., 0., ctx.sig0["Pn"], ctx.sig0["DI"]])
    def update(self, t, sig, xc, u, ctx):
        ep = ctx.demand(t) - sig["Pn"]
        ed = sig["DI"] - 0.0                       # note the sign, as in ex 8
        dyp = (sig["Pn"] - xc[2]) / self.Tf
        dyd = (sig["DI"] - xc[3]) / self.Tf
        cp = float(np.clip(self.kp[0]*ep + self.ki[0]*xc[0] - self.kd[0]*dyp,
                           -self.v_max, self.v_max))
        cd = float(np.clip(self.kp[1]*ed + self.ki[1]*xc[1] + self.kd[1]*dyd,
                           -self.v_max, self.v_max))
        if self.modal:
            vm, va = cp - cd, cp + cd
            k = min(1.0, self.v_max / max(abs(vm), abs(va), 1e-12))
            u[IU["vrod_M"]]  += k * vm
            u[IU["vrod_AO"]] += k * va
        else:
            u[IU["vrod_M"]]  += cp
            u[IU["vrod_AO"]] += cd
        return np.array([ep, ed, dyp, dyd])
```

Pairing, with `ki = 0`:

```
groups   IAE_Pn 1.4554   |DI|max 0.02329   Pn_end 0.8999   DI_end -0.00009   rodmax 72.0
modal    IAE_Pn 1.2428   |DI|max 0.00517   Pn_end 0.8999   DI_end -0.00000   rodmax 72.0
```

Modal is **4.5× tighter on axial offset** and 15% better on power tracking.

![Direct versus modal pairing](figures/ex09_pairing.png)

Both pairings track power identically — the difference is entirely in the right
panel. Set the x-limits to the transient (`ax.set_xlim(40, 400)`); the remaining
500 s is flat and only shrinks what you care about. The reason
is in the gain matrix above: AOrod moves `DI` (+0.085) *more* than it moves power (+0.026), so
the direct pairing has its two loops fighting through the cross-term. The modal
transform diagonalises it.

Now the integral gain, on the modal pairing:

```
ki_power    IAE_Pn    |DI|max   Pn_end
     0.0     1.2428   0.00517   0.8999
     1.0     1.4691   0.00518   0.8997
     5.0     1.9389   0.00519   0.8995
    20.0     2.0650   0.00520   0.9000
```

**Monotonically worse, and it buys nothing** — there was no offset to remove.
The drive already integrates, so adding `ki` stacks a second integrator on a
plant with transport delay. The "PID" that actually works on these rods is
P with optional filtered D.

Note the scaling in the modal branch. The drive limit is **per rod**, so clipping
each mode at 72 and then summing lets the commanded rod speed reach 144. Scale
the pair instead.

---

## 4. Constrained MPC on both rods

### What a QP is, and whether you need one

MPC minimises a quadratic cost over a horizon of future moves. Unconstrained,
that minimisation is a single matrix solve — numpy only. **Constraints** (rod
speed ≤ 72 spm) turn it into a search, and a *quadratic program* solver does
that search.

You don't need a dedicated QP package here. When the only constraints are bounds
on the moves, the problem is a box-constrained least squares, and
`scipy.optimize.lsq_linear` solves exactly that. Real constraint handling, no new
dependency. For *output* constraints (a hard AO band) you would need `osqp` or
similar.

```python
from scipy.optimize import lsq_linear
from scipy.signal import cont2discrete

class RodMPC(Controller):
    name = "rod_mpc"
    def __init__(self, N=30, Ts=2.0, q=(1.0, 3.0), r=1e-9, v_max=72.0,
                 outputs=("Pn", "DI")):
        self.N, self.Ts, self.q, self.r = N, Ts, np.array(q, float), r
        self.v_max, self.outputs = v_max, outputs

    def initialize(self, ctx):
        A, B, C, D, S = linearize(ctx.plant, ctx.x0, ctx.u0)
        iu = [IU["vrod_M"], IU["vrod_AO"]]
        iy = [OUTPUT_NAMES.index(o) for o in self.outputs]
        Ad, Bd, Cd, _, _ = cont2discrete((A, B[:, iu], C[iy, :], 0), self.Ts, method="zoh")
        self.Ad, self.Bd, self.Cd, self.S = Ad, Bd, Cd, S
        self.nu, self.ny, self.nx = Bd.shape[1], Cd.shape[0], Ad.shape[0]
        N = self.N
        Psi   = np.zeros((N*self.ny, self.nx))
        Theta = np.zeros((N*self.ny, N*self.nu))
        Ak = np.eye(self.nx)
        for k in range(N):
            Ak = Ak @ Ad if k else Ad
            Psi[k*self.ny:(k+1)*self.ny, :] = Cd @ Ak
            for j in range(k+1):
                Theta[k*self.ny:(k+1)*self.ny, j*self.nu:(j+1)*self.nu] = \
                    Cd @ np.linalg.matrix_power(Ad, k-j) @ Bd
        self.Psi, self.Theta = Psi, Theta
        self.Wy = np.diag(np.tile(np.sqrt(self.q), N))
        self.Amat = np.vstack([self.Wy @ Theta, np.sqrt(self.r)*np.eye(N*self.nu)])
        self.lo = np.full(N*self.nu, -self.v_max)
        self.hi = np.full(N*self.nu, +self.v_max)
        return EMPTY

    def update(self, t, sig, xc, u, ctx):
        z = (ctx.x - ctx.x0) / self.S                   # scaled deviation state
        tgt = np.array([ctx.demand(t) - ctx.sig0["Pn"], 0.0 - ctx.sig0["DI"]])
        ref = np.tile(tgt, self.N)
        b = np.concatenate([self.Wy @ (ref - self.Psi @ z), np.zeros(self.N*self.nu)])
        sol = lsq_linear(self.Amat, b, bounds=(self.lo, self.hi),
                         lsmr_tol="auto", max_iter=40)
        v = sol.x[:self.nu]                             # receding horizon
        u[IU["vrod_M"]]  += float(v[0])
        u[IU["vrod_AO"]] += float(v[1])
        return EMPTY
```

Two features of the model make this a *simple* MPC rather than a project:

- **No observer needed.** `ctx.x` is the full 53-state vector, so the controller
  does full-state feedback. (This is why v2 exposes the state; v1 gave
  controllers only the 17 signals.) A real plant would need a Kalman filter here.
- **No sampled-time plumbing needed.** Setting `Ts` makes it zero-order-hold
  automatically: `update` fires once per sample and its output is held in
  between.

**Always scale.** `linearize` returns the scaling vector `S`; the states span
seven decades and the condensed matrices are hopeless without it.

Horizon and sample period matter more than the weights:

```
 N   Ts      r        IAE    |DI|max   Pn_end   runtime
20  5.0   1e-06   37.2031    0.01479   0.8211     2.6 s
20  5.0   1e-08   16.9512    0.00540   0.8682     2.0 s
20  2.0   1e-08    3.2436    0.00600   0.8956     6.1 s
60  2.0   1e-09    1.9250    0.00620   0.8982    20.7 s
```

---

## 5. PID vs MPC

Same plant, same valve loop, same demand. `PID` is the modal 2×2 from example 3;
`MPC` is `N=30, Ts=2`.

Both loops here also carry `PressurizerLevelPI()`, and it is **load-bearing, not
decoration.** Surge is an input (`Config.pzr_surge=False`), so with `mdot_sur`
pinned at zero nothing moves mass into or out of the pressurizer. A rod
transient at the 72 spm drive limit expands the coolant faster than the
pressurizer can accommodate, the level state `lw` goes non-physical, and the run
dies inside the solver with `array must not contain infs or NaNs` — no mention of
the pressurizer anywhere in the traceback. Close the level loop, or command
`mdot_sur` yourself, before running anything aggressive. The PID cases survive
without it only because they are gentler.

**Small step, 1.0 → 0.9:**

```
            IAE      |DI|max   Pn_end   rodmax   travel   runtime
PID      1.1914      0.00517   0.8999     72.0    147.4     1.6 s
MPC      2.4023      0.00587   0.8982     72.0    149.7     8.5 s
```

**Large step, 1.0 → 0.7:**

```
            IAE      |DI|max   Pn_end   rodmax   travel   runtime
PID     11.6027      0.00582   0.6998     72.0    151.4     1.4 s
MPC     85.5898      0.03511   0.5996     72.0    147.1     9.9 s
```

PID wins both, and on the large step it is not close. Two separate reasons, both
worth reporting rather than hiding:

![PID versus MPC](figures/ex11_pid_vs_mpc.png)

Small multiples: rows are the two step sizes, columns the two controlled
outputs. Same colour for the same controller across all four panels, so the eye
can carry identity between them.

**MPC has no integral action.** It settles at 0.8982 instead of 0.9000 on the
small step — a pure state-feedback law with a move penalty has a steady-state
offset. The standard fix is an offset-free formulation: augment the model with a
disturbance state, estimate it, and correct the target. That is the first thing
to add.

**The prediction model is linearized at 100% power.** CP8 measures the linear
validity envelope at about a 12% step for 5% error. A 30% step is far outside it,
so the MPC confidently drives to where its model says 0.70 is and lands at 0.60.
PID uses no model and doesn't care. The fix is re-linearizing per operating point
— the CP9 setpoint family is exactly that input — or an LPV/gain-scheduled MPC.

**This is not an argument that MPC is worse.** It is an argument that MPC's
advantages aren't exercised by this problem. The 2×2 rod problem is perfectly
decoupled in modal coordinates (off-diagonal/diagonal = 0.0000 in the modal `C·A·B`),
has no active output constraints, and has one dominant timescale. Those are the
conditions under which a well-tuned PID is hard to beat.

Where MPC should win here, and what to try next:

- **Hard output constraints.** The AO band `|DI| ≤ 0.03` is currently a dead-band
  hack. In MPC it is a row in the QP — but output constraints need a real QP
  solver (`osqp`), not `lsq_linear`.
- **Preview.** Grid load schedules are known in advance. No PID arrangement can
  use that; MPC gets it free by putting the future demand into `ref`.
- **Long-horizon xenon.** The axial mode is unstable with a doubling time of
  hours (CP8 finds a right-half-plane pair). A 60 s horizon cannot see it. A
  two-layer scheme — fast MPC on power, slow outer loop on offset — is the
  realistic structure.

---

## Making the comparison fair

The metrics used above, and why:

```python
def score(r):
    t  = r.df['t'].to_numpy()
    ep = r['Pn'] - r.df['demand'].to_numpy()
    return dict(
        iae    = float(np.trapezoid(np.abs(ep), t)),              # tracking
        dimax  = float(np.abs(r['DI']).max()),                    # the other output
        offset = float(ep[-1]),                                   # does it close?
        rodmax = float(max(np.abs(r.df['u_vrod_M']).max(),
                           np.abs(r.df['u_vrod_AO']).max())),     # saturation
        travel = float(np.abs(np.diff(r.df['u_vrod_M'].to_numpy())).sum()),
    )
```

Four rules that keep a comparison honest:

- **Same trim point.** Pass `x0=x0, u0=u0` to every `ClosedLoop` so none of them
  starts from a different equilibrium — and it skips the 0.6 s trim each time.
- **Report actuator effort, not just error.** `rodmax` tells you whether the
  drive limit was active; `travel` tells you how much wear the controller asks
  for. A controller that wins on IAE by saturating the rods hasn't won.
- **Report both outputs.** A power controller that looks excellent while `DI`
  runs to 0.45 and climbing (example 1) is not excellent.
- **Report runtime.** 1.4 s against 9.9 s is a real difference if you intend to
  run a thousand scenarios.

---

## 6. Beyond one loop: feedforward and cascade

Both are possible, and the code for both is below. One of the two wins only on
one side of the plant, which is the more useful half of this section.

(MPC is in `QUICKSTART.md` section 17, which builds a thirty-line version and
measures it. Example 4 above is the two-input constrained case.)

### What the architecture allows

| structure | how to write it |
|---|---|
| **feedforward** | `ctx.demand(t)` may be called at **any** `t`, including the future. `GridDemand` and `ValveOnDemand` already do this. For *disturbance* feedforward, read the disturbance from `u` and add a term to the actuator. |
| **cascade** | **One** `Controller` containing both loops, with every integrator declared in `states`. |

> **Do not pass a setpoint between two controllers through an attribute.** In
> continuous mode the solver rejects and re-takes steps, so an attribute written
> during a rejected step is garbage. Either keep the cascade inside one
> controller, or sample it.

### Feedforward — it works, and the sign is not obvious

Feed-water temperature is measurable and enters through the steam generator, so
its effect can be cancelled at the valve *before* the shaft responds. A 15 °C
drop at t = 100 s:

| `kff` | peak \|Δω\| (Hz) | IAE speed |
|---|---|---|
| 0 (plain governor) | 0.1307 | 8.151 |
| −0.001 | 0.0835 | 4.800 |
| **−0.002** | **0.0456** | **2.533** |
| −0.0025 | 0.0399 | 2.748 |
| −0.005 | 0.1678 | 10.457 |

Peak excursion falls **2.9×**. The gain is **negative**: colder feed-water lowers
steam pressure, hence flow, hence turbine power, so the shaft *decelerates*
(59.87 Hz at the peak) and the valve must open. Overshoot the gain and the
feedforward becomes the dominant disturbance — at −0.005 it is already worse than
none at all. That is feedforward's standing risk: it acts open-loop, so a wrong
gain is never corrected.

> **Ordering bites here.** Controllers run in list order, so a feedforward
> controller reading `u[IU["T_fw"]]` must come *after* whatever writes it. Put
> the disturbance last and the feedforward term is identically zero — the
> measurement is unchanged to four decimals and nothing warns you. That was a
> real bug while writing this section.

### Cascade — wins on the reactor side, loses on the secondary

A cascade helps when the disturbance enters at the **inner** loop and the outer
variable only sees it later. On this plant that is reactivity: a drift changes
power in seconds but takes a minute to reach average temperature.

`TavgThroughPower` puts an outer `Tavg` loop around an inner `Pn` loop. Against
a 0.8 spm reactivity drift:

| | peak \|ΔTavg\| | peak \|ΔPn\| | end gap | rod travel |
|---|---|---|---|---|
| single, deadband 0.55 | 0.6033 °C | 0.00037 | −0.5496 | 8.9 steps |
| single, deadband 0.0 | 0.0669 °C | 0.00038 | +0.0000 | 5.8 steps |
| **cascade** | **0.0116 °C** | **0.00019** | +0.0000 | **5.0 steps** |

The middle row is the fair comparison — against a single loop with its dead band
removed, the cascade is still **5.8× better on peak temperature error, using 14 %
less rod travel.** That is a real win.

**Three cascades that do not work**, all measured, because the negative results
are what tell you when to reach for this:

- **Outer `Pn` → inner `Tavg`.** Pointless. At steady state power is set by how
  much steam the valve passes, not by the rods, so the outer loop has nothing to
  correct — measured residual power error 5e-5. An early version of this section
  "showed" it working; it was closing a *temperature* gap only because the inner
  loop had no dead band, not because the cascade did anything.
- **Outer speed → inner steam pressure → valve.** Worse than the plain governor:
  peak 0.382 Hz against 0.131. Steam pressure moves *opposite* to flow, so the
  inner loop fights the outer.
- **Outer speed → inner steam flow → valve.** Worse again, 0.586 Hz. The
  valve-to-flow path is essentially algebraic — `ṁ = Ctg·cv·ps`, the only
  dynamics being the valve's own second order — so there is no inner lag to
  tighten and the cascade just adds an integrator and phase lag.

The rule those three illustrate: **a cascade needs genuine dynamics between the
actuator and the inner measurement.** Where the inner path is fast and clean,
an inner loop costs you and buys nothing.

### The code

Both controllers in full. `GovernorWithFF` replaces `TurbineSpeedPI`;
`TavgThroughPower` replaces `RodsOnTavg`.

```python
class GovernorWithFF(Controller):
    """Governor, plus feedforward from the measured feed-water temperature."""
    name   = "gov_ff"
    states = ("I",)

    def __init__(self, kp=0.333, ki=4.3e-3, kff=-0.002,
                 lo=0.0, hi=1.5, t_track=10.0):
        self.kp, self.ki, self.kff = kp, ki, kff
        self.lo, self.hi, self.t_track = lo, hi, t_track

    def initialize(self, ctx):
        self.base = float(ctx.u0[IU["utg"]])
        self.w0   = ctx.plant.p.w_tur0
        self.tfw0 = float(ctx.u0[IU["T_fw"]])
        return np.zeros(1)

    def update(self, t, sig, xc, u, ctx):
        e = self.w0 - sig["w_tur"]
        # the disturbance is an INPUT, so read it from u, not from sig
        d_tfw = float(u[IU["T_fw"]]) - self.tfw0
        raw = self.base + self.kp*e + self.ki*xc[0] + self.kff*d_tfw
        cmd = float(np.clip(raw, self.lo, self.hi))
        u[IU["utg"]] = cmd
        return np.array([e + (cmd - raw) / self.t_track])


class TavgThroughPower(Controller):
    """Cascade.  Outer loop (Tavg) sets a POWER setpoint for the inner loop."""
    name   = "cascade_rx"
    states = ("I_out",)

    def __init__(self, kp_o=0.02, ki_o=2e-4, kp_i=3000.0,
                 sp_max=0.15, v_max=72.0, t_track=30.0):
        self.kp_o, self.ki_o, self.kp_i = kp_o, ki_o, kp_i
        self.sp_max, self.v_max, self.t_track = sp_max, v_max, t_track

    def initialize(self, ctx):
        self.prog = tavg_program(ctx.plant)
        return np.zeros(1)

    def update(self, t, sig, xc, u, ctx):
        e_o  = self.prog(ctx.demand(t)) - sig["Tavg"]        # OUTER: temperature
        raw  = self.kp_o*e_o + self.ki_o*xc[0]
        trim = float(np.clip(raw, -self.sp_max, self.sp_max))
        sp   = ctx.demand(t) + trim                          # -> power setpoint
        v = float(np.clip(self.kp_i * (sp - sig["Pn"]),      # INNER: power
                          -self.v_max, self.v_max))
        u[IU["vrod_M"]]  += v
        u[IU["vrod_AO"]] += v
        return np.array([e_o + (trim - raw) / self.t_track])
```

Only the outer loop declares a state. The inner one does not need an integrator:
a proportional law on rod **speed** is already integral action on power.

The disturbance used above is a plain `Controller`, not a special mechanism:

```python
class ReactivityDrift(Controller):
    name = "drift"
    def update(self, t, sig, xc, u, ctx):
        u[IU["vrod_M"]] += 0.8 if 200 < t < 800 else 0.0
        return EMPTY
```

## Debugging

A wrong sign does not raise an error — the run grinds, and `ClosedLoop.run`
only reports once the solver gives up. Two anchors to check against:

- A valve step of **+0.05** moves `ps` by **−0.144 MPa**.
- Positive `vrod` **raises** power; `vrod_AO` **raises** `DI`, `vrod_M`
  **lowers** it.

Both anchors depend on `Config.rod_map`. The default is Li's Case 2 — Mrod in
the bottom half, AOrod in the top — and **every hardcoded differential sign in
this file assumes it.** Pass a map that swaps the halves and `AOrodP` and
`RodPID2x2` both drive `DI` away from target, with no error raised.

I got the sign wrong three times on these examples: once on `AOrodP`, once on
the differential term in `RodPID2x2`, and once on both at the same time when the
`rod_map` default was corrected on 2026-10-11 and these two were not flipped
with it. The first two showed up as power running away to 2.1 with no error
message; the third showed up only because `verify_guide.py` checks `DI` as well
as `Pn`. That is the argument for the verifier.

If a run takes more than about 30 s (or 60 s with a sampled controller), kill it
and check signs before anything else.
