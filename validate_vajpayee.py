#!/usr/bin/env python3
"""
validate_vajpayee.py -- shape validation against Vajpayee Figs. 10 and 11.

Section 7 of the paper perturbs one input at t = 20 s and reports the open-loop
response of 16 variables over 200 s, as four cases:

    Case A   control rod movement   step in reactivity          (sec 7.1)
    Case B   valve coefficient      step increase in u_tg       (sec 7.2)
    Case C   heater input           +50 % pulse, t = 20..100 s  (sec 7.3)
    Case D   feed-water temperature ramp, t = 20..100 s         (sec 7.4)

Vajpayee's core is single-point with no xenon, so this runs with
Config(two_point=False, xenon=False) -- the axial split collapsed and the
poison removed, which is the honest comparison.

    python validate_vajpayee.py

Writes validation_fig10.png and validation_fig11.png, panel-for-panel in the
paper's order, plus a table of start/end values against the paper's figures.
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pwr_two_point_plant_v2 import (IU, ClosedLoop, Config, Controller, EMPTY,
                                    Manual, Params, Plant, Result, trim)

T_END, T_PERT = 200.0, 20.0
#: Matched to [V]'s own figures so the two can be read side by side:
#: case A blue, B red, C purple, D green.  Validated colourblind-safe
#: (worst adjacent pair dE 16.9 deutan, 20.9 normal; all >= 3:1 on surface).
COLORS = ["#1f6fb4", "#c0392b", "#7d3c98", "#1e8449"]


# ---------------------------------------------------------------------------
# the four input perturbations
# ---------------------------------------------------------------------------
RHO_A = 8.0e-4            # 80 pcm = 0.8 mK/K, the step [V] Fig. 11g shows

#: Control-bank travel, steps.  NOT in [V] Table 1 -- it is the standard
#: Westinghouse figure (228 steps, 231 with overlap) from the Technology Course
#: R-104P Manual he cites in sec 5.1.3.  His own Fig. 11h confirms it:
#:     80 pcm / G(0.9679 cents/step) = 12.71 steps;  12.71 / 231 = 5.50 %
#: against the ~5.5 % his panel shows.
ROD_FULL_TRAVEL = 231.0


def run_case_A(plant, x0, u0, t_end, n_out):
    """[V]'s "instantaneous perturbation": an EXACT step in rod reactivity.

    Rods in this model are VELOCITY inputs ([V] eq 50), so no history of `vrod`
    can produce a step in `rrod` -- `rrod` is a state, and any input pulse has
    to ramp it over a finite window.  During that window the fuel is already
    heating and feeding reactivity back, so the TOTAL never reaches the full
    step: a 2 s pulse peaks at 0.54 mK/K and a 0.1 s pulse at 0.76, against his
    0.80.  Shortening the pulse converges on 0.8 but never arrives, and each
    halving needs a smaller `max_step` to be seen at all.

    So step the STATE instead: integrate to t = 20 s, add `RHO_A` to both rod
    reactivity states, and continue.  That is exactly what "instantaneous"
    means, and it needs no fine step bound.
    """
    pre = ClosedLoop(plant, [], x0=x0, u0=u0).run(
        T_PERT, n_out=max(2, int(n_out * T_PERT / t_end)), max_step=T_PERT / 20)
    x_mid = pre.X[:, -1].copy()
    for nm in ("rrod_1", "rrod_2"):                 # the step itself
        x_mid[plant.ix[nm]] += RHO_A
    post = ClosedLoop(plant, [], x0=x_mid, u0=u0).run(
        t_end - T_PERT, n_out=n_out - len(pre.t), max_step=0.25)

    import pandas as pd
    a, b = pre.df.copy(), post.df.copy()
    b["t"] = b["t"] + T_PERT                        # shift onto the real clock
    b["t_h"] = b["t"] / 3600.0
    return pd.concat([a, b], ignore_index=True)


def case_B(plant, u0):
    """+5 % step in the turbine-governor valve signal."""
    b = u0[IU["utg"]]
    return [Manual("utg", lambda t, s: b * (1.05 if t > T_PERT else 1.0))]


def case_C(plant, u0):
    """+50 % heater pulse from t = 20 s to t = 100 s."""
    i_max = 1.3                                     # installed heater current
    return [Manual("iheat", lambda t, s: 0.5*i_max if T_PERT < t < 100.0 else 0.0)]


def case_D(plant, u0):
    """Ramp in feed-water temperature from t = 20 s to t = 100 s, +5 %."""
    T0 = u0[IU["T_fw"]]
    def f(t, s):
        if t <= T_PERT:
            return T0
        if t >= 100.0:
            return T0 * 1.05
        return T0 * (1.0 + 0.05 * (t - T_PERT) / 80.0)
    return [Manual("T_fw", f)]


#: Case A's pulse is 0.1 s wide, so it needs a step bound small enough to see
#: it.  The other three perturbations are slow and do not.
MAX_STEP = {"Case A": 0.02, "Case B": 0.25, "Case C": 0.25, "Case D": 0.25}

CASES = [("Case A", "rod / reactivity step (state step)", None),
         ("Case B", "valve coefficient +5 %", case_B),
         ("Case C", "heater +50 %, 20-100 s", case_C),
         ("Case D", "feed-water temp ramp", case_D)]


def total_reactivity(plant, df):
    """Total reactivity in mK/K, matching [V]'s Fig. 11g axis (1 mK/K = 100 pcm)."""
    p, ref = plant.p, plant.ref_point
    ac = p.alpha_c * (0.5 if plant.half_coolant_coeff else 1.0)
    rho = (df["rrod_1"].to_numpy()
           + p.alpha_f * (df["Tf_h1"].to_numpy() - ref["Tf_h1"])
           + ac * (df["Tc1_h1"].to_numpy() - ref["Tc1_h1"])
           + ac * (df["Tc2_h1"].to_numpy() - ref["Tc2_h1"])
           + p.alpha_p * (df["pp"].to_numpy() - p.pp0))
    return rho * 1000.0


# Fig. 10 and Fig. 11 panel maps: (paper label, df column, axis label)
# Panel tables: (caption, column, y label, (ymin, ymax), y tick, legend loc).
# Captions, labels, limits and ticks are [V]'s own, read off his Figs. 10 and
# 11 at 160 dpi, so the two sets can be laid side by side.  His "(f) Cool-leg"
# spelling in Fig. 10 is reproduced as printed.
DEG = r" ($^o$C)"

FIG10 = [("(a) Reactor power.",   "Pn",     "Normalized power",            (0.95, 1.15),   0.05,  "upper right"),
         ("(b) Fuel temperature.", "_Tf",   "Fuel temperature" + DEG,      (615, 645),     5,     "upper right"),
         ("(c) Coolant node 1 temperature.", "Tc2_h1",
          "Coolant node 1 temperature" + DEG,                              (311, 314),     0.5,   "center right"),
         ("(d) Coolant node 2 temperature.", "Tc2_h2",
          "Coolant node 2 temperature" + DEG,                              (326.5, 329.5), 0.5,   "center right"),
         ("(e) Hot-leg temperature.", "Thot",
          "Hot leg temperature" + DEG,                                     (326.5, 329.5), 0.5,   "center right"),
         ("(f) Cool-leg temperature.", "Tcold",
          "Cold leg temperature" + DEG,                                    (295.5, 298.5), 0.5,   "center right"),
         ("(g) PCL 1 temperature.", "Tp1",
          "Primary coolant lump 1 temperature" + DEG,                      (305.5, 308.5), 0.5,   "upper right"),
         ("(h) PCL 2 temperature.", "Tp2",
          "Primary coolant lump 2 temperature" + DEG,                      (295.5, 298.5), 0.5,   "center right")]

FIG11 = [("(a) MTL 1 temperature.", "Tm1",
          "Metal coolant lump 1 temperature" + DEG,                        (296, 299),     0.5,   "center right"),
         ("(b) MTL 2 temperature.", "Tm2",
          "Metal coolant lump 2 temperature" + DEG,                        (291, 294),     0.5,   "upper right"),
         ("(c) SCL temperature.", "Ts",
          "Secondary coolant lump temperature" + DEG,                      (286.5, 289.5), 0.5,   "center right"),
         ("(d) Steam generator pressure.", "ps",
          "Secondary coolant lump pressure (MPa)",                         (7.1, 7.45),    0.05,  "center right"),
         ("(e) Pressurizer pressure.", "pp",
          "Pressure at pressurizer (MPa)",                                 (15.4, 15.48),  0.01,  "center right"),
         ("(f) Turbine output.", "P_tur",
          "Normalized turbine power output",                               (0.995, 1.035), 0.005, "center right"),
         ("(g) Total reactivity.", "_rho",
          r"Total reactivity ($\Delta$mK/K)",                              (-0.1, 0.9),    0.1,   "upper right"),
         ("(h) Actuator outputs.", "_act",
          "Variation in actuator output (%)",                              (-5, 55),       10,    "upper right")]

# what the paper's figures show, read off Figs 10 and 11, for the table
PAPER = {
 "Pn":    {"start": 1.000,  "A": 1.018, "B": 1.021, "C": 1.000, "D": 0.972},
 "_Tf":   {"start": 626.66, "A": 634.0, "B": 634.0, "C": 626.7, "D": 619.5},
 "Tc2_h1":{"start": 312.13, "A": 313.8, "B": 311.4, "C": 312.2, "D": 312.8},
 "Thot":  {"start": 327.30, "A": 329.2, "B": 327.05,"C": 327.3, "D": 327.5},
 "Tcold": {"start": 296.96, "A": 298.3, "B": 295.85,"C": 297.0, "D": 298.0},
 "Tm1":   {"start": 297.40, "A": 298.8, "B": 296.3, "C": 297.4, "D": 298.5},
 "Ts":    {"start": 288.06, "A": 289.3, "B": 286.6, "C": 288.1, "D": 289.3},
 "ps":    {"start": 7.280,  "A": 7.410, "B": 7.140, "C": 7.280, "D": 7.410},
 "pp":    {"start": 15.410, "A": 15.410,"B": 15.410,"C": 15.4725,"D": 15.410},
 "P_tur": {"start": 1.000,  "A": 1.018, "B": 1.029, "C": 1.000, "D": 1.018},
}


def run_all(printed: bool = False, surge: bool = False,
            scale: Optional[float] = None, hp_tau: bool = True):
    """Run the four cases.

    printed : bool
        `False` (default) uses the shipped parameters, which deviate from [V]
        Table 1 in six places -- see VALIDATION.md section 2.1.  `True` uses
        `Params.as_printed()`, every flagged value exactly as the paper prints
        it, eq 34 included (`--printed` forces `hp_tau=False`).  Expect 40/40
        and 30/40 respectively.
    surge : bool
        ADD [V] eq 32 -- surge computed from coolant expansion -- on top of the
        `mdot_sur` input.  `False` (default, and the standard configuration)
        leaves surge as a pure input, which is what reproduces Fig. 11e.
        `True` matches his section 7.1-7.2 text instead.  See VALIDATION.md 2.1c.
    scale : float, optional
        Override `pzr_partial_scale`.  `700` is the value fitted against the
        one case in Fig. 11e that moves; the shipped default is `1000`.

    The shipped default (`surge=False`) scores 40/40.  With `scale=700` it
    matches Fig. 11e to four decimal places on all four cases.
    """
    import pwr_two_point_plant_v2 as _M
    _M.HP_TAU = hp_tau
    kw = {} if scale is None else dict(pzr_partial_scale=scale)
    p = Params.as_printed(**kw) if printed else Params(**kw)
    plant = Plant(p, Config(two_point=False, xenon=False, pzr_surge=surge))
    x0, u0, res = trim(plant, 1.0)
    print(f"plant: {plant.n} states  (two_point=False, xenon=False)")
    print(f"parameters: {'[V] Table 1 AS PRINTED' if printed else 'shipped defaults'}"
          f"   pzr_partial_scale = {p.pzr_partial_scale:g}")
    print(f"surge ([V] eq 32): {'INPUT + computed expansion' if surge else 'INPUT only (standard)'}")
    print(f"HP turbine (eq 34): {'+tau_hp, overdamped (default)' if hp_tau else 'AS PRINTED, zeta=0.32, rings'}")
    print(f"trim residual {res:.2e}\n")

    runs = {}
    for tag, desc, builder in CASES:
        if builder is None:                         # Case A: step the state
            d = run_case_A(plant, x0, u0, T_END, 1200)
        else:
            loop = ClosedLoop(plant, builder(plant, u0), x0=x0, u0=u0)
            d = loop.run(T_END, n_out=4000, max_step=MAX_STEP[tag]).df.copy()
        d["_rho"] = total_reactivity(plant, d)
        # Vajpayee has ONE fuel node; the two-point core has one per half and
        # they sit either side of his value (619.06 / 634.23, mean 626.64 vs
        # his 626.66).  The mean is the like-for-like comparison.
        d["_Tf"] = 0.5 * (d["Tf_h1"] + d["Tf_h2"])
        # actuator trace, as % deviation from its trim value
        # [V] Fig. 11h plots "% variation in actuator output".  For a channel
        # whose trim value is non-zero (utg, T_fw) that is % of the trim value.
        # For the heater, whose trim value is ~0, % of trim is meaningless --
        # normalise by the INSTALLED capacity instead, which is what makes his
        # case C read 50 % for a "+50 % pulse".  Dividing by the ~0 base gives
        # 65 % and does not match his panel.
        act = {"Case A": "u_vrod_M", "Case B": "u_utg",
               "Case C": "u_iheat", "Case D": "u_T_fw"}[tag]
        base = d[act].iloc[0]
        full = {"Case C": 1.3}.get(tag)           # heater installed current, mA
        ref_v = full if full is not None else (abs(base) if abs(base) > 1e-9 else 1.0)
        d["_act"] = (d[act] - base) / ref_v * 100.0
        if tag == "Case A":
            # [V] sec 7.1: "The % variation in REACTIVITY is shown in Fig. 11h."
            # Each of secs 7.1-7.4 names a different quantity for this panel --
            # reactivity, valve coefficient, heater output, feed-water
            # temperature -- so (h) is "% variation in whatever was perturbed".
            #
            # Rods are a velocity input, but his own G converts reactivity to
            # rod steps, and rod steps over full bank travel is a percentage:
            #     rrod / G = steps;  steps / 231 = fraction of travel
            # That reproduces his ~5.5 % exactly.  See ROD_FULL_TRAVEL.
            d["_act"] = (d["rrod_1"] / plant.p.G_dkk_per_step
                         / ROD_FULL_TRAVEL * 100.0)
        runs[tag] = d
        print(f"  {tag}: {desc}")
    return plant, runs


def draw(runs, panels, fname, title):
    """Lay the panels out as [V] does: boxed axes, his limits and ticks, a
    framed legend on every panel, and the caption BELOW each axes."""
    fig, axs = plt.subplots(4, 2, figsize=(11.5, 15.5))
    for ax, (cap, col, ylab, ylim, ytick, loc) in zip(axs.ravel(), panels):
        for (tag, _d, _b), c in zip(CASES, COLORS):
            ax.plot(runs[tag]["t"], runs[tag][col], color=c, lw=1.5, label=tag)
        ax.set_xlim(0, 200)
        ax.set_xticks(np.arange(0, 201, 20))
        ax.set_ylim(*ylim)
        # ticks on multiples of `ytick`, not on the axis limit: his (h) runs
        # -5..55 but is ticked 0, 10, ... 50.
        t0 = np.ceil(ylim[0] / ytick - 1e-9) * ytick
        ax.set_yticks(np.arange(t0, ylim[1] + ytick / 2, ytick))
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7.5, loc=loc, framealpha=1.0,
                  edgecolor="#b8b6b1", fancybox=False)
        # [V] captions sit under the axes, not above them
        ax.text(0.5, -0.30, cap, transform=ax.transAxes, ha="center",
                va="top", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=4.0)
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  -> {fname}")


def compare(runs):
    print("\n" + "=" * 74)
    print("SHAPE VALIDATION -- end value at t = 200 s, model vs paper figures")
    print("=" * 74)
    print(f"{'variable':9s} {'case':6s} {'start':>9s} {'model end':>10s} "
          f"{'paper end':>10s} {'sign':>6s}")
    agree = total = 0
    for col, ref in PAPER.items():
        for tag in ("Case A", "Case B", "Case C", "Case D"):
            d = runs[tag]
            y0, y1 = d[col].iloc[0], d[col].iloc[-1]
            pe = ref[tag.split()[-1]]
            dm, dp = y1 - y0, pe - ref["start"]
            # "flat" if the paper moves it by less than 1 % of the largest move
            scale = max(abs(ref[k] - ref["start"]) for k in "ABCD")
            flat = abs(dp) < 0.05 * scale if scale > 0 else True
            ok = (abs(dm) < 0.05 * scale) if flat else (np.sign(dm) == np.sign(dp))
            agree += ok; total += 1
            print(f"{col:9s} {tag.split()[-1]:6s} {y0:9.3f} {y1:10.3f} "
                  f"{pe:10.3f} {'OK' if ok else 'MISMATCH':>6s}")
    print("=" * 74)
    print(f"direction of change agrees in {agree}/{total} comparisons")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    printed = "--printed" in argv
    surge = "--surge" in argv
    scale = None
    # --printed means EVERYTHING as the paper prints it, eq 34 included.
    # (run_all(printed=True, hp_tau=True) still gives the mixed case from the
    #  Python API if you want it.)
    hp_tau = ("--no-hp-tau" not in argv) and not printed
    if "--scale" in argv:
        scale = float(argv[argv.index("--scale") + 1])

    tag = ("_printed" if printed else "") + ("_surge" if surge else "") \
          + ("" if (hp_tau or printed) else "_eq34printed")
    sub = (" — Table 1 as printed" if printed else "") + \
          (" — eq 32 surge added" if surge else "") + \
          ("" if (hp_tau or printed) else " — eq 34 as printed")

    plant, runs = run_all(printed, surge, scale, hp_tau)
    print()
    draw(runs, FIG10, f"validation_fig10{tag}.png",
         f"Validation against Vajpayee Fig. 10 — open loop, "
         f"perturbation at t = 20 s{sub}")
    draw(runs, FIG11, f"validation_fig11{tag}.png",
         f"Validation against Vajpayee Fig. 11 — open loop, "
         f"perturbation at t = 20 s{sub}")
    compare(runs)
