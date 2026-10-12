#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pwr_two_point_plant_v2.py
=========================

VERSION 2.  Same physics as v1, two changes to the control testbed:

  1.  CONTROLLERS CAN READ THE FULL STATE.  `ctx.x` is the live 53-element
      state vector, refreshed before every call to update(), and
      `ctx.state("Tc2_h2")` reads one entry by name.  v1 handed controllers
      only the ~17 signals, which ruled out full-state feedback, pole
      placement, and any sliding surface defined on states.

  2.  CONTROLLERS CAN RUN IN DISCRETE TIME.  Set `Ts` on a Controller and it
      becomes a sampled, zero-order-hold controller: update() is called
      exactly once per sample, its outputs are held constant in between, and
      its states are updated as a difference equation rather than an ODE.
      v1 evaluated every controller inside the ODE right-hand side, at times
      the solver chose, including steps it later rejected -- which made MPC,
      z-domain design, sign() switching, rate limits, latching logic and
      measurement noise impossible.

  Controllers written for v1 run unchanged on v2.

Integrated non-linear PWR plant model: Li's two-point axial core transplanted
into Vajpayee's plant skeleton.  Built to the checkpoint scaffold.

Sources
-------
[V]  Vajpayee, Becerra, Bausch, Deng, Shimjith, Arul (2020),
     "Dynamic modelling, simulation, and control design of a pressurized
     water-type nuclear power plant", Nucl. Eng. Des. 370, 110901.
     -> the plant skeleton and the full SI parameter table (Table 1).
[L]  Li (2014), Ann. Nucl. Energy.
     -> the two-point axial core: kinetics + I/Xe per half, diffusion coupling.
[K]  Kerlin, Katz, Thakkar, Strange (1976), Nucl. Technol. 30(3) 299-316.
     -> H. B. Robinson measurements, used only for validation at CP10.

Architecture
------------
Vajpayee's single-point core (his eqs 3-7) is REPLACED by two Mann stacks in
series, one per axial half, each carrying its own point kinetics, precursors
and iodine/xenon.  Everything outside the core is Vajpayee's, unchanged.

The two interface substitutions are the whole of the joint:
    core exit -> loop :  [V] eq 8 takes Tc2_h2 (half 2, coolant node 2)
    loop -> core inlet:  [V] eq 13 gives Trxi, which becomes [L] T1Cin

Quick start (notebook)
----------------------
    from pwr_two_point_plant_v2 import Plant, turbine_leading, ramp

    loop = turbine_leading(Plant(), demand=ramp(1.0, 0.5, hold=4*3600))
    res  = loop.run(12 * 3600)              # ~5 s wall clock
    res.df.plot(x="t_h", y=["demand", "Pn", "P_tur"])
    print(res.summary())

`res.df` is a pandas DataFrame with every state, every derived signal and
every actuator as a named column.  To test your own strategy, subclass
Controller and hand it to ClosedLoop.  See README.md.

A sampled controller is just a Controller with Ts set:

    class MyDigitalPI(Controller):
        Ts     = 2.0                      # seconds between samples
        states = ("I",)
        def update(self, t, sig, xc, u, ctx):
            e = ...                       # ctx.x is available here
            u[IU["utg"]] = ...
            return np.array([xc[0] + self.Ts * e])   # NEXT value, not d/dt

Mixing is allowed: continuous and sampled controllers in the same loop, each
with its own period.

Validation
----------
The checkpoint suite that exercises and validates this model lives in a
separate script:

    python pwr_validation.py --cp all

CP0 parameter audit, CP0b core-swap arithmetic, CP1-CP6 subsystem build-up,
CP7 trim sweep, CP8-CP9 linearization, CP10 validation against H. B. Robinson,
CP11 model reduction, CP12 load following both architectures, CP13 sampled
control.  Nothing in this module depends on it.
Transcription caveats
---------------------
Every value flagged FLAG below is one where [V]'s printed Table 1 or equation
could not be reconciled with the rest of the paper.  Each is resolved here by
an energy or steady-state balance and the discrepancy is reported by
`consistency_report()`.  Check these against your own copy of the paper before
publishing anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Output location for checkpoint plots
# ----------------------------------------------------------------------------
OUTDIR = os.environ.get("PWR_OUTDIR", "validation")


# ----------------------------------------------------------------------------
# HP turbine damping -- [V] eq 34 vs eqs 35 and 36
# ----------------------------------------------------------------------------
#: [V] prints the first-derivative coefficient of the HP turbine (eq 34) as
#:
#:     (O_rv + tau_ip) / (tau_hp * tau_ip)
#:
#: but the IP and LP stages (eqs 35, 36), which have identical structure, both
#: print
#:
#:     (O_rv * tau_hp + tau_ip) / (tau_hp * tau_ip)
#:
#: The missing `tau_hp` is decisive, not cosmetic.  With `O_rv = 1`,
#: `tau_hp = 10`, `tau_ip = 1.01`:
#:
#:     as printed      zeta = 0.3162   UNDERDAMPED, 21 s ringing
#:     with tau_hp     zeta = 1.7322   overdamped, smooth
#:
#: [V]'s own Fig. 11f shows case B rising smoothly to a single peak of ~1.0315
#: with no ringing, which only the second form produces.  This is the same class
#: of typesetting defect as eq 36's missing denominator (see `Params.as_printed`).
#:
#: THE CORRECTED FORM IS THE DEFAULT.  Three independent arguments agree:
#:
#:   1. eq 34 as printed is structurally inconsistent with eqs 35 and 36, which
#:      describe the same kind of turbine stage and both carry the `tau_hp`;
#:   2. [V]'s own Fig. 11f shows no ringing, which the printed form cannot give;
#:   3. a reversal count across all 13 validated panels isolates the ringing to
#:      `P_tur` alone -- case B shows 6 direction reversals after the
#:      perturbation, where every thermal and pressure panel settles
#:      monotonically.  Setting this flag takes that 6 to 0 and leaves all the
#:      other panels untouched.
#:
#: It is NOT a solver artefact: rtol 1e-7 -> 1e-11, atol 1e-9 -> 1e-14, and
#: LSODA / BDF / Radau all agree to 3.3e-8.
#:
#: `HP_TAU = False` restores [V] exactly as printed.
#:
#: Module-level because `linearize`'s free function needs it too.  Set it before
#: constructing a `Plant`.
HP_TAU = True


# ============================================================================
# 1.  PARAMETERS                                                        (CP0)
# ============================================================================
#
# Units are SI throughout: W, degC, MPa, kg, s, J.  [V] Table 1 is already SI,
# which is why this model does not carry the Kerlin unit-conversion machinery.
#
# REACTIVITY CONVENTION.  [V]'s nomenclature lists rho in cents while Table 1
# gives alpha in dk/k per degC and G in cents/step.  This file works in
# ABSOLUTE dk/k everywhere inside the integrator.  G is converted on the way in.
# Getting this wrong is a factor of ~100 and the model still looks plausible.

@dataclass
class Params:
    # ---- point kinetics, [V] Table 1 -------------------------------------
    lam: np.ndarray = field(default_factory=lambda: np.array(
        [1.2437e-2, 3.05e-2, 1.1141e-1, 3.013e-1, 1.12866, 3.0130]))
    beta_i: np.ndarray = field(default_factory=lambda: np.array(
        [2.15e-4, 1.424e-3, 1.274e-3, 2.568e-3, 7.48e-4, 2.73e-4]))
    Lam: float = 3.0e-5                       # neutron generation time, s

    # ---- core thermal-hydraulics (Mann), [V] eqs 5-7 ---------------------
    Hf: float = 71.8725                       # degC/s per unit power
    Hc: float = 1.1254                        # degC/s per unit power
    tau_f: float = 4.376                      # s
    tau_c: float = 7.166                      # s
    tau_r: float = 0.674                      # s, WHOLE core (halved per half)

    # ---- reactivity coefficients, [V] Table 1 ----------------------------
    alpha_f: float = -2.16e-5                 # dk/k per degC, fuel
    alpha_c: float = -1.80e-4                 # dk/k per degC, PER COOLANT NODE
    alpha_p: float = +1.5664e-4               # dk/k per MPa, primary pressure
    # Xenon worth at full-power equilibrium.  [L] writes alpha_X * Xf0; this is
    # that product, i.e. the reactivity held by equilibrium Xe at 100%.
    # NOT in [V] (his model has no xenon).  Replace with [L] Table 1 if you
    # have it; -0.028 dk/k (2800 pcm) is the standard PWR figure.
    rho_xe_full: float = -0.028               # FLAG: from literature, not [V]

    # ---- iodine / xenon, [L] ---------------------------------------------
    lam_I: float = 2.93e-5                    # 1/s  (I-135, T1/2 6.57 h)
    lam_X: float = 2.106e-5                   # 1/s  (Xe-135, T1/2 9.14 h)
    gamma_I: float = 0.0639                   # fission yield
    gamma_X: float = 0.00237                  # fission yield
    sigma_X_phi0: float = 7.95e-5             # sigma_Xe * phi at full power, 1/s
                                              # (2.65e-18 cm^2 * 3e13 n/cm^2 s)

    # ---- axial coupling, [L] eqs 7-10 ------------------------------------
    # F = 4*d*vs/h^2, units 1/s.  THE one genuinely free parameter of the whole
    # model: [V] has no spatial term and cannot supply it.  Calibrate at CP2 so
    # the axial xenon oscillation period lands in the 20-30 h band.
    #   F = 4*d*nu_s/h^2  with d ~ 0.0085 m (thermal diffusion coefficient),
    #   nu_s ~ 2200 m/s (thermal neutron speed), h ~ 3.66 m (core height)
    #   -> F ~ 5.6 1/s.  Sweep it at CP2 and record what you chose.
    F_couple: float = 5.6                     # FLAG: calibrate at CP2

    # ---- piping and plenums, [V] eqs 8-13, Table 1 ------------------------
    tau_rxu: float = 2.517
    tau_hot: float = 0.234
    tau_sgi: float = 0.659
    tau_sgu: float = 0.726
    tau_cold: float = 1.310
    tau_rxi: float = 2.145

    # ---- steam generator primary / metal, [V] eqs 14-17 -------------------
    tau_p1: float = 1.2815
    tau_p2: float = 1.2815
    tau_pm1: float = 0.5826
    tau_pm2: float = 0.5826
    tau_mp1: float = 0.3519
    tau_mp2: float = 0.1676
    tau_ms1: float = 0.3519
    tau_ms2: float = 0.1676

    # ---- steam generator secondary, [V] eq 24 -----------------------------
    UmsSms1: float = 1.7295e8                 # W/degC
    UmsSms2: float = 3.6312e8                 # W/degC
    Ks: float = 8.1016e7                      # J/MPa
    h_ss: float = 2.7656e6                    # J/kg, steam enthalpy
    cp_fw: float = 5.4791e3                   # J/kg degC
    T_fw0: float = 232.2                      # degC, feedwater
    dTsat_dps: float = 9.47                   # degC/MPa, [V] eq 26
    mdot_sor: float = 2.1642e3                # kg/s, rated steam flow

    # ---- pressurizer, [V] eqs 27-33 --------------------------------------
    Ap: float = 3.566                         # m^2
    l_pzr: float = 14.2524                    # m
    d_w: float = 595.6684                     # kg/m^3
    d_s: float = 100.9506                     # kg/m^3
    nu_w: float = 1.7e-3                      # m^3/kg
    nu_s: float = 9.9e-3                      # m^3/kg
    h_w: float = 1.6266e6                     # J/kg
    h_spr: float = 1.336e6                    # J/kg
    Jp: float = 5.4027                        # conversion factor
    # FLAG -- the four pressurizer partials are the least trustworthy numbers
    # in the whole transcription; [V] Table 1 prints them with unit labels that
    # do not parse (K3p as J/m^3.MPa, K2p and K4p identical).  Taken literally
    # they give dpp/dt ~ 3.2 MPa per kg/s of surge, which runs the pressurizer
    # away in ~20 s and, through alpha_p, shuts the core down -- see CP6.
    # K1p and K2p are density derivatives: for saturated water/steam near
    # 15.4 MPa they are order 1-10 kg/m^3 per MPa, i.e. the printed values are
    # short by 1e3 (a kPa/MPa slip).  pzr_partial_scale applies that factor and
    # is the single knob to revisit if you get a better copy of the table.
    pzr_partial_scale: float = 1.0e3
    K1p: float = -8.152e-3                    # d(d_w)/d(pp), kg/m^3 per MPa
    K2p: float = 4.708e-3                     # d(d_s)/d(pp), kg/m^3 per MPa
    K3p: float = -1.118e-4                    # d(h_w)/d(pp)
    K4p: float = 4.708e-3                     # d(nu_s)/d(pp)
    # V_j * theta_j for [V] eq 32 (surge rate), in his node order:
    # lower plenum, Tc1, Tc2, upper plenum, hot leg, inlet plenum,
    # PCL1, PCL2, outlet plenum, cold leg.
    Vtheta: np.ndarray = field(default_factory=lambda: np.array(
        [0.5991, 0.1814, 0.1814, 1.3164, 0.2752, 2.776,
         0.6022, 0.6022, 0.2776, 0.1927]))    # kg/degC  FLAG: V6 looks like an
                                              # OCR artefact (2.776 vs 0.2776)

    # ---- turbine, [V] eqs 34-38 ------------------------------------------
    tau_hp: float = 10.0
    tau_ip: float = 1.01
    tau_lp: float = 0.4
    F_hp: float = 0.33
    F_ip: float = 0.0                         # zero in Table 1: IP branch inert
    F_lp: float = 0.67
    O_rv: float = 1.0
    kappa_hp: float = 0.8
    J_tur: float = 5.4040
    I_tg: float = 1.99642e5
    w_tur0: float = 60.0                      # Hz
    #: Power base for [V] eq 38, in W.  `None` -> `Pth0`, which is the correct
    #: choice; `1.0` reproduces the original per-unit bug (see below).
    #:
    #: Eq 38 reads  dw/dt = (P_tur - Pdem) / ((2*pi)^2 * J_tur * w * I_tg).
    #: `P_tur` and `Pdem` are NORMALISED everywhere else in this model, but eq
    #: 38's constants only make sense if its power terms are in WATTS:
    #:
    #:     power in W      -> implied inertia 1.079e6 kg.m^2, H = 23.7 s
    #:     power per-unit  -> 1.5e10 x too slow; a 10 % load rejection moves
    #:                        turbine speed by 9.4e-8 Hz in 50 minutes
    #:
    #: The ratio between those two readings is exactly `Pth0`, which is what
    #: identifies this as a units mismatch rather than a modelling choice.
    #: Feeding per-unit power was a bug in THIS implementation, not a faithful
    #: transcription, so the corrected form is the default.
    #:
    #: H = 23.7 s is still ~4x high for a real turbine-generator (3-6 s is
    #: typical).  To pick an H, scale the inertia:
    #:
    #:     Params(I_tg=1.99642e5 * H_target / 23.7)
    eq38_power_base: Optional[float] = None

    # ---- governor valve, [V] eq 51 ---------------------------------------
    zeta_tg: float = 0.4933
    w_n_tg: float = 14.6253                   # rad/s

    # ---- pressurizer heater, [V] eq 52 -----------------------------------
    C_heat: float = 11.3
    R_heat: float = 0.088
    K_heat: float = 1.0e6                     # W/mA  (Table says 1000 kW/mA)

    # ---- control rod drive, [V] eq 50 ------------------------------------
    G_cents_per_step: float = 9.679e-1
    rod_spm_to_per_s: float = 1.0 / 60.0      # vrod is in steps per MINUTE

    # ---- sensors, [V] eqs 45-49 ------------------------------------------
    K_lo: float = 1.95692
    kappa_lo: float = 1.1067e10
    tau_1: float = 5.0e-8                     # neglected by default (see cfg)
    tau_2: float = 2.0e-3
    tau_3: float = 1.01
    tau_4: float = 47.065
    K_lr: float = 8.2
    tau_rtd: float = 1.0
    K_rtd: float = 10.667

    # ---- condenser, [V] eqs 39-42 ----------------------------------------
    tau_co: float = 7.0
    m_coh: float = 41422.9                    # kg
    h_cow: float = 6.974e5                    # FLAG: Table prints 69.74 J/kg,
                                              # which is not an enthalpy
    h_lp: float = 1.036e6                     # FLAG: Table prints 1036 J/kg

    # ---- tabulated initial conditions, [V] Table 1 -----------------------
    Tf0: float = 626.66
    Tc10: float = 312.13                      # single-core node 1
    Tc20: float = 327.30                      # single-core node 2
    Trxu0: float = 327.30
    Thot0: float = 327.30
    Tsgi0: float = 327.30
    Tp10: float = 306.75
    Tp20: float = 296.96
    Tm10: float = 297.41
    Tm20: float = 292.51
    Tsgu0: float = 296.96
    Tcold0: float = 296.96
    Trxi0: float = 296.96
    Ts0: float = 288.06
    ps0: float = 7.28                         # MPa
    pp0: float = 15.41                        # MPa
    lw0: float = 8.5527                       # m  FLAG: Table lists lw0 = 28.06
                                              # which exceeds l = 14.2524
    ilo0: float = 19.65
    ilr0: float = 12.0
    irtd0: float = 14.66
    Trtd10: float = 327.30
    Trtd20: float = 327.30

    #: Use [V] Table 1's printed `Ctg = 2.0481` instead of deriving the valve
    #: coefficient from the SG heat balance.  See `as_printed()`.
    use_printed_Ctg: bool = False
    Ctg_printed: float = 2.0481

    # ---- derived at construction -----------------------------------------
    beta: float = field(init=False)
    tau_r_half: float = field(init=False)
    Pth0: float = field(init=False)           # full core thermal power, W
    Ctg0: float = field(init=False)           # valve coeff, kg/(s MPa)
    G_dkk_per_step: float = field(init=False)

    def __post_init__(self) -> None:
        self.beta = float(np.sum(self.beta_i))
        self.tau_r_half = self.tau_r / 2.0    # CP0b: tau_r halves, tau_f/tau_c do not
        # Full-power thermal duty from the SG heat balance at the tabulated
        # initial conditions.  This is how Ctg0 and Pth0 are pinned; see
        # consistency_report() for why the tabulated Ctg = 2.0481 is not used.
        q0 = (self.UmsSms1 * (self.Tm10 - self.Ts0)
              + self.UmsSms2 * (self.Tm20 - self.Ts0))
        self.Pth0 = q0
        dh = self.h_ss - self.cp_fw * self.T_fw0
        self.Ctg0 = (self.Ctg_printed if self.use_printed_Ctg
                     else (q0 / dh) / self.ps0)   # FLAG: Table prints 2.0481
        self.G_dkk_per_step = self.G_cents_per_step / 100.0 * self.beta
        self.K1p *= self.pzr_partial_scale
        self.K2p *= self.pzr_partial_scale
        if self.eq38_power_base is None:      # [V] eq 38 expects W, not per-unit
            self.eq38_power_base = self.Pth0

    # ------------------------------------------------------------------
    @classmethod
    def as_printed(cls, **overrides) -> "Params":
        """Every flagged value restored to [V] Table 1 / eq 50 **as printed**.

        The shipped defaults deviate from the paper in six places, each because
        the printed value is dimensionally impossible or self-inconsistent (see
        VALIDATION.md section 2.1).  This classmethod undoes all six at once so
        the deviations are auditable rather than baked in:

        ==================  ==================  ========================
        quantity            shipped             as printed
        ==================  ==================  ========================
        K1p, K2p            x 1e3               literal
        lw0                 8.5527 m            28.06 m  (> l = 14.2524)
        h_cow               6.974e5 J/kg        69.74 J/kg
        h_lp                1.036e6 J/kg        1036 J/kg
        Ctg0                derived, 2.0896     2.0481
        rod_spm_to_per_s    1/60                1.0  (eq 50 literal)
        ==================  ==================  ========================

        Three further deviations live outside Table 1 and are NOT undone here,
        because the paper gives nothing to restore them to:

        * eq 36's right-hand side is printed without its denominator; `b0` is
          reconstructed to make the DC gain `F_lp`.
        * eq 31's spray term is already implemented as printed -- which is why
          spray raises pressure and `PressurizerSprayPI` ships disabled.
        * eq 58's sign is already implemented as printed in `VAJPAYEE_SIGN_NOTES`;
          pass `sign=-1.0` to `SGPressurePI` for the literal form.

        Pass `Params.as_printed()` to `Plant` and expect it to fail -- that is
        the point.  See `compare_printed.py`.
        """
        kw = dict(pzr_partial_scale=1.0,
                  lw0=28.06,
                  h_cow=69.74,
                  h_lp=1036.0,
                  use_printed_Ctg=True,
                  rod_spm_to_per_s=1.0)
        kw.update(overrides)
        return cls(**kw)


@dataclass
class Config:
    """Switches the scaffold asks you to make explicitly rather than silently."""
    two_point: bool = True          # False -> both halves forced identical
    xenon: bool = True              # [L]'s contribution; [V] has none
    pressurizer: bool = True        # CP6: goes in from the start
    sensors: bool = True            # ex-core amplifiers + RTDs
    condenser: bool = False         # decoupled from the plant; off by default
    #: ADD [V] eq 32 -- surge computed from primary coolant expansion -- on top
    #: of the `mdot_sur` input.  Default `False`, so surge is purely an input.
    #:
    #: `False` is the standard configuration.  It reproduces [V] Fig. 11e exactly
    #: and is the only reading under which his eq 32, his section 8.3.1 surge
    #: DISTURBANCE and his eq 61 surge COMMAND are mutually consistent.
    #: `True` additionally computes expansion-driven surge, which is physically
    #: complete and matches his section 7.1-7.2 text -- but then the `mdot_sur`
    #: input is superimposed on a quantity the model also calculates, so a
    #: controller acting on it is fighting the plant.  See VALIDATION.md 2.1c.
    pzr_surge: bool = False
    include_ip: bool = False        # F_ip = 0, so these 2 states are inert
    fast_log_amp: bool = False      # True keeps tau_1 = 5e-8 s (brutally stiff)
    # Rod group -> half mapping.  Columns are (Mrod, AOrod), rows are the two
    # halves (half 1 = bottom = [L]'s point 1, half 2 = top = point 2).
    #
    # DEFAULT IS [L]'s CASE 2 ([L] eq 12): Mrod in point 1, AOrod in point 2.
    #
    #   Mrod  -> half 1 (bottom)   dPn = +2.2e-3,  dDI = -2.2e-2
    #   AOrod -> half 2 (top)      dPn = +5.1e-3,  dDI = +1.9e-2
    #   (for a 10 spm / 30 s pulse)
    #
    # PROVENANCE.  Until 2026-10-11 the default was [[0,1],[1,0]] -- Case 2
    # with the two rod labels exchanged -- and the docstring wrongly called it
    # Case 1.  Origin unknown; most likely a transposition of [L] eqs 11/12 when
    # this model was first written.  It was NOT a design decision, despite an
    # earlier comment here claiming it was chosen for Mrod's power authority:
    # that justification was constructed after the error was found.  The two
    # maps are the same plant (identical A; B columns merely exchanged), so
    # nothing was wrong with results produced under it -- only with the label.
    #
    # [L] sec. 6 fixes the geometry: core height 366 cm, point 1 / point 2
    # boundary at 183 cm, AO band [236, 316] cm.  A rod model in differentials
    # tracks where the rod TIP is, not where its body sits, so:
    #
    #   [L] Case 1 (SISO)   np.array([[0., 0.],   Mrod starts at 366 cm, fully
    #                                 [1., 1.]])  withdrawn, so it sweeps the
    #                       TOP half.  AOrod is held motionless, giving a
    #                       rank-1 input matrix and a single-input plant.
    #
    #   [L] Case 2 (MIMO)   np.array([[1., 0.],   Mrod starts at 183 cm, tip at
    #                                 [0., 1.]])  the midplane, so it sweeps the
    #                       BOTTOM half.  AOrod stays in the top. THE DEFAULT.
    #
    # CASE 1 IS NOT CASE 2 WITH AOrod FROZEN.  Freezing AOrod in Case 2 leaves
    # Mrod driving the BOTTOM half (dDI = -0.082 for a 20-step move); Case 1 has
    # it driving the TOP (dDI = +0.072).  Opposite halves, opposite sign.  The
    # cases differ in where Mrod STARTS, which is a different rod_map, not a
    # different command.
    #
    # The map is CONSTANT, so it encodes one fixed assumption about where both
    # tips sit.  It is a linearisation about a nominal insertion and is wrong if
    # a tip crosses the midplane -- nothing detects that (there is no position
    # state; see `U_NAMES`).  Fine for load-following excursions of tens of
    # steps out of a ~231-step bank.
    rod_map: np.ndarray = field(default_factory=lambda: np.array(
        [[1.0, 0.0],
         [0.0, 1.0]]))


# ============================================================================
# 2.  STATE VECTOR LAYOUT
# ============================================================================
#
# One place assembles the state vector.  Every module below takes and returns
# named quantities, never a raw index, because you will reorder states.

CORE_HALF = ["Pr", "C1", "C2", "C3", "C4", "C5", "C6", "I", "X", "Tf", "Tc1", "Tc2"]
LOOP = ["Trxu", "Thot", "Tsgi", "Tp1", "Tp2", "Tm1", "Tm2", "Tsgu", "Tcold", "Trxi"]

#: Input vector.  `mdot_sur` (index 7) is the pressurizer surge flow, in kg/s,
#: positive into the pressurizer.
#:
#: It is an INPUT, not a computed quantity, and that is deliberate.  [V] gives
#: the surge an equation (eq 32: sum over ten coolant nodes of V_j * theta_j *
#: dT_j/dt) -- but he also, in section 8.3.1, "applies a -1 kg/s perturbation"
#: to it as an external disturbance, and his level loop (eq 61, Table 2)
#: COMMANDS it as a controller output.  A quantity eq 32 calculates cannot also
#: be injected and commanded.  Treating it as an input is the reading that makes
#: his equations, his section 8 and his Fig. 11e mutually consistent: in section
#: 7 there are no controllers and no injected disturbance, so the channel sits
#: at zero -- which is exactly the dead-flat Fig. 11e.  See VALIDATION.md 2.1c.
#:
#: `Config(pzr_surge=True)` restores eq 32 as a computed term ADDED to this
#: input, which is the physically complete model and what [V]'s section 7.1-7.2
#: text describes.
#:
#: The channel also serves as CVCS charging/letdown, which enters the mass
#: balance identically; `mdot_ch` remains a valid alias for the same index.
U_NAMES = ["vrod_M", "vrod_AO", "utg", "iheat", "mdot_spr", "Pdem", "T_fw",
           "mdot_sur"]
IU = {n: i for i, n in enumerate(U_NAMES)}   # input name -> index
IU["mdot_ch"] = IU["mdot_sur"]               # backwards-compatible alias
ROD_IDX = (IU["vrod_M"], IU["vrod_AO"])      # the two accumulating channels


def build_layout(cfg: Config) -> Tuple[Dict[str, int], List[str]]:
    names: List[str] = []
    for h in (1, 2):
        names += [f"{s}_h{h}" for s in CORE_HALF]
    names += LOOP
    names += ["ps"]
    if cfg.pressurizer:
        names += ["pp", "lw"]
    names += ["Php_1", "Php_2"]
    if cfg.include_ip:
        names += ["Pip_1", "Pip_2"]
    names += ["Plp_1", "Plp_2", "Plp_3"]
    names += ["w_tur"]
    names += ["cv_1", "cv_2"]                 # normalized valve coeff + deriv
    names += ["Qheat"]
    names += ["rrod_1", "rrod_2"]
    if cfg.sensors:
        names += ["ilo"]
        if cfg.fast_log_amp:
            names += ["ilo_d"]
        names += ["ilr_1", "ilr_2", "Trtd1", "Trtd2"]
    if cfg.condenser:
        names += ["hwo", "mdot_cos"]
    return {n: i for i, n in enumerate(names)}, names


# ============================================================================
# 3.  THE PLANT
# ============================================================================

class Plant:
    """Nonlinear plant.  `deriv(t, x, u)` is the single entry point."""

    def __init__(self, p: Optional[Params] = None, cfg: Optional[Config] = None):
        self.p = p or Params()
        self.cfg = cfg or Config()
        self.ix, self.names = build_layout(self.cfg)
        self.n = len(self.names)
        self.m = len(U_NAMES)
        self.x0, self.u0 = self.nominal_point()

    # -- convenience -------------------------------------------------------
    def i(self, name: str) -> int:
        return self.ix[name]

    def get(self, x: np.ndarray, name: str) -> float:
        return float(x[self.ix[name]])

    # ------------------------------------------------------------------
    # 3.1  Core: two-point kinetics, [L] eqs 1-4 + coupling 9-10
    # ------------------------------------------------------------------
    def core_kinetics(self, Pr: Sequence[float], C: Sequence[np.ndarray],
                      rho: Sequence[float]) -> Tuple[List[float], List[np.ndarray]]:
        p = self.p
        F = p.F_couple if self.cfg.two_point else 0.0
        # Halves carry equal full power, so the Pf20/Pf10 ratios in [L] eqs
        # 9-10 are unity.  Keep them explicit so an asymmetric axial shape is
        # a one-line change.
        r12 = 1.0          # Pf20 / Pf10
        r21 = 1.0          # Pf10 / Pf20
        Pr12 = F * Pr[0] - F * r12 * Pr[1]      # leakage out of half 1
        Pr21 = F * Pr[1] - F * r21 * Pr[0]      # leakage out of half 2
        leak = (Pr12, Pr21)

        dPr, dC = [], []
        for k in (0, 1):
            src = float(np.sum(p.beta_i * C[k]) / p.Lam)
            dPr.append((rho[k] - p.beta) / p.Lam * Pr[k] + src - leak[k])
            dC.append(p.lam * (Pr[k] - C[k]))
        return dPr, dC

    # ------------------------------------------------------------------
    # 3.2  Core thermal: [V] eqs 5-7, instantiated once per half
    #      CP0b: Hf, Hc, tau_f, tau_c carry over unchanged;  tau_r halves.
    # ------------------------------------------------------------------
    def core_thermal(self, Pr: float, Tf: float, Tc1: float, Tc2: float,
                     T_in: float) -> Tuple[float, float, float]:
        p = self.p
        tr = p.tau_r_half
        q_fc = (Tf - Tc1) / p.tau_c
        dTf = p.Hf * Pr - (Tf - Tc1) / p.tau_f
        dTc1 = p.Hc * Pr + q_fc - 2.0 * (Tc1 - T_in) / tr
        dTc2 = p.Hc * Pr + q_fc - 2.0 * (Tc2 - Tc1) / tr
        return dTf, dTc1, dTc2

    # ------------------------------------------------------------------
    # 3.3  Iodine / xenon, [L] eqs 1 and 4, in normalized form
    #      dIr/dt  = lam_I (Pr - Ir)
    #      dXr/dt  = (lam_X + s0)/(gI+gX) * (gX*Pr + gI*Ir) - (lam_X + s0*Pr)*Xr
    #      with s0 = sigma_Xe * phi0.  Equilibrium at Pr=1 is Ir=Xr=1.
    #      The s0*Pr*Xr burnup term is a real nonlinearity; do not drop it.
    # ------------------------------------------------------------------
    def xenon(self, Pr: float, Ir: float, Xr: float) -> Tuple[float, float]:
        p = self.p
        if not self.cfg.xenon:
            return 0.0, 0.0
        s0 = p.sigma_X_phi0
        g = p.gamma_I + p.gamma_X
        # Guard: a diverging axial mode can drive one half's Pr below zero,
        # which would make the burnup term regenerate xenon and blow the
        # integrator up.  Clamp the flux, not the kinetics.
        Prp = max(Pr, 0.0)
        dI = p.lam_I * (Pr - Ir)
        dX = ((p.lam_X + s0) / g) * (p.gamma_X * Prp + p.gamma_I * Ir) \
             - (p.lam_X + s0 * Prp) * Xr
        return dI, dX

    def xenon_equilibrium(self, Pr: float) -> Tuple[float, float]:
        """Closed form.  Eliminate these before calling any root finder."""
        p = self.p
        if not self.cfg.xenon:
            return 0.0, 0.0
        s0 = p.sigma_X_phi0
        g = p.gamma_I + p.gamma_X
        Ir = Pr
        Xr = ((p.lam_X + s0) / g) * (p.gamma_X * Pr + p.gamma_I * Ir) \
             / (p.lam_X + s0 * Pr)
        return Ir, Xr

    # ------------------------------------------------------------------
    # 3.4  Reactivity, [L] eqs 3/6 merged with [V] eqs 43-44
    #
    #      Each half's rho is the LOCAL reactivity seen by that half's neutron
    #      population, so the full coefficients apply to local deviations.  If
    #      every temperature moves together, both halves see the same rho as
    #      [V]'s single-point model would.  That is the reduction test at CP0b.
    #
    #      NOTE the convention difference flagged in the scaffold: [V] eq 44
    #      puts the full alpha_c on EACH coolant node, [L] eq 3 puts alpha_C/2
    #      on each.  This follows [V] so the model reduces to his.  Flip
    #      `half_coolant_coeff` to follow [L].
    # ------------------------------------------------------------------
    half_coolant_coeff = False

    def reactivity(self, k: int, Tf: float, Tc1: float, Tc2: float,
                   Xr: float, pp: float, rrod: float,
                   ref: Dict[str, float]) -> float:
        p = self.p
        ac = p.alpha_c * (0.5 if self.half_coolant_coeff else 1.0)
        rho = rrod
        rho += p.alpha_f * (Tf - ref[f"Tf_h{k+1}"])
        rho += ac * (Tc1 - ref[f"Tc1_h{k+1}"])
        rho += ac * (Tc2 - ref[f"Tc2_h{k+1}"])
        rho += p.alpha_p * (pp - p.pp0)
        if self.cfg.xenon:
            rho += p.rho_xe_full * (Xr - ref[f"X_h{k+1}"])
        return rho

    # ------------------------------------------------------------------
    # 3.5  Outputs, [L] eqs 13-14
    # ------------------------------------------------------------------
    @staticmethod
    def outputs(Pr1: float, Pr2: float) -> Tuple[float, float]:
        """Return (normalized total power, axial power difference DI)."""
        Pr = 0.5 * (Pr1 + Pr2)
        DI = 0.5 * (Pr2 - Pr1)
        return Pr, DI

    # ------------------------------------------------------------------
    # 3.6  The derivative function
    # ------------------------------------------------------------------
    def deriv(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p, cfg, ix = self.p, self.cfg, self.ix
        dx = np.zeros_like(x)
        ref = self.ref_point

        # ---- unpack core -------------------------------------------------
        Pr = [x[ix["Pr_h1"]], x[ix["Pr_h2"]]]
        C = [np.array([x[ix[f"C{j}_h{k}"]] for j in range(1, 7)]) for k in (1, 2)]
        Ir = [x[ix["I_h1"]], x[ix["I_h2"]]]
        Xr = [x[ix["X_h1"]], x[ix["X_h2"]]]
        Tf = [x[ix["Tf_h1"]], x[ix["Tf_h2"]]]
        Tc1 = [x[ix["Tc1_h1"]], x[ix["Tc1_h2"]]]
        Tc2 = [x[ix["Tc2_h1"]], x[ix["Tc2_h2"]]]

        pp = x[ix["pp"]] if cfg.pressurizer else p.pp0
        ps = x[ix["ps"]]
        Trxi = x[ix["Trxi"]]

        # ---- inputs ------------------------------------------------------
        vrod = np.array([u[0], u[1]])         # (Mrod, AOrod), steps/min
        utg = u[2]
        iheat = u[3]
        mdot_spr = u[4]
        Pdem = u[5]
        T_fw = u[6]
        mdot_sur_in = u[7] if len(u) > 7 else 0.0  # commanded surge / charging, kg/s

        # ---- reactivity and kinetics -------------------------------------
        rho = [self.reactivity(k, Tf[k], Tc1[k], Tc2[k], Xr[k], pp,
                               x[ix[f"rrod_{k+1}"]], ref) for k in (0, 1)]
        dPr, dC = self.core_kinetics(Pr, C, rho)
        for k in (0, 1):
            dx[ix[f"Pr_h{k+1}"]] = dPr[k]
            for j in range(6):
                dx[ix[f"C{j+1}_h{k+1}"]] = dC[k][j]
            dI, dX = self.xenon(Pr[k], Ir[k], Xr[k])
            dx[ix[f"I_h{k+1}"]] = dI
            dx[ix[f"X_h{k+1}"]] = dX

        # ---- core thermal, halves in series ------------------------------
        T_in = [Trxi, Tc2[0]]                 # [L]: T2Cin = TC12
        for k in (0, 1):
            dTf, dTc1, dTc2 = self.core_thermal(Pr[k], Tf[k], Tc1[k], Tc2[k], T_in[k])
            dx[ix[f"Tf_h{k+1}"]] = dTf
            dx[ix[f"Tc1_h{k+1}"]] = dTc1
            dx[ix[f"Tc2_h{k+1}"]] = dTc2

        # ---- secondary saturation temperature, [V] eq 26 (deviation form)
        Ts = p.Ts0 + p.dTsat_dps * (ps - p.ps0)

        # ---- primary loop, [V] eqs 8-13 ----------------------------------
        # INTERFACE 1: core exit -> upper plenum.  [V]'s Tc2 becomes Tc2_h2.
        Trxu, Thot, Tsgi = x[ix["Trxu"]], x[ix["Thot"]], x[ix["Tsgi"]]
        Tp1, Tp2 = x[ix["Tp1"]], x[ix["Tp2"]]
        Tm1, Tm2 = x[ix["Tm1"]], x[ix["Tm2"]]
        Tsgu, Tcold = x[ix["Tsgu"]], x[ix["Tcold"]]

        dx[ix["Trxu"]] = (Tc2[1] - Trxu) / p.tau_rxu
        dx[ix["Thot"]] = (Trxu - Thot) / p.tau_hot
        dx[ix["Tsgi"]] = (Thot - Tsgi) / p.tau_sgi
        # ---- SG primary and metal, [V] eqs 14-17 -------------------------
        dx[ix["Tp1"]] = (Tsgi - Tp1) / p.tau_p1 - (Tp1 - Tm1) / p.tau_pm1
        dx[ix["Tp2"]] = (Tp1 - Tp2) / p.tau_p2 - (Tp2 - Tm2) / p.tau_pm2
        dx[ix["Tm1"]] = (Tp1 - Tm1) / p.tau_mp1 - (Tm1 - Ts) / p.tau_ms1
        dx[ix["Tm2"]] = (Tp2 - Tm2) / p.tau_mp2 - (Tm2 - Ts) / p.tau_ms2
        dx[ix["Tsgu"]] = (Tp2 - Tsgu) / p.tau_sgu
        dx[ix["Tcold"]] = (Tsgu - Tcold) / p.tau_cold
        # INTERFACE 2: lower plenum -> core inlet.  Trxi IS [L]'s T1Cin.
        dx[ix["Trxi"]] = (Tcold - Trxi) / p.tau_rxi

        # ---- governor valve, [V] eq 51 (normalized coefficient) ----------
        cv, cvd = x[ix["cv_1"]], x[ix["cv_2"]]
        wn, z = p.w_n_tg, p.zeta_tg
        dx[ix["cv_1"]] = cvd
        dx[ix["cv_2"]] = wn * wn * (utg - cv) - 2.0 * z * wn * cvd

        # ---- steam flow, [V] eq 25 (critical flow), and secondary, eq 24 --
        mdot_so = p.Ctg0 * cv * ps                          # kg/s
        q_sec = (p.UmsSms1 * (Tm1 - Ts) + p.UmsSms2 * (Tm2 - Ts))
        dx[ix["ps"]] = (q_sec - mdot_so * (p.h_ss - p.cp_fw * T_fw)) / p.Ks

        # ---- turbine, [V] eqs 34-37, observer canonical form -------------
        mn = mdot_so / p.mdot_sor                           # normalized flow
        thp, tip, tlp = p.tau_hp, p.tau_ip, p.tau_lp
        Orv, kap = p.O_rv, p.kappa_hp
        # HP, 2nd order with input derivative -> observer canonical form
        a1 = ((Orv * thp if HP_TAU else Orv) + tip) / (thp * tip)
        a0 = Orv / (thp * tip)
        b1 = (1.0 + kap) * p.F_hp / thp
        b0 = Orv * p.F_hp / (thp * tip)
        h1, h2 = x[ix["Php_1"]], x[ix["Php_2"]]
        dx[ix["Php_1"]] = -a1 * h1 + h2 + b1 * mn
        dx[ix["Php_2"]] = -a0 * h1 + b0 * mn
        Php = h1
        # IP, inert when F_ip = 0
        Pip = 0.0
        if cfg.include_ip:
            c1 = (Orv * thp + tip) / (thp * tip)
            c0 = Orv / (thp * tip)
            d0 = Orv * p.F_ip / (thp * tip)
            g1, g2 = x[ix["Pip_1"]], x[ix["Pip_2"]]
            dx[ix["Pip_1"]] = -c1 * g1 + g2
            dx[ix["Pip_2"]] = -c0 * g1 + d0 * mn
            Pip = g1
        # LP, 3rd order.  FLAG: the RHS of [V] eq 36 is printed without its
        # denominator; b0 below is the value that makes the DC gain F_lp.
        e2 = (Orv * thp + tip) / (thp * tip) + 1.0 / tlp
        e1 = (Orv * (tlp + thp) + tip) / (thp * tip * tlp)
        e0 = Orv / (thp * tip * tlp)
        f0 = Orv * p.F_lp / (thp * tip * tlp)
        l1, l2, l3 = x[ix["Plp_1"]], x[ix["Plp_2"]], x[ix["Plp_3"]]
        dx[ix["Plp_1"]] = -e2 * l1 + l2
        dx[ix["Plp_2"]] = -e1 * l1 + l3
        dx[ix["Plp_3"]] = -e0 * l1 + f0 * mn
        Plp = l1
        P_tur = Php + Pip + Plp

        # ---- turbine-generator inertia, [V] eq 38 ------------------------
        # A pure integrator unless P_tur == Pdem.  Nonlinear in w_tur.
        # Guard the 1/w singularity: a stiff solver probing the Jacobian can
        # push w through zero, and the resulting overflow kills the run with a
        # message that points nowhere near the cause.
        w = max(float(x[ix["w_tur"]]), 1.0)
        dx[ix["w_tur"]] = ((P_tur - Pdem) * p.eq38_power_base
                           / ((2.0 * np.pi) ** 2 * p.J_tur * w * p.I_tg))

        # ---- pressurizer heater, [V] eq 52 -------------------------------
        Q = x[ix["Qheat"]]
        dx[ix["Qheat"]] = (p.K_heat * iheat - Q / p.R_heat) / p.C_heat

        # ---- control rod drive, [V] eq 50 --------------------------------
        rates = p.G_dkk_per_step * p.rod_spm_to_per_s * (cfg.rod_map @ vrod)
        dx[ix["rrod_1"]] = rates[0]
        dx[ix["rrod_2"]] = rates[1]

        # ---- pressurizer, [V] eqs 27-33 ----------------------------------
        # Directed chain, NOT an implicit solve:
        #   (1) all temperature derivatives  (done above)
        #   (2) surge rate            eq 32
        #   (3) pressurizer pressure  eq 31
        #   (4) water level           eq 28
        if cfg.pressurizer:
            mdot_sur = ((self._surge_rate(dx) if cfg.pzr_surge else 0.0)
                        + mdot_sur_in)
            dpp = self._dpp_dt(x, Q, mdot_sur, mdot_spr)
            dx[ix["pp"]] = dpp
            dx[ix["lw"]] = self._dlw_dt(x, dpp, mdot_sur, mdot_spr)

        # ---- sensors, [V] eqs 45-49 --------------------------------------
        if cfg.sensors:
            Pn, _ = self.outputs(Pr[0], Pr[1])
            target = p.K_lo * np.log10(max(p.kappa_lo * Pn, 1e-12))
            if cfg.fast_log_amp:
                ilo, ilod = x[ix["ilo"]], x[ix["ilo_d"]]
                dx[ix["ilo"]] = ilod
                dx[ix["ilo_d"]] = (target - ilo - (p.tau_1 + p.tau_2) * ilod) \
                                  / (p.tau_1 * p.tau_2)
                dilo = ilod
            else:
                ilo = x[ix["ilo"]]
                dilo = (target - ilo) / p.tau_2
                dx[ix["ilo"]] = dilo
            r1, r2 = x[ix["ilr_1"]], x[ix["ilr_2"]]
            dx[ix["ilr_1"]] = r2
            dx[ix["ilr_2"]] = (p.K_lr * dilo - (r1 - 12.0)
                               - (p.tau_3 + p.tau_4) * r2) / (p.tau_3 * p.tau_4)
            # CP0b: with four half-lumps in series, the single-core model's
            # node 1 and node 2 sit exactly at Tc2_h1 and Tc2_h2.
            Tc1_eq, Tc2_eq = Tc2[0], Tc2[1]
            dx[ix["Trtd1"]] = (-x[ix["Trtd1"]] + 2.0 * Tc1_eq - Trxi) / p.tau_rtd
            dx[ix["Trtd2"]] = (-x[ix["Trtd2"]] + 2.0 * Tc2_eq - Trxu) / p.tau_rtd

        # ---- condenser, [V] eqs 39-42 (passive appendage) ----------------
        if cfg.condenser:
            hwo = x[ix["hwo"]]
            mdot_lp = mdot_so
            mdot_cow = mdot_lp * (p.h_lp - p.h_cow) / p.h_cow
            mdot_coh = mdot_lp - mdot_cow
            dx[ix["hwo"]] = (mdot_coh + mdot_cow) * (p.h_cow - hwo) / p.m_coh
            dx[ix["mdot_cos"]] = (mdot_cow - x[ix["mdot_cos"]]) / p.tau_co

        return dx

    # -- pressurizer internals --------------------------------------------
    def _surge_rate(self, dx: np.ndarray) -> float:
        """[V] eq 32.  Four core coolant nodes at half volume each (CP0b)."""
        p, ix = self.p, self.ix
        V = p.Vtheta
        core_total = V[1] + V[2]
        vq = core_total / 4.0
        s = 0.0
        s += V[0] * dx[ix["Trxi"]]                        # lower plenum
        for k in (1, 2):                                  # four core nodes
            s += vq * dx[ix[f"Tc1_h{k}"]]
            s += vq * dx[ix[f"Tc2_h{k}"]]
        s += V[3] * dx[ix["Trxu"]]
        s += V[4] * dx[ix["Thot"]]
        s += V[5] * dx[ix["Tsgi"]]
        s += V[6] * dx[ix["Tp1"]]
        s += V[7] * dx[ix["Tp2"]]
        s += V[8] * dx[ix["Tsgu"]]
        s += V[9] * dx[ix["Tcold"]]
        return s

    def _pzr_geometry(self, lw: float) -> Tuple[float, float, float, float, float]:
        p = self.p
        C1p = p.d_w / p.d_s - 1.0
        C2p = p.Ap * (p.l_pzr - lw) * (p.d_w / p.d_s) * p.K2p + p.Ap * lw * p.K1p
        mw = p.d_w * p.Ap * lw
        ms = p.d_s * p.Ap * (p.l_pzr - lw)
        Vw = mw * p.nu_w
        return C1p, C2p, mw, ms, Vw

    def _dpp_dt(self, x: np.ndarray, Q: float, msur: float, mspr: float) -> float:
        """[V] eq 31.  At equilibrium Q = msur = mspr = 0 gives dpp/dt = 0
        regardless of the denominator, which is why a transcription error in
        the thermodynamic partials shows up as a wrong *rate*, never as drift
        at the trim point.  Watch CP6 Plot A."""
        p, ix = self.p, self.ix
        pp, lw = x[ix["pp"]], x[ix["lw"]]
        C1p, C2p, mw, ms, Vw = self._pzr_geometry(lw)
        num = (Q
               + msur * (pp * p.nu_s / (p.Jp * C1p) + p.h_w / C1p)
               + mspr * (p.h_spr - p.h_w + p.h_w / C1p + pp * p.nu_w / (p.Jp * C1p)))
        den = (mw * (p.K3p + p.K4p * pp / p.Jp)
               + ms * p.K4p * pp / p.Jp
               - Vw / p.Jp
               + (C2p / C1p) * (p.h_w + pp * p.nu_s / p.Jp))
        if abs(den) < 1e-9:
            return 0.0
        return num / den

    def _dlw_dt(self, x: np.ndarray, dpp: float, msur: float, mspr: float) -> float:
        """[V] eq 28.  No restoring term: a pure integrator (see CP7)."""
        p, ix = self.p, self.ix
        lw = x[ix["lw"]]
        C1p, C2p, mw, ms, Vw = self._pzr_geometry(lw)
        t1 = (p.Ap * (p.l_pzr - lw) * p.K2p - C2p / C1p) * dpp
        t2 = (C2p * dpp - msur - mspr) / (C1p ** 2)
        t3 = msur / C1p
        return (t1 + t2 + t3) / (p.d_s * p.Ap)

    # ------------------------------------------------------------------
    # 3.7  Nominal (100 %, symmetric, equilibrium-xenon) operating point
    # ------------------------------------------------------------------
    def nominal_point(self) -> Tuple[np.ndarray, np.ndarray]:
        p, ix = self.p, self.ix
        x = np.zeros(len(self.names))
        ref: Dict[str, float] = {}

        # Core, analytic.  At Pr = 1:  Tf - Tc1 = Hf * tau_f,  and each of the
        # four half-lumps rises by the same amount.
        dTf_c1 = p.Hf * p.tau_f
        q = p.Hc + dTf_c1 / p.tau_c
        rise = q * p.tau_r_half / 2.0
        Tc = [p.Trxi0 + rise * (j + 1) for j in range(4)]   # 4 lumps in series
        Ir, Xr = self.xenon_equilibrium(1.0)
        for k, (c1, c2) in enumerate(((Tc[0], Tc[1]), (Tc[2], Tc[3])), start=1):
            x[ix[f"Pr_h{k}"]] = 1.0
            for j in range(1, 7):
                x[ix[f"C{j}_h{k}"]] = 1.0
            x[ix[f"I_h{k}"]] = Ir
            x[ix[f"X_h{k}"]] = Xr
            x[ix[f"Tc1_h{k}"]] = c1
            x[ix[f"Tc2_h{k}"]] = c2
            x[ix[f"Tf_h{k}"]] = c1 + dTf_c1
            ref[f"Tf_h{k}"] = c1 + dTf_c1
            ref[f"Tc1_h{k}"] = c1
            ref[f"Tc2_h{k}"] = c2
            ref[f"X_h{k}"] = Xr
        self.ref_point = ref

        # Loop: the tabulated values, which are mutually consistent
        for nm, val in (("Trxu", p.Trxu0), ("Thot", p.Thot0), ("Tsgi", p.Tsgi0),
                        ("Tp1", p.Tp10), ("Tp2", p.Tp20), ("Tm1", p.Tm10),
                        ("Tm2", p.Tm20), ("Tsgu", p.Tsgu0), ("Tcold", p.Tcold0),
                        ("Trxi", p.Trxi0)):
            x[ix[nm]] = val
        x[ix["ps"]] = p.ps0
        if self.cfg.pressurizer:
            x[ix["pp"]] = p.pp0
            x[ix["lw"]] = p.lw0

        # Valve normalized to 1 at rated flow
        x[ix["cv_1"]] = 1.0
        x[ix["cv_2"]] = 0.0

        # Turbine, observer canonical steady state at unit normalized flow
        thp, tip, tlp, Orv, kap = p.tau_hp, p.tau_ip, p.tau_lp, p.O_rv, p.kappa_hp
        a1 = ((Orv * thp if HP_TAU else Orv) + tip) / (thp * tip)
        b1 = (1.0 + kap) * p.F_hp / thp
        x[ix["Php_1"]] = p.F_hp
        x[ix["Php_2"]] = a1 * p.F_hp - b1 * 1.0
        if self.cfg.include_ip:
            c1 = (Orv * thp + tip) / (thp * tip)
            x[ix["Pip_1"]] = p.F_ip
            x[ix["Pip_2"]] = c1 * p.F_ip
        e2 = (Orv * thp + tip) / (thp * tip) + 1.0 / tlp
        e1 = (Orv * (tlp + thp) + tip) / (thp * tip * tlp)
        x[ix["Plp_1"]] = p.F_lp
        x[ix["Plp_2"]] = e2 * p.F_lp
        x[ix["Plp_3"]] = e1 * p.F_lp
        x[ix["w_tur"]] = p.w_tur0
        x[ix["Qheat"]] = 0.0
        x[ix["rrod_1"]] = 0.0
        x[ix["rrod_2"]] = 0.0

        if self.cfg.sensors:
            x[ix["ilo"]] = p.K_lo * np.log10(p.kappa_lo * 1.0)
            if self.cfg.fast_log_amp:
                x[ix["ilo_d"]] = 0.0
            x[ix["ilr_1"]] = 12.0
            x[ix["ilr_2"]] = 0.0
            x[ix["Trtd1"]] = 2.0 * Tc[1] - p.Trxi0
            x[ix["Trtd2"]] = 2.0 * Tc[3] - p.Trxu0
        if self.cfg.condenser:
            x[ix["hwo"]] = p.h_cow
            x[ix["mdot_cos"]] = 0.0

        # Inputs: vrod = 0, valve at 1, no heater, no spray, demand = rated
        u = np.array([0.0, 0.0, 1.0, 0.0, 0.0, p.F_hp + p.F_ip + p.F_lp,
                      p.T_fw0, 0.0])
        return x, u

    # ------------------------------------------------------------------
    # 3.8  Derived signals for plotting
    # ------------------------------------------------------------------
    def signals(self, x: np.ndarray,
                u: Optional[np.ndarray] = None) -> Dict[str, float]:
        # `u` is accepted for backwards compatibility and is not read: every
        # signal below is a function of the state alone.  That is what lets a
        # controller call plant.signals(x) without an algebraic loop.
        p, ix = self.p, self.ix
        Pr1, Pr2 = x[ix["Pr_h1"]], x[ix["Pr_h2"]]
        Pn, DI = self.outputs(Pr1, Pr2)
        cv, ps = x[ix["cv_1"]], x[ix["ps"]]
        mdot_so = p.Ctg0 * cv * ps
        P_tur = x[ix["Php_1"]] + x[ix["Plp_1"]]
        if self.cfg.include_ip:
            P_tur += x[ix["Pip_1"]]
        Tavg = 0.25 * (x[ix["Tc1_h1"]] + x[ix["Tc2_h1"]]
                       + x[ix["Tc1_h2"]] + x[ix["Tc2_h2"]])
        out = dict(Pn=Pn, DI=DI, Pr1=Pr1, Pr2=Pr2, mdot_so=mdot_so,
                   P_tur=P_tur, Tavg=Tavg, ps=ps,
                   Ts=p.Ts0 + p.dTsat_dps * (ps - p.ps0),
                   Tcold=x[ix["Tcold"]], Thot=x[ix["Thot"]],
                   P_th=Pn * p.Pth0, w_tur=x[ix["w_tur"]])
        if self.cfg.pressurizer:
            out["pp"] = x[ix["pp"]]
            out["lw"] = x[ix["lw"]]
        if self.cfg.sensors:
            out["ilo"] = x[ix["ilo"]]
            Trtd = 0.5 * (x[ix["Trtd1"]] + x[ix["Trtd2"]])
            out["irtd"] = p.K_rtd * (Trtd - p.Trxi0) / (p.Trxu0 - p.Trxi0) + 4.0
        return out

    # ------------------------------------------------------------------
    # 3.9  Integration helper (stiff by construction, so BDF/Radau only)
    # ------------------------------------------------------------------
    def simulate(self, x0: np.ndarray, t_end: float,
                 u_of_t: Callable[[float, np.ndarray], np.ndarray],
                 n_out: int = 2000, method: str = "BDF",
                 rtol: float = 1e-8, atol: float = 1e-10):
        from scipy.integrate import solve_ivp
        f = lambda t, x: self.deriv(t, x, u_of_t(t, x))
        t_eval = np.linspace(0.0, t_end, n_out)
        sol = solve_ivp(f, (0.0, t_end), x0, method=method, t_eval=t_eval,
                        rtol=rtol, atol=atol, max_step=t_end / 20.0)
        if not sol.success:
            print(f"  ! integration warning: {sol.message}")
        return sol


# ============================================================================
# 4.  EQUILIBRIUM TRIM                                                  (CP7)
# ============================================================================
#
# Three techniques, per the scaffold:
#   1. eliminate the analytic states (precursors, iodine, xenon) first;
#   2. know which integrators survive and handle each separately;
#   3. use continuation from 100 % downward -- never a cold solve.
#
# Surviving integrators and how each is handled here:
#   rho_rod      [V] eq 50   -> algebraic free variable, vrod held at 0
#   turbine speed[V] eq 38   -> Pdem imposed equal to P_tur, not an input
#   pzr level    [V] eq 28   -> free; settles wherever msur = mspr = 0 puts it

TRIM_UNKNOWNS = ["Tf_h1", "Tc1_h1", "Tc2_h1", "Tf_h2", "Tc1_h2", "Tc2_h2",
                 "Trxu", "Thot", "Tsgi", "Tp1", "Tp2", "Tm1", "Tm2",
                 "Tsgu", "Tcold", "Trxi", "cv_1", "rrod_1", "rrod_2"]


def trim(plant: Plant, power: float, di_target: float = 0.0,
         ps_target: Optional[float] = None,
         guess: Optional[np.ndarray] = None,
         verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve for an equilibrium at the given normalized power.

    Returns (x0, u0, residual_norm).  19 unknowns against 19 residuals:
    2 kinetics + 6 core thermal + 10 loop + 1 secondary pressure.  Power and
    DI are imposed directly on Pr1, Pr2 rather than solved for, and ps is
    pinned at its programmed value so the valve coefficient is what floats.
    """
    from scipy.optimize import root

    p, ix = plant.p, plant.ix
    ps_t = p.ps0 if ps_target is None else ps_target
    Pr1 = power - di_target
    Pr2 = power + di_target
    Ir1, Xr1 = plant.xenon_equilibrium(Pr1)
    Ir2, Xr2 = plant.xenon_equilibrium(Pr2)

    x = plant.x0.copy()
    u = plant.u0.copy()
    x[ix["Pr_h1"]], x[ix["Pr_h2"]] = Pr1, Pr2
    for j in range(1, 7):
        x[ix[f"C{j}_h1"]] = Pr1
        x[ix[f"C{j}_h2"]] = Pr2
    x[ix["I_h1"]], x[ix["X_h1"]] = Ir1, Xr1
    x[ix["I_h2"]], x[ix["X_h2"]] = Ir2, Xr2
    x[ix["ps"]] = ps_t
    x[ix["Qheat"]] = 0.0
    u[0] = u[1] = 0.0                     # vrod: rods parked
    u[3] = 0.0                            # heater off
    u[4] = 0.0                            # no spray
    u[6] = p.T_fw0
    u[7] = 0.0                            # no charging flow

    idx = [ix[n] for n in TRIM_UNKNOWNS]
    resid_names = (["Pr_h1", "Pr_h2"]
                   + ["Tf_h1", "Tc1_h1", "Tc2_h1", "Tf_h2", "Tc1_h2", "Tc2_h2"]
                   + LOOP + ["ps"])
    ridx = [ix[n] for n in resid_names]

    def residual(z: np.ndarray) -> np.ndarray:
        xx = x.copy()
        xx[idx] = z
        # turbine states follow the valve algebraically
        mn = p.Ctg0 * xx[ix["cv_1"]] * xx[ix["ps"]] / p.mdot_sor
        thp, tip, tlp = p.tau_hp, p.tau_ip, p.tau_lp
        Orv, kap = p.O_rv, p.kappa_hp
        a1 = ((Orv * thp if HP_TAU else Orv) + tip) / (thp * tip)
        b1 = (1.0 + kap) * p.F_hp / thp
        xx[ix["Php_1"]] = p.F_hp * mn
        xx[ix["Php_2"]] = a1 * p.F_hp * mn - b1 * mn
        e2 = (Orv * thp + tip) / (thp * tip) + 1.0 / tlp
        e1 = (Orv * (tlp + thp) + tip) / (thp * tip * tlp)
        xx[ix["Plp_1"]] = p.F_lp * mn
        xx[ix["Plp_2"]] = e2 * p.F_lp * mn
        xx[ix["Plp_3"]] = e1 * p.F_lp * mn
        uu = u.copy()
        uu[2] = xx[ix["cv_1"]]            # valve command matches position
        uu[5] = (p.F_hp + p.F_ip + p.F_lp) * mn   # Pdem = P_tur, kills the
                                                  # speed integrator
        d = plant.deriv(0.0, xx, uu)
        return d[ridx]

    z0 = (guess if guess is not None else x[idx]).copy()
    sol = root(residual, z0, method="hybr", tol=1e-12,
               options=dict(maxfev=20000, xtol=1e-13))
    x[idx] = sol.x
    mn = p.Ctg0 * x[ix["cv_1"]] * x[ix["ps"]] / p.mdot_sor
    residual(sol.x)                        # refresh turbine states in x
    thp, tip, tlp = p.tau_hp, p.tau_ip, p.tau_lp
    Orv, kap = p.O_rv, p.kappa_hp
    a1 = ((Orv * thp if HP_TAU else Orv) + tip) / (thp * tip)
    b1 = (1.0 + kap) * p.F_hp / thp
    x[ix["Php_1"]] = p.F_hp * mn
    x[ix["Php_2"]] = a1 * p.F_hp * mn - b1 * mn
    e2 = (Orv * thp + tip) / (thp * tip) + 1.0 / tlp
    e1 = (Orv * (tlp + thp) + tip) / (thp * tip * tlp)
    x[ix["Plp_1"]] = p.F_lp * mn
    x[ix["Plp_2"]] = e2 * p.F_lp * mn
    x[ix["Plp_3"]] = e1 * p.F_lp * mn
    u[2] = x[ix["cv_1"]]
    u[5] = (p.F_hp + p.F_ip + p.F_lp) * mn
    if plant.cfg.sensors:
        Pn, _ = plant.outputs(Pr1, Pr2)
        x[ix["ilo"]] = p.K_lo * np.log10(p.kappa_lo * Pn)
        x[ix["Trtd1"]] = 2.0 * x[ix["Tc2_h1"]] - x[ix["Trxi"]]
        x[ix["Trtd2"]] = 2.0 * x[ix["Tc2_h2"]] - x[ix["Trxu"]]

    res = float(np.linalg.norm(plant.deriv(0.0, x, u)[ridx]))
    if verbose:
        print(f"  trim {power:5.2f}  ok={sol.success}  |f|={res:.3e}  "
              f"cv={x[ix['cv_1']]:.4f}  rrod=({x[ix['rrod_1']]:+.2e},"
              f"{x[ix['rrod_2']]:+.2e})")
    return x, u, res


def trim_sweep(plant: Plant, powers: Sequence[float], di_target: float = 0.0,
               verbose: bool = True):
    """Continuation from the top down.  A cold solve at 30 % will not converge."""
    out = []
    guess = None
    for pw in powers:
        x, u, r = trim(plant, pw, di_target=di_target, guess=guess, verbose=verbose)
        guess = x[[plant.ix[n] for n in TRIM_UNKNOWNS]].copy()
        out.append((pw, x, u, r))
    return out


# ============================================================================
# 5.  LINEARIZATION                                                  (CP8/CP9)
# ============================================================================

OUTPUT_NAMES = ["Pn", "DI", "Tavg", "ps", "pp", "P_tur", "w_tur", "lw"]


def typical_scales(plant: Plant) -> np.ndarray:
    """Hand-set typical magnitudes.  States span many decades; without this the
    A matrix is badly conditioned and its eigenvalues are unreliable."""
    s = np.ones(plant.n)
    for k, nm in enumerate(plant.names):
        if nm.startswith(("Tf", "Tc", "Trxu", "Thot", "Tsg", "Tp", "Tm",
                          "Tcold", "Trxi", "Trtd")):
            s[k] = 300.0
        elif nm == "ps":
            s[k] = 7.28
        elif nm == "pp":
            s[k] = 15.41
        elif nm == "lw":
            s[k] = 8.55
        elif nm == "w_tur":
            s[k] = 60.0
        elif nm.startswith("rrod"):
            s[k] = 1.0e-3
        elif nm == "Qheat":
            s[k] = 1.0e5
        elif nm.startswith("ilo"):
            s[k] = 20.0
        elif nm.startswith("ilr"):
            s[k] = 12.0
    return s


def linearize(plant: Plant, x0: np.ndarray, u0: np.ndarray,
              scaled: bool = True):
    """Central finite differences with RELATIVE perturbations.

    Returns (A, B, C, D, S) where S is the diagonal scaling actually applied:
    A_scaled = S^-1 A S,  B_scaled = S^-1 B,  C_scaled = C S.
    """
    n, m = plant.n, plant.m
    xt = typical_scales(plant)
    eps = np.sqrt(np.finfo(float).eps)

    def y_of(x, u):
        sg = plant.signals(x, u)
        return np.array([sg.get(k, 0.0) for k in OUTPUT_NAMES])

    A = np.zeros((n, n))
    C = np.zeros((len(OUTPUT_NAMES), n))
    for j in range(n):
        d = eps * max(abs(x0[j]), xt[j])
        xp, xm = x0.copy(), x0.copy()
        xp[j] += d
        xm[j] -= d
        A[:, j] = (plant.deriv(0.0, xp, u0) - plant.deriv(0.0, xm, u0)) / (2 * d)
        C[:, j] = (y_of(xp, u0) - y_of(xm, u0)) / (2 * d)

    ut = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 300.0, 1.0])
    B = np.zeros((n, m))
    D = np.zeros((len(OUTPUT_NAMES), m))
    for j in range(m):
        d = eps * max(abs(u0[j]), ut[j])
        up, um = u0.copy(), u0.copy()
        up[j] += d
        um[j] -= d
        B[:, j] = (plant.deriv(0.0, x0, up) - plant.deriv(0.0, x0, um)) / (2 * d)
        D[:, j] = (y_of(x0, up) - y_of(x0, um)) / (2 * d)

    if not scaled:
        return A, B, C, D, np.ones(n)
    S = xt
    A = (A * S[None, :]) / S[:, None]
    B = B / S[:, None]
    C = C * S[None, :]
    return A, B, C, D, S


def dc_gain(A, B, C, D):
    """Steady-state gain.  Integrators make A singular, so use a pseudo-inverse
    and read the result as 'gain of the non-integrating part'."""
    try:
        return -C @ np.linalg.pinv(A) @ B + D
    except np.linalg.LinAlgError:
        return np.full((C.shape[0], B.shape[1]), np.nan)


def freq_response(A, B, C, D, w, iu, iy):
    n = A.shape[0]
    I = np.eye(n)
    g = np.zeros(len(w), dtype=complex)
    for k, wk in enumerate(w):
        try:
            g[k] = (C @ np.linalg.solve(1j * wk * I - A, B[:, iu]) + D[:, iu])[iy]
        except np.linalg.LinAlgError:
            g[k] = np.nan
    return g


# ============================================================================
# 6.  CONSISTENCY REPORT                                                (CP0)
# ============================================================================

def consistency_report(plant: Plant) -> None:
    p = plant.p
    print("\n--- transcription / consistency audit -------------------------")
    dTf = p.Hf * p.tau_f
    print(f"  Tf - Tc1 at 100%  = Hf*tau_f = {dTf:8.2f} degC")
    print(f"    tabulated Tf0 - Tc10        = {p.Tf0 - p.Tc10:8.2f} degC   "
          f"(match confirms Hf, tau_f, and the Mann form)")
    q = p.Hc + dTf / p.tau_c
    rise = q * p.tau_r / 2.0
    print(f"  per-node coolant rise (single core) = {rise:6.2f} degC; "
          f"tabulated Tc10-Trxi0 = {p.Tc10 - p.Trxi0:.2f}")
    print(f"  core thermal duty from SG balance   = {p.Pth0/1e9:6.3f} GWth "
          f"(plant is 1.2 GWe -> eta = {1.2e9/p.Pth0:.3f})")
    print(f"  Ctg0 required by that balance       = {p.Ctg0:8.3f} kg/(s MPa)")
    print( "    FLAG: [V] Table 1 prints Ctg = 2.0481, which gives "
          f"{2.0481*p.ps0:.2f} kg/s against a rated {p.mdot_sor:.0f} kg/s.")
    print( "    Resolved here by the energy balance; the tabulated value is unused.")
    ts_lit = p.dTsat_dps * p.ps0
    print(f"  [V] eq 26 taken literally: Ts = 9.47*ps = {ts_lit:.2f} degC "
          f"vs tabulated Ts0 = {p.Ts0:.2f}")
    print( "    FLAG: implemented in deviation form, Ts = Ts0 + 9.47*(ps - ps0).")
    print(f"  pressurizer: Ap*lw = {p.Ap*p.lw0:.2f} m^3 vs tabulated Vw = 30.4988")
    print( "    FLAG: [V] Table 1 also lists lw0 = 28.06 m, which exceeds "
          f"l = {p.l_pzr} m; lw0 = {p.lw0} m is used.")
    print(f"  ex-core amplifier at Pn=1: ilo = {p.K_lo*np.log10(p.kappa_lo):.2f} "
          f"vs tabulated {p.ilo0}")
    print(f"  F_ip = {p.F_ip}  -> IP turbine branch carries no power (correct)")
    print(f"  F_couple = {p.F_couple} 1/s is NOT from either paper; "
          "calibrate at CP2")
    print(f"  rho_xe_full = {p.rho_xe_full} dk/k is NOT from either paper")
    C1p = p.d_w / p.d_s - 1.0
    C2p = (p.Ap * (p.l_pzr - p.lw0) * (p.d_w / p.d_s) * p.K2p
           + p.Ap * p.lw0 * p.K1p)
    print(f"  pressurizer sensitivity dpp/dt ~ 1/C2p = {1.0/C2p:.4f} MPa per kg/s "
          "of surge")
    print(f"    (pzr_partial_scale = {p.pzr_partial_scale:g}; with the printed "
          f"partials it is {p.pzr_partial_scale/C2p:.1f} MPa/(kg/s), which")
    print( "     runs the pressurizer away in ~20 s and shuts the core down "
          "through alpha_p -- see CP6)")
    print("---------------------------------------------------------------\n")



# ============================================================================
# 7.  SIMULATION HELPERS
# ============================================================================
#
# Low-level escape hatches, for when ClosedLoop is in the way.  The
# checkpoint suite that used to live here is now pwr_validation.py.

def freeze(plant: Plant, keep: Sequence[str]):
    """Return a deriv wrapper that zeroes every state not in `keep`.

    Used by CP1-CP5 to exercise one subsystem at a time.  Names are matched by
    prefix, so 'Tc' keeps every coolant node.
    """
    mask = np.zeros(plant.n)
    for k, nm in enumerate(plant.names):
        if any(nm.startswith(pref) for pref in keep):
            mask[k] = 1.0

    def d(t, x, u):
        return plant.deriv(t, x, u) * mask
    return d


# NOTE: "rrod" must be in every keep-list that uses a rod controller, or the
# rod-drive integrator ([V] eq 50) is frozen and vrod silently does nothing.


def run(dfun, x0, t_end, u_of_t, n_out=1500, method="BDF",
        rtol=1e-8, atol=1e-10):
    from scipy.integrate import solve_ivp
    f = lambda t, x: dfun(t, x, u_of_t(t, x))
    sol = solve_ivp(f, (0.0, t_end), x0, method=method,
                    t_eval=np.linspace(0.0, t_end, n_out),
                    rtol=rtol, atol=atol)
    if not sol.success:
        print(f"  ! integrator: {sol.message}")
    return sol


def const_u(u):
    return lambda t, x: u

# ============================================================================
# 8.  CONTROL TESTBED
# ============================================================================
#
# Everything below is the black-box layer.  Sections 1-8 are the physics and
# do not need to be read to use it.
#
#     plant   = Plant()
#     loop    = turbine_leading(plant, demand=ramp(1.0, 0.5, ...))
#     result  = loop.run(6 * 3600)
#     result.df.plot(x="t_h", y=["Pn", "P_tur", "Tavg"])
#
# A Controller reads plant signals and writes actuator commands.  If it needs
# integral action it declares `states`, and those states are appended to the
# plant's state vector and integrated by the same stiff solver -- which is the
# only way to get a correct integrator through an adaptive method that rejects
# and re-takes steps.  Do NOT accumulate in a closure.

EMPTY = np.zeros(0)


@dataclass
class LoopContext:
    """Everything a controller may need besides the live signals.

    CHANGE 1 (v2): `x` is the live state vector, refreshed by ClosedLoop
    before every controller call, so a controller can do full-state feedback:

        ctx.x                      the whole 53-element vector
        ctx.state("Tc2_h2")        one entry, by name

    Treat `x` as read-only.  Writing to it writes to the solver's working
    array and will corrupt the integration.
    """
    plant: "Plant"
    x0: np.ndarray                 # trim state
    u0: np.ndarray                 # trim input
    sig0: Dict[str, float]         # trim signals
    demand: Callable[[float], float]
    x: Optional[np.ndarray] = None       # live state -- set by ClosedLoop

    def state(self, name: str) -> float:
        """One state by name, e.g. ctx.state("rrod_1")."""
        return float(self.x[self.plant.ix[name]])

    def states(self, *names: str) -> np.ndarray:
        """Several states by name, as an array."""
        return np.array([self.x[self.plant.ix[n]] for n in names])


class Controller:
    """Base class.  Override `update`; declare `states` only if you integrate.

    update(t, sig, xc, u, ctx) -> dxc
        t    simulation time, s
        sig  dict of plant signals at this instant (see Plant.signals)
        xc   this controller's own states, shape (len(self.states),)
        u    the input vector, MUTATED IN PLACE
        ctx  LoopContext
        returns the derivative of xc

    Rod commands ACCUMULATE: write `u[IU["vrod_M"]] += v`, never `=`.  That is
    what lets a power/temperature loop and an axial-offset loop share the two
    rod groups.  Every other actuator is written by exactly one controller.

    CONTINUOUS (Ts = None, the default)
        update() is called by the ODE solver whenever it wants a derivative --
        several times per step, and again on steps it later rejects.  So `t`
        is not monotonic, and you must NOT keep memory in Python attributes.
        Return the DERIVATIVE of xc.

    SAMPLED (Ts = a number, in seconds)                        [CHANGE 2, v2]
        update() is called exactly once per sample instant, in order, with t
        increasing.  Outputs are held constant until the next sample.  Return
        the NEXT VALUE of xc, not its derivative.  Because the call pattern is
        now predictable, ordinary Python attributes work as memory:

            self.u_prev, a ring buffer of past measurements, a covariance
            matrix, a mode flag, anything.

        This is what makes MPC, discrete observers, sign() switching, rate
        limits, latching logic and injected measurement noise possible.
    """
    name: str = "controller"
    states: Tuple[str, ...] = ()
    Ts: Optional[float] = None      # None -> continuous; a number -> sampled

    def initialize(self, ctx: LoopContext) -> np.ndarray:
        return np.zeros(len(self.states))

    def update(self, t: float, sig: Dict[str, float], xc: np.ndarray,
               u: np.ndarray, ctx: LoopContext) -> np.ndarray:
        return EMPTY


# ---------------------------------------------------------------------------
# Load profiles
# ---------------------------------------------------------------------------

def ramp(start: float = 1.0, target: float = 0.5, t_start: float = 3600.0,
         ramp_time: float = 7200.0, hold: float = 21600.0,
         t_return: Optional[float] = None) -> Callable[[float], float]:
    """Down-ramp, hold, up-ramp.  Fractional full power, not MW."""
    t_ret = (t_start + ramp_time + hold) if t_return is None else t_return

    def f(t: float) -> float:
        if t < t_start:
            return start
        if t < t_start + ramp_time:
            return start + (target - start) * (t - t_start) / ramp_time
        if t < t_ret:
            return target
        if t < t_ret + ramp_time:
            return target + (start - target) * (t - t_ret) / ramp_time
        return start
    return f


def step(start: float = 1.0, target: float = 0.9,
         t_step: float = 1800.0) -> Callable[[float], float]:
    return lambda t: start if t < t_step else target


def tavg_program(plant: "Plant", full_power_tavg: Optional[float] = None
                 ) -> Callable[[float], float]:
    """Programmed average coolant temperature against load.

    Derived, not assumed.  Trim this model at constant steam pressure and Tavg
    comes out exactly linear in load, extrapolating to Ts0 = 288.06 degC at
    zero load -- i.e. at no load the primary sits at secondary saturation
    temperature, which is the right answer and a good check that the loop
    model is consistent.  Span is about 27.9 degC over the full range.
    """
    p = plant.p
    hot = plant.signals(plant.x0)["Tavg"] if full_power_tavg is None \
        else full_power_tavg
    return lambda load: p.Ts0 + (hot - p.Ts0) * float(np.clip(load, 0.0, 1.2))


# ---------------------------------------------------------------------------
# Stock controllers
# ---------------------------------------------------------------------------

class GridDemand(Controller):
    """Turbine-generator load demand follows the profile.  Feedforward."""
    name = "grid_demand"

    def update(self, t, sig, xc, u, ctx):
        u[IU["Pdem"]] = ctx.u0[IU["Pdem"]] * ctx.demand(t)
        return EMPTY


class ValveOnDemand(Controller):
    """Governor valve follows load demand directly.  THIS is what makes a run
    turbine-leading: the secondary moves first and the reactor follows."""
    name = "valve_on_demand"

    def update(self, t, sig, xc, u, ctx):
        u[IU["utg"]] = ctx.u0[IU["utg"]] * ctx.demand(t)
        return EMPTY


class SGPressurePI(Controller):
    """Valve holds steam pressure.  Used in REACTOR-leading mode, where the
    rods set power and the valve extracts whatever the core produces.

    Sign note: a valve step of +0.05 moves ps by -0.144 MPa, so negative
    feedback needs u = +k*(ps - ref).  [V] eq 58 prints (ps0 - ps), which with
    a positive gain is positive feedback; his tabulated magnitudes are used
    here with the sign that is actually stable.
    """
    name = "sg_pressure"
    states = ("I_ps",)

    def __init__(self, kp: float = 0.537, ki: float = 0.117,
                 ref: Optional[float] = None, u_min: float = 0.0,
                 u_max: float = 1.5, t_track: float = 10.0):
        self.kp, self.ki, self.ref = kp, ki, ref
        self.u_min, self.u_max, self.t_track = u_min, u_max, t_track

    def update(self, t, sig, xc, u, ctx):
        ref = ctx.sig0["ps"] if self.ref is None else self.ref
        e = sig["ps"] - ref
        raw = ctx.u0[IU["utg"]] + self.kp * e + self.ki * xc[0]
        cmd = float(np.clip(raw, self.u_min, self.u_max))
        u[IU["utg"]] = cmd
        # Back-calculation anti-windup.  A hard "stop integrating when
        # saturated" switch is discontinuous and makes the stiff solver crawl;
        # this bleeds the integrator smoothly instead.
        return np.array([e + (cmd - raw) / self.t_track])


class RodsOnPower(Controller):
    """Common-mode rod speed tracks the power demand.

    [V] eq 50 integrates rod SPEED into reactivity, so a proportional law here
    is integral action on power and tracks with no offset.  No extra state.
    """
    name = "rods_on_power"

    def __init__(self, kp: float = 3000.0, v_max: float = 72.0,
                 deadband: float = 0.0):
        self.kp, self.v_max, self.deadband = kp, v_max, deadband

    def update(self, t, sig, xc, u, ctx):
        e = ctx.demand(t) - sig["Pn"]
        if abs(e) <= self.deadband:
            e = 0.0
        v = float(np.clip(self.kp * e, -self.v_max, self.v_max))
        u[IU["vrod_M"]] += v
        u[IU["vrod_AO"]] += v
        return EMPTY


class RodsOnTavg(Controller):
    """Common-mode rod speed holds Tavg on its load program.

    This is the reactor half of real load following: the turbine takes the
    load, the primary temperature moves, and the rods bring Tavg back to
    program.  Power is never commanded -- it is what comes out.
    """
    name = "rods_on_tavg"

    def __init__(self, program: Optional[Callable[[float], float]] = None,
                 kp: float = 6.0, v_max: float = 72.0, deadband: float = 0.55):
        self.program, self.kp = program, kp
        self.v_max, self.deadband = v_max, deadband

    def initialize(self, ctx):
        if self.program is None:
            self.program = tavg_program(ctx.plant)
        return EMPTY

    def update(self, t, sig, xc, u, ctx):
        e = self.program(ctx.demand(t)) - sig["Tavg"]
        # [V] quotes a +-0.55 degC dead band on the rod speed program
        e = e - float(np.clip(e, -self.deadband, self.deadband))
        v = float(np.clip(self.kp * e, -self.v_max, self.v_max))
        u[IU["vrod_M"]] += v
        u[IU["vrod_AO"]] += v
        return EMPTY


class AxialOffsetBand(Controller):
    """Differential rod speed holds DI inside a band.

    With the default core parameters the axial xenon mode is unstable (CP8
    finds a right-half-plane pair), so without this DI saturates on any
    transient longer than a few hours.  That is why real plants run an AO band.
    """
    name = "axial_offset"

    def __init__(self, band: float = 0.03, kp: float = 2000.0,
                 v_max: float = 72.0, target: float = 0.0):
        self.band, self.kp, self.v_max, self.target = band, kp, v_max, target

    def update(self, t, sig, xc, u, ctx):
        d = sig["DI"] - self.target
        e = d - float(np.clip(d, -self.band, self.band))
        vd = float(np.clip(self.kp * e, -self.v_max, self.v_max))
        # Differential mode.  Under the default rod_map ([L] Case 2) AOrod
        # drives the TOP half, so +AOrod / -Mrod raises DI.  Flip these two
        # lines if you pass a rod_map that swaps the halves.
        u[IU["vrod_AO"]] -= vd          # AOrod acts in half 2 (top)
        u[IU["vrod_M"]] += vd           # Mrod acts in half 1 (bottom)
        return EMPTY


class PressurizerHeaterPI(Controller):
    """Heater holds pressurizer pressure from below.  Heat-only, so the output
    is clipped at zero: it cannot bring pressure down.  Spray is the other half
    and is deliberately NOT shipped -- see the spray sign FLAG at CP0."""
    name = "pzr_heater"
    states = ("I_pp",)

    #: Installed heater capacity is 1300 kW ([K] Table III) and [V] gives
    #: K_heat = 1000 kW/mA, so the current saturates at about 1.3 mA.  [V]'s
    #: tabulated gains (4092 mA/MPa) would hit that limit at 0.3 kPa of error,
    #: i.e. pure bang-bang, which is realistic for heater BANKS but very hard
    #: on a stiff solver.  The defaults here are retuned for full output at
    #: 0.1 MPa of error; pass [V]'s values if you want the bang-bang case.
    def __init__(self, kp: float = 13.0, ki: float = 1.3,
                 ref: Optional[float] = None, i_max: float = 1.3,
                 t_track: float = 60.0):
        self.kp, self.ki, self.ref = kp, ki, ref
        self.i_max, self.t_track = i_max, t_track

    def update(self, t, sig, xc, u, ctx):
        ref = ctx.sig0["pp"] if self.ref is None else self.ref
        e = ref - sig["pp"]
        raw = self.kp * e + self.ki * xc[0]
        cmd = float(np.clip(raw, 0.0, self.i_max))
        u[IU["iheat"]] = cmd
        return np.array([e + (cmd - raw) / self.t_track])


class Manual(Controller):
    """Escape hatch: drive any input from a function of (t, sig).

        Manual("utg", lambda t, sig: 1.0 + 0.05*np.sin(t/300))
    """
    name = "manual"

    def __init__(self, input_name: str, fn: Callable[[float, Dict], float],
                 accumulate: bool = False):
        self.j = IU[input_name]
        self.fn = fn
        self.accumulate = accumulate

    def update(self, t, sig, xc, u, ctx):
        v = float(self.fn(t, sig))
        if self.accumulate:
            u[self.j] += v
        else:
            u[self.j] = v
        return EMPTY


# ---------------------------------------------------------------------------
# Vajpayee's control loops, section 5 and Table 2
# ---------------------------------------------------------------------------
# Every gain below is the paper's Table 2 value.  Where a sign had to be
# flipped against the paper's written equation to get negative feedback, the
# class says so and `VAJPAYEE_SIGN_NOTES` lists it.
#
#   loop                  input      output   eq   KP          KI
#   Core Neutronics       vrod       ilo      55   3.087e-2    3.947e-3
#   Temperature           vrod       irtd     56   1.930e-3    1.006e-5
#   Combined              vrod       both     57   retuned
#   SG Pressure           utg        ps       58   5.368e-1    1.169e-1
#   Pressurizer (Heater)  iheat      pp       59   4.092e+3    2.861e+2
#   Pressurizer (Spray)   mdot_spr   pp       60   2.935e+5    1.695e+5
#   Pressurizer Level     mdot_sur   lw       61   1.275e+3    7.366e+2
#   Turbine Speed         utg        w_tur    62   6.430e+2    8.426e+0

#: Table 2's gains EXACTLY as printed.  They are the paper's values; they do
#: NOT work in this implementation -- see VAJPAYEE_GAIN_NOTE.
VAJPAYEE_TABLE2 = {
    "core_neutronics": dict(kp=3.087e-2, ki=3.947e-3),
    "temperature":     dict(kp=1.930e-3, ki=1.006e-5),
    "sg_pressure":     dict(kp=5.368e-1, ki=1.169e-1),
    "pzr_heater":      dict(kp=4.092e+3, ki=2.861e+2),
    "pzr_spray":       dict(kp=2.935e+5, ki=1.695e+5),
    "pzr_level":       dict(kp=1.275e+3, ki=7.366e+2),
    "turbine_speed":   dict(kp=6.430e+2, ki=8.426e+0),
}

VAJPAYEE_GAIN_NOTE = """
Only the SG-pressure gains from Table 2 transfer directly.  The rod-loop gains
do not, by about five orders of magnitude, and the cause is in eq 50:

    d(rho_rod)/dt = G * v_rod          [V] eq 50

with G in cents/step and v_rod in steps per MINUTE, but every other derivative
in the paper is per SECOND.  Taken literally, full rod speed injects ~70
cents/s, which is prompt-critical in under a second.  This implementation
divides by 60, giving ~1.16 cents/s at the 72 spm limit -- physical, and
consistent with the ~12-cent corrective action the paper's own Fig. 12b shows.
That accounts for a factor of 60; the rest is unrecoverable from the text, so
Table 2's rod gains were evidently tuned against a different internal scaling.

The defaults below are retuned for THIS implementation and reject a reactivity
disturbance to ~4e-5 mA of residual ex-core current with <25 spm of rod motion.
Pass **VAJPAYEE_TABLE2["core_neutronics"] to use the paper's numbers verbatim.

The TURBINE SPEED gains do not transfer either, for an unrelated reason.  Eq 62
with KP = 643 implies a governor droop of 1/643 = 0.16 %, where real machines
run 4-5 %.  At that gain a 0.13 Hz speed error commands 80 units of valve travel
against a range of 0-1.5, so the valve slams shut and the set undershoots to
57.7 Hz.  The shipped defaults are droop-based: KP = 0.333 per Hz, which holds
a 10 % load rejection to a 0.66 Hz peak and returns to 60.003 Hz with the valve
landing at 0.84 -- no saturation.
"""

VAJPAYEE_SIGN_NOTES = {
    "SGPressurePI": "eq 58 writes (ps0 - ps); the valve has NEGATIVE gain on "
                    "ps (+0.05 valve -> -0.144 MPa), so as written with a "
                    "positive KP it is positive feedback. Implemented with the "
                    "sign that is actually stable.",
    "PressurizerSprayPI": "eq 60 writes (pp - pp0), i.e. spray arms above "
                          "setpoint. But in eq 31 as transcribed the spray "
                          "mass-addition term (h_w/C1p = +3.3e5) outweighs the "
                          "condensation term (h_spr - h_w = -2.9e5), so spray "
                          "RAISES pressure. The loop is therefore unstable as "
                          "written. Ships disabled by default.",
}


class _PI(Controller):
    """Shared PI body: one signal -> one actuator, with anti-windup."""
    states = ("I",)

    def __init__(self, kp, ki, signal, actuator, ref=None, lo=-1e12, hi=1e12,
                 sign=+1.0, t_track=30.0, accumulate=False, name=None):
        self.kp, self.ki, self.signal, self.actuator = kp, ki, signal, actuator
        self.ref, self.lo, self.hi, self.sign = ref, lo, hi, sign
        self.t_track, self.accumulate = t_track, accumulate
        self.j = IU[actuator]
        self.name = name or f"pi_{signal}_{actuator}"

    def initialize(self, ctx):
        self.base = 0.0 if self.accumulate else float(ctx.u0[self.j])
        return np.zeros(1)

    def update(self, t, sig, xc, u, ctx):
        ref = ctx.sig0[self.signal] if self.ref is None else self.ref
        e = self.sign * (ref - sig[self.signal])
        raw = self.base + self.kp * e + self.ki * xc[0]
        cmd = float(np.clip(raw, self.lo, self.hi))
        if self.accumulate:
            u[self.j] += cmd
        else:
            u[self.j] = cmd
        return np.array([e + (cmd - raw) / self.t_track])


class CoreNeutronicsPI(_PI):
    """[V] eq 55 -- reactor power via the EX-CORE DETECTOR current.

    This is the paper's loop, and it differs from `RodsOnPower` in a way that
    matters: it reads `ilo`, the amplifier output, not the true power. The
    amplifier carries real lag, so this is the honest version.
    """
    def __init__(self, kp=2.0e3, ki=2.0e2, v_max=72.0, **kw):   # retuned
        super().__init__(kp, ki, "ilo", "vrod_M", lo=-v_max, hi=v_max,
                         accumulate=True, name="core_neutronics", **kw)

    def update(self, t, sig, xc, u, ctx):
        d = super().update(t, sig, xc, u, ctx)
        u[IU["vrod_AO"]] += u[IU["vrod_M"]] - self.base   # common mode
        return d


class TemperaturePI(_PI):
    """[V] eq 56 -- reactor power via the RTD current (temperature loop)."""
    def __init__(self, kp=1.25e2, ki=5.0e-1, v_max=72.0, **kw):  # retuned
        super().__init__(kp, ki, "irtd", "vrod_M", lo=-v_max, hi=v_max,
                         accumulate=True, name="temperature", **kw)

    def update(self, t, sig, xc, u, ctx):
        d = super().update(t, sig, xc, u, ctx)
        u[IU["vrod_AO"]] += u[IU["vrod_M"]] - self.base
        return d


class CombinedPowerPI(Controller):
    """[V] eq 57 -- sum of the neutronics and temperature PI signals.

    The paper says the gains are "retuned" but does not give the retuned
    values, so these default to Table 2's and are yours to tune.
    """
    name = "combined_power"; states = ("I_lo", "I_rtd")

    def __init__(self, kp_lo=1.0e3, ki_lo=1.0e2,
                 kp_rtd=6.0e1, ki_rtd=2.5e-1, v_max=72.0, t_track=30.0):
        self.kp_lo, self.ki_lo = kp_lo, ki_lo
        self.kp_rtd, self.ki_rtd = kp_rtd, ki_rtd
        self.v_max, self.t_track = v_max, t_track

    def initialize(self, ctx):
        return np.zeros(2)

    def update(self, t, sig, xc, u, ctx):
        e_lo = ctx.sig0["ilo"] - sig["ilo"]
        e_rtd = ctx.sig0["irtd"] - sig["irtd"]
        raw = (self.kp_lo * e_lo + self.ki_lo * xc[0]
               + self.kp_rtd * e_rtd + self.ki_rtd * xc[1])
        v = float(np.clip(raw, -self.v_max, self.v_max))
        u[IU["vrod_M"]] += v
        u[IU["vrod_AO"]] += v
        bleed = (v - raw) / self.t_track
        return np.array([e_lo + bleed, e_rtd + bleed])


class PressurizerSprayPI(_PI):
    """[V] eq 60 -- pressurizer pressure from above, via spray.

    DISABLED BY DEFAULT and unstable if enabled: see VAJPAYEE_SIGN_NOTES.
    Shipped so the loop exists once eq 31 is corrected.
    """
    def __init__(self, kp=2.935e5, ki=1.695e5, m_max=50.0, **kw):
        super().__init__(kp, ki, "pp", "mdot_spr", lo=0.0, hi=m_max,
                         sign=-1.0, name="pzr_spray", **kw)


class PressurizerLevelPI(_PI):
    """[V] eq 61 -- pressurizer level.

    The paper manipulates `mdot_sur`, which eq 32 COMPUTES from coolant
    temperature derivatives and which therefore cannot be commanded. This acts
    on `mdot_ch` instead -- an added charging-flow input that enters the
    pressurizer mass balance identically. See the U_NAMES note.
    """
    def __init__(self, kp=1.275e3, ki=7.366e2, m_max=50.0, **kw):
        super().__init__(kp, ki, "lw", "mdot_ch", lo=-m_max, hi=m_max,
                         name="pzr_level", **kw)


class TurbineSpeedPI(_PI):
    """[V] eq 62 -- turbine speed via the governor valve.

    This is the loop a real PWR governor runs, and the one that makes `Pdem`
    a live disturbance rather than a dead end.

    Defaults are DROOP-BASED, not [V] Table 2.  A real governor runs 4-5 %
    droop -- a 5 % speed change moves 100 % load -- which is

        dutg/dw = 1.0 / (0.05 * 60 Hz) = 0.333 per Hz

    [V] prints KP = 643, about 1900x that, which drives the valve hard against
    a stop on any real speed error.  Pass **VAJPAYEE_TABLE2["turbine_speed"]
    for his numbers.  See VAJPAYEE_GAIN_NOTE.

    Requires `eq38_power_base` to be set correctly (it is, by default) or the
    speed signal is inert and this loop does nothing -- see `Params`.
    """

    def __init__(self, kp=0.333, ki=4.3e-3, u_min=0.0, u_max=1.5, **kw):
        super().__init__(kp, ki, "w_tur", "utg", lo=u_min, hi=u_max,
                         name="turbine_speed", **kw)


class LQG(Controller):
    """[V] section 6, eqs 63-70 -- Kalman filter plus LQR, one SISO loop.

        LQG(plant, "vrod_M", "Pn", q=1e-3, r=1e5, xi=5.0, theta=1.0)

    Estimator (eq 66):   dxh/dt = A xh + B du + Kf (y - C xh)
    Regulator (eq 70):   du     = -Kc xh
    with Kc from eq 69 and Kf from eq 64, each solved from its ARE.

    Everything is done in the SCALED coordinates `linearize` returns; the
    states span seven decades and the AREs are hopeless otherwise.

    `drop` removes states before the design.  `w_tur` is there by default
    because it is an UNCONTROLLABLE integrator (PBH rank 52/53 at the origin),
    which makes (A, B) unstabilisable and the LQR ARE fail outright.
    """
    name = "lqg"

    def __init__(self, plant, actuator, measurement, q=1e-3, r=1e5,
                 xi=5.0, theta=1.0, drop=("w_tur",), u_lo=-1e12, u_hi=1e12,
                 accumulate=None, name=None, shift=1e-3):
        from scipy.linalg import solve_continuous_are
        self.actuator, self.measurement = actuator, measurement
        self.j = IU[actuator]
        self.accumulate = (self.j in ROD_IDX) if accumulate is None else accumulate
        self.u_lo, self.u_hi = u_lo, u_hi
        self.name = name or f"lqg_{measurement}_{actuator}"
        self.keep = [i for i, nm in enumerate(plant.names) if nm not in drop]
        self.states = tuple(f"xh{i}" for i in range(len(self.keep)))
        self._design = dict(q=q, r=r, xi=xi, theta=theta)
        self.shift = shift
        self._are = solve_continuous_are

    def initialize(self, ctx):
        A, B, C, D, S = linearize(ctx.plant, ctx.x0, ctx.u0)
        k = self.keep
        A = A[np.ix_(k, k)]
        Bu = B[np.ix_(k, [self.j])]
        iy = OUTPUT_NAMES.index(self.measurement)
        Cy = C[np.ix_([iy], k)]
        self.A, self.B, self.C, self.S = A, Bu, Cy, S[k]
        n = A.shape[0]
        d = self._design
        # With ONE actuator, several integrators (the other rod group, the
        # pressurizer level) become uncontrollable, so (A, B) is not
        # stabilisable and the ARE has no finite solution.  Designing against
        # A - shift*I moves those modes just inside the left half plane, where
        # an uncontrollable mode is permitted.  Standard practice, and it means
        # the closed loop is guaranteed faster than -shift.
        Ad = A - self.shift * np.eye(n)
        Pc = self._are(Ad, Bu, d["q"] * np.eye(n), np.atleast_2d(d["r"]))
        self.Kc = (np.linalg.inv(np.atleast_2d(d["r"])) @ Bu.T @ Pc)   # eq 69
        Pf = self._are(Ad.T, Cy.T, d["xi"] * np.eye(n),
                       np.atleast_2d(d["theta"]))
        self.Kf = Pf @ Cy.T / d["theta"]                                # eq 64
        self.base = 0.0 if self.accumulate else float(ctx.u0[self.j])
        return np.zeros(n)

    def update(self, t, sig, xc, u, ctx):
        y = sig[self.measurement] - ctx.sig0[self.measurement]
        du = float(-(self.Kc @ xc)[0])                                  # eq 70
        cmd = float(np.clip(self.base + du, self.u_lo, self.u_hi))
        if self.accumulate:
            u[self.j] += cmd
        else:
            u[self.j] = cmd
        dxh = (self.A @ xc + self.B[:, 0] * du
               + self.Kf[:, 0] * (y - float((self.C @ xc)[0])))         # eq 66
        return dxh


# ---------------------------------------------------------------------------
# Sampled controllers                                            [CHANGE 2]
# ---------------------------------------------------------------------------

class DiscretePI(Controller):
    """A zero-order-hold PI loop: any signal -> any actuator.

        DiscretePI("ps", "utg", kp=0.537, ki=0.117, Ts=2.0)

    Demonstrates what sampling buys you.  `self.u_prev` is an ordinary Python
    attribute and is legal here only because update() is called exactly once
    per sample -- which is also what makes the rate limit implementable, since
    a rate limit needs the previous command and v1 rebuilt `u` every call.
    """
    states = ("I",)

    def __init__(self, signal: str, actuator: str, kp: float, ki: float,
                 Ts: float = 1.0, ref: Optional[float] = None,
                 lo: float = -1e9, hi: float = 1e9, sign: float = +1.0,
                 rate_limit: Optional[float] = None, t_track: float = 30.0,
                 name: Optional[str] = None):
        self.signal, self.actuator = signal, actuator
        self.kp, self.ki, self.Ts, self.ref = kp, ki, Ts, ref
        self.lo, self.hi, self.sign = lo, hi, sign
        self.rate_limit, self.t_track = rate_limit, t_track
        self.j = IU[actuator]
        self.accumulate = self.j in ROD_IDX
        self.name = name or f"dpi_{signal}_{actuator}"
        self.u_prev: Optional[float] = None

    def initialize(self, ctx):
        self.base = 0.0 if self.accumulate else float(ctx.u0[self.j])
        self.u_prev = None
        return np.zeros(1)

    def update(self, t, sig, xc, u, ctx):
        ref = ctx.sig0[self.signal] if self.ref is None else self.ref
        e = self.sign * (ref - sig[self.signal])
        raw = self.base + self.kp * e + self.ki * xc[0]
        cmd = float(np.clip(raw, self.lo, self.hi))
        if self.rate_limit is not None and self.u_prev is not None:
            step_max = self.rate_limit * self.Ts
            cmd = self.u_prev + float(np.clip(cmd - self.u_prev,
                                              -step_max, step_max))
        self.u_prev = cmd
        if self.accumulate:
            u[self.j] += cmd
        else:
            u[self.j] = cmd
        # difference equation, with back-calculation anti-windup
        return np.array([xc[0] + self.Ts * (e + (cmd - raw) / self.t_track)])


class SlidingAO(Controller):
    """Relay sliding-mode control of axial offset.  Uses BOTH v2 changes.

    Surface:   s = d(DI)/dt + lam * (DI - target)
    Control:   differential rod speed = -K * sign(s)

    Two things make this possible in v2 and not in v1:

      * `sign()` is discontinuous.  Inside the ODE right-hand side it collapses
        the step size; at a sample instant it is just a number.  Sampling is
        also the physically honest model -- the rod drive moves in discrete
        steps with a minimum interval ([K] quotes 0.833 s for H. B. Robinson).

      * d(DI)/dt is not in `sig`.  It is computable from the state alone,
        because rod SPEED enters d(rho_rod)/dt and not d(Pr)/dt -- so the power
        derivatives do not depend on u.  That needs `ctx.x`, which v1 withheld.

    Relative degree from rod speed to DI is 2 (speed -> reactivity -> power),
    which is why the surface carries the derivative term.
    """
    name = "sliding_ao"

    def __init__(self, K: float = 2.0, lam: float = 0.2, Ts: float = 5.0,
                 target: float = 0.0, v_max: float = 72.0,
                 boundary: float = 0.0):
        self.K, self.lam, self.Ts = K, lam, Ts
        self.target, self.v_max, self.boundary = target, v_max, boundary

    def surface(self, ctx, sig) -> float:
        pl = ctx.plant
        dx = pl.deriv(0.0, ctx.x, np.zeros(pl.m))     # u does not affect dPr
        dDI = 0.5 * (dx[pl.ix["Pr_h2"]] - dx[pl.ix["Pr_h1"]])
        return float(dDI + self.lam * (sig["DI"] - self.target))

    def update(self, t, sig, xc, u, ctx):
        s = self.surface(ctx, sig)
        if self.boundary > 0.0:
            sw = float(np.tanh(s / self.boundary))
        else:
            sw = float(np.sign(s))
        # vd > 0 withdraws half 1 and inserts half 2, which DECREASES DI,
        # so driving s to zero needs +K, not -K.  (Check it against the DC
        # gain: DI responds positively to vrod_M - vrod_AO.)
        vd = float(np.clip(self.K * sw, -self.v_max, self.v_max))
        # Differential mode.  Under the default rod_map ([L] Case 2) AOrod
        # drives the TOP half, so +AOrod / -Mrod raises DI.  Flip these two
        # lines if you pass a rod_map that swaps the halves.
        u[IU["vrod_AO"]] -= vd          # AOrod acts in half 2 (top)
        u[IU["vrod_M"]] += vd           # Mrod acts in half 1 (bottom)
        return EMPTY


# ---------------------------------------------------------------------------
# ClosedLoop: plant + controllers as one augmented ODE
# ---------------------------------------------------------------------------

class Result:
    """Simulation output.  `.df` is a pandas DataFrame, one row per sample,
    with every state, every derived signal, and every input as a named column.
    """

    def __init__(self, loop: "ClosedLoop", t, Z, U, demand):
        self.loop, self.t, self.Z, self.U = loop, t, Z, U
        self.X = Z[:loop.plant.n]
        self.XC = Z[loop.plant.n:]
        import pandas as pd
        cols = {"t": t, "t_h": t / 3600.0, "demand": demand}
        for k, nm in enumerate(loop.plant.names):
            cols[nm] = self.X[k]
        sig = [loop.plant.signals(self.X[:, j]) for j in range(len(t))]
        for k in sig[0]:
            cols[k] = np.array([sv[k] for sv in sig])
        for k, nm in enumerate(U_NAMES):
            cols["u_" + nm] = U[k]
        for k, nm in enumerate(loop.ctl_state_names):
            cols["xc_" + nm] = self.XC[k]
        self.df = pd.DataFrame(cols)

    def __getitem__(self, key):
        return self.df[key].to_numpy()

    #: Validated categorical colours, assigned in fixed order (never cycled).
    #: Worst adjacent-pair separation: colour-blind dE 9.1, normal-vision 19.6.
    COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948")

    def plot(self, cols: Sequence[str], x: Optional[str] = None, ax=None,
             subplots: bool = False, label: Optional[str] = None,
             figsize=None, **kw):
        """Plot one or more columns against time.

        Parameters
        ----------
        cols : list of str
            Column names, e.g. ``["Pn", "DI"]``.  Any column of ``self.df``:
            a signal, a state, an actuator (``u_*``) or a controller state
            (``xc_*``).
        x : {"t", "t_h"}, optional
            Time axis.  Default picks seconds for runs under an hour and
            hours above, which is almost always what you want.
        ax : matplotlib Axes or array of Axes, optional
            Draw onto existing axes instead of making new ones.  Pass the
            return value of an earlier ``plot`` call to overlay two runs.
        subplots : bool, default False
            One panel per column.  Use whenever the columns have different
            units -- ``Pn`` is about 1 and ``Tavg`` about 316, and on a shared
            axis the small one becomes a flat line.
        label : str, optional
            Legend label.  With one column this replaces the column name
            (useful when overlaying runs); with several it is used as a prefix.

        Returns
        -------
        matplotlib Axes, or an array of Axes when ``subplots=True``.

        Examples
        --------
        >>> res.plot(["Pn", "DI"])                      # one axis
        >>> res.plot(["Pn", "Tavg"], subplots=True)     # different units
        >>> ax = res_p.plot(["ps"], label="P")          # overlay two runs
        >>> res_pi.plot(["ps"], ax=ax, label="PI")
        """
        import matplotlib.pyplot as plt
        if x is None:
            x = "t" if float(self.t[-1]) <= 3600.0 else "t_h"
        xlabel = "time (s)" if x == "t" else "time (h)"
        cols = list(cols)

        if subplots:
            if ax is None:
                n = len(cols)
                figsize = figsize or (4.6 * n, 3.4)
                _f, ax = plt.subplots(1, n, figsize=figsize, squeeze=False,
                                      constrained_layout=True)
                ax = ax.ravel()
            for a, c, col in zip(np.atleast_1d(ax), cols, self.COLORS):
                a.plot(self.df[x], self.df[c], color=kw.pop("color", col),
                       lw=kw.get("lw", 1.8),
                       label=(label or c) if label else None, **{k: v for k, v
                       in kw.items() if k != "lw"})
                a.set_title(c, fontsize=10)
                a.set_xlabel(xlabel)
                a.grid(alpha=0.3)
                a.spines[["top", "right"]].set_visible(False)
                if label:
                    a.legend(fontsize=8, frameon=False)
            return np.atleast_1d(ax)

        if ax is None:
            _f, ax = plt.subplots(figsize=figsize or (8, 4.2),
                                  constrained_layout=True)
            ax._pwr_next = 0
        start = getattr(ax, "_pwr_next", 0)
        for k, c in enumerate(cols):
            lab = label if (label and len(cols) == 1) else \
                  (f"{label} {c}" if label else c)
            ax.plot(self.df[x], self.df[c],
                    color=self.COLORS[(start + k) % len(self.COLORS)],
                    lw=1.8, label=lab, **kw)
        ax._pwr_next = start + len(cols)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    def setpoint(self, ax, y: float, label: Optional[str] = None):
        """Draw a reference line in neutral grey, so colour always means
        'a controller' and never 'a target'."""
        ax.axhline(y, color="#8a8984", lw=1.0, ls=(0, (4, 3)), zorder=0,
                   label=label)
        if label:
            ax.legend(fontsize=8, frameon=False)
        return ax

    def summary(self, cols=("Pn", "DI", "Tavg", "ps", "pp", "P_tur")) -> str:
        lines = [f"{'signal':>8} {'start':>12} {'min':>12} {'max':>12} {'end':>12}"]
        for c in cols:
            if c not in self.df:
                continue
            v = self.df[c].to_numpy()
            lines.append(f"{c:>8} {v[0]:12.5f} {v.min():12.5f} "
                         f"{v.max():12.5f} {v[-1]:12.5f}")
        return "\n".join(lines)


class ClosedLoop:
    """Plant plus a list of controllers, integrated together.

        loop = ClosedLoop(plant, [ValveOnDemand(), RodsOnTavg()],
                          demand=ramp(1.0, 0.5))
        res  = loop.run(8 * 3600)

    Controllers run in list order and write into the input vector, so a later
    controller can override an earlier one on the same actuator.  Rod commands
    accumulate rather than overwrite (see Controller).  Controller states are
    appended after the plant states and integrated by the same solver.
    """

    def __init__(self, plant: Plant, controllers: Sequence[Controller],
                 demand: Optional[Callable[[float], float]] = None,
                 power: float = 1.0, di_target: float = 0.0,
                 x0: Optional[np.ndarray] = None,
                 u0: Optional[np.ndarray] = None,
                 measurement_noise: Optional[Callable[[float, Dict], Dict]] = None):
        self.plant = plant
        self.controllers = list(controllers)
        self.demand = demand if demand is not None else (lambda t: power)
        if x0 is None or u0 is None:
            x0, u0, res = trim(plant, power, di_target=di_target)
            self.trim_residual = res
        else:
            self.trim_residual = float(np.linalg.norm(plant.deriv(0.0, x0, u0)))
        self.x0, self.u0 = x0, u0
        self.ctx = LoopContext(plant, x0, u0, plant.signals(x0), self.demand)

        self.slices, self.ctl_state_names = [], []
        off = plant.n
        for c in self.controllers:
            k = len(c.states)
            self.slices.append(slice(off, off + k))
            self.ctl_state_names += [f"{c.name}.{sname}" for sname in c.states]
            off += k
        self.n = off
        self.ctx.x = x0
        self.xc0 = np.concatenate(
            [np.atleast_1d(c.initialize(self.ctx)).astype(float)
             for c in self.controllers] + [EMPTY])

        # CHANGE 2: split the controller list by sample period
        self.discrete = [i for i, c in enumerate(self.controllers)
                         if getattr(c, "Ts", None)]
        self.continuous = [i for i, c in enumerate(self.controllers)
                           if not getattr(c, "Ts", None)]
        self._holds: Dict[int, object] = {i: None for i in self.discrete}
        #: optional f(t, sig_dict) -> sig_dict, applied ONLY at sample instants.
        #: Injecting noise inside a continuous controller would corrupt the
        #: solver's Jacobian; at a sample instant it is harmless.
        self.measurement_noise = measurement_noise

    # -- the control law, exposed so you can inspect it without integrating --
    def _base_u(self) -> np.ndarray:
        u = self.u0.copy()
        u[IU["vrod_M"]] = 0.0        # rod channels accumulate from zero
        u[IU["vrod_AO"]] = 0.0
        return u

    def _apply_hold(self, u: np.ndarray, hold) -> None:
        """Re-apply what a sampled controller commanded at its last sample.

        Rod channels accumulate, so replay the DELTA; every other actuator is
        assigned, so replay the absolute value.
        """
        absolute, delta, touched = hold
        for j in np.nonzero(touched)[0]:
            if j in ROD_IDX:
                u[j] += delta[j]
            else:
                u[j] = absolute[j]

    def inputs(self, t: float, x: np.ndarray, xc_all: np.ndarray):
        """Evaluate the control law.  Sampled controllers contribute their
        held output and a zero state derivative; continuous ones are evaluated
        now."""
        u = self._base_u()
        self.ctx.x = x                      # CHANGE 1: live state for controllers
        sig = self.plant.signals(x)
        nc = self.n - self.plant.n
        dxc = np.zeros(nc)
        # sampled controllers first, so a continuous one can still override
        for i in self.discrete:
            if self._holds[i] is not None:
                self._apply_hold(u, self._holds[i])
        for i in self.continuous:
            c, sl = self.controllers[i], self.slices[i]
            k0, k1 = sl.start - self.plant.n, sl.stop - self.plant.n
            d = np.atleast_1d(
                c.update(t, sig, xc_all[k0:k1], u, self.ctx)).astype(float)
            if k1 > k0:
                dxc[k0:k1] = d
        return u, dxc

    def sample(self, t: float, x: np.ndarray, xc_all: np.ndarray,
               which: Optional[Sequence[int]] = None) -> None:
        """Fire the sampled controllers once: recompute their held outputs and
        advance their states as a difference equation.  Mutates xc_all."""
        self.ctx.x = x
        sig = self.plant.signals(x)
        if self.measurement_noise is not None:
            sig = self.measurement_noise(t, dict(sig))
        base = self._base_u()
        for i in (self.discrete if which is None else which):
            c, sl = self.controllers[i], self.slices[i]
            k0, k1 = sl.start - self.plant.n, sl.stop - self.plant.n
            scratch = base.copy()
            nxt = np.atleast_1d(
                c.update(t, sig, xc_all[k0:k1], scratch, self.ctx)).astype(float)
            delta = scratch - base
            self._holds[i] = (scratch.copy(), delta, np.abs(delta) > 0.0)
            if k1 > k0:
                xc_all[k0:k1] = nxt

    def deriv(self, t: float, z: np.ndarray) -> np.ndarray:
        n = self.plant.n
        x, xc = z[:n], z[n:]
        u, dxc = self.inputs(t, x, xc)
        return np.concatenate([self.plant.deriv(t, x, u), dxc])

    def run(self, t_end: float, n_out: int = 1500,
            method: Optional[str] = None,
            rtol: float = 1e-7, atol: float = 1e-9,
            x0: Optional[np.ndarray] = None,
            max_step: Optional[float] = None) -> Result:
        """Integrate the closed loop.

        If any controller has Ts set, the sampled path is used: the run is
        chopped at every sample instant, the discrete controllers fire once at
        each, and their outputs are held constant across the interval between.
        Otherwise the whole run is one call to the solver.

        COST.  The sampled path restarts the solver once per sample, at roughly
        80 ms a restart, so it scales with t_end/Ts and is about an order of
        magnitude slower than the continuous path.  Sample only the loop that
        needs it and leave the rest continuous.

        LSODA is the default for the continuous path, BDF for the sampled one: it switches between stiff and non-stiff itself
        and copes better than BDF with the discontinuities that dead bands and
        saturation limits introduce.  Pass method="BDF" or "Radau" if you want
        a pure implicit method.
        """
        # Bound the step, or the solver will march straight over a short
        # input pulse and never see it.  A persistent step is only smeared;
        # a 5 s pulse in a 300 s run can be missed entirely.
        if max_step is None:
            max_step = t_end / 200.0
        self._max_step = max_step
        z0 = np.concatenate([self.x0 if x0 is None else x0, self.xc0])
        if self.discrete:
            # BDF restarts about twice as fast as LSODA, and the sampled path
            # restarts the solver at every sample instant.
            return self._run_sampled(z0, t_end, n_out, method or "BDF",
                                     rtol, atol)
        return self._run_continuous(z0, t_end, n_out, method or "LSODA",
                                    rtol, atol)

    # -- purely continuous: one solver call, inputs recomputed afterwards ---
    def _run_continuous(self, z0, t_end, n_out, method, rtol, atol) -> Result:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(self.deriv, (0.0, t_end), z0, method=method,
                        t_eval=np.linspace(0.0, t_end, n_out),
                        rtol=rtol, atol=atol,
                        max_step=getattr(self, "_max_step", np.inf))
        if not sol.success:
            self._explain(sol)
        n = self.plant.n
        U = np.zeros((len(U_NAMES), len(sol.t)))
        for j in range(len(sol.t)):
            U[:, j], _ = self.inputs(sol.t[j], sol.y[:n, j], sol.y[n:, j])
        dem = np.array([self.demand(tt) for tt in sol.t])
        return Result(self, sol.t, sol.y, U, dem)

    # -- sampled: chop at every sample instant                 [CHANGE 2] ---
    def sample_grid(self, t_end: float) -> np.ndarray:
        """Union of every sampled controller's instants, plus 0 and t_end."""
        marks = {0.0, float(t_end)}
        for i in self.discrete:
            Ts = float(self.controllers[i].Ts)
            marks.update(np.arange(0.0, t_end + 0.5 * Ts, Ts))
        return np.array(sorted(m for m in marks if m <= t_end))

    def _run_sampled(self, z0, t_end, n_out, method, rtol, atol) -> Result:
        from scipy.integrate import solve_ivp
        n = self.plant.n
        grid = self.sample_grid(t_end)
        if len(grid) > 20000:
            print(f"  ! {len(grid)} sample intervals: this will be slow. "
                  "Raise Ts or shorten the run.")
        want = np.linspace(0.0, t_end, n_out)

        z = z0.copy()
        T, Z, U = [], [], []
        for a, b in zip(grid[:-1], grid[1:]):
            # fire every controller whose instant this is
            due = [i for i in self.discrete
                   if abs((a / self.controllers[i].Ts)
                          - round(a / self.controllers[i].Ts)) < 1e-9]
            if due:
                self.sample(a, z[:n], z[n:], which=due)
            keep = want[(want >= a) & (want < b)]
            t_eval = np.unique(np.concatenate([[a], keep, [b]]))
            sol = solve_ivp(self.deriv, (a, b), z, method=method,
                            t_eval=t_eval, rtol=rtol, atol=atol,
                            max_step=getattr(self, "_max_step", np.inf))
            if not sol.success:
                self._explain(sol)
                break
            # record only the requested output times, with the inputs that
            # were ACTUALLY applied -- holds are correct right now
            for j, tj in enumerate(sol.t):
                if tj in keep or (tj == 0.0 and 0.0 in want):
                    T.append(tj)
                    Z.append(sol.y[:, j].copy())
                    uj, _ = self.inputs(tj, sol.y[:n, j], sol.y[n:, j])
                    U.append(uj)
            z = sol.y[:, -1].copy()
        # final point
        T.append(float(grid[-1]))
        Z.append(z.copy())
        uf, _ = self.inputs(float(grid[-1]), z[:n], z[n:])
        U.append(uf)

        t = np.array(T)
        Zm = np.array(Z).T
        Um = np.array(U).T
        dem = np.array([self.demand(tt) for tt in t])
        return Result(self, t, Zm, Um, dem)

    @staticmethod
    def _explain(sol) -> None:
        print(f"  ! integrator stopped at t = {sol.t[-1] if len(sol.t) else 0:.1f} s: "
              f"{sol.message}")
        print("    The usual cause is a sign error in a controller.  Check "
              "that negative")
        print("    feedback really is negative: a valve step of +0.05 moves "
              "ps by -0.14 MPa,")
        print("    and withdrawing rods (positive vrod) RAISES power.")


# ---------------------------------------------------------------------------
# Reference configurations
# ---------------------------------------------------------------------------

def turbine_leading(plant: Optional[Plant] = None,
                    demand: Optional[Callable[[float], float]] = None,
                    ao_band: float = 0.03, pzr: bool = True,
                    **kw) -> ClosedLoop:
    """Real load following: the grid moves the turbine, the reactor follows.

    Valve tracks load demand -> steam flow changes -> primary temperature moves
    -> rods bring Tavg back to its load program.  Reactor power is never
    commanded; it is the outcome.  Steam pressure is NOT controlled and stays
    near constant on its own, because the Tavg program is exactly the one that
    holds it there (see tavg_program).
    """
    plant = plant or Plant()
    ctls: List[Controller] = [GridDemand(), ValveOnDemand(),
                              RodsOnTavg(), AxialOffsetBand(band=ao_band)]
    if pzr:
        ctls.append(PressurizerHeaterPI())
    return ClosedLoop(plant, ctls, demand=demand, **kw)


def reactor_leading(plant: Optional[Plant] = None,
                    demand: Optional[Callable[[float], float]] = None,
                    ao_band: float = 0.03, pzr: bool = True,
                    **kw) -> ClosedLoop:
    """The other way round: rods set power, the valve holds steam pressure.

    Useful as a contrast and for isolated core studies, but it is not how a
    plant on a grid behaves.
    """
    plant = plant or Plant()
    ctls: List[Controller] = [GridDemand(), RodsOnPower(),
                              AxialOffsetBand(band=ao_band), SGPressurePI()]
    if pzr:
        ctls.append(PressurizerHeaterPI())
    return ClosedLoop(plant, ctls, demand=demand, **kw)


def open_loop(plant: Optional[Plant] = None, power: float = 1.0,
              **kw) -> ClosedLoop:
    """No controllers at all.  Every actuator sits at its trim value; add your
    own with `loop.controllers.append(...)` before the first run, or pass a
    list to ClosedLoop directly."""
    return ClosedLoop(plant or Plant(), [], power=power, **kw)
