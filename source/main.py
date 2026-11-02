#!/usr/bin/env python3
"""
main.py -- entry point of the pickup analysis pipeline.

The modelling functions live in the modules
    pickup_rlc.py      geometry -> R, L, C of one coil
    pickup_network.py  netlist + nodal analysis solver, plot helpers
    pickup_design.py   one-step Design API (topology + output chain)
This script only defines the studies to run and calls those functions.
All plots and CSV files are written to <repo>/output.

    python source/main.py                 # run every study
    python source/main.py impedance       # run selected studies
    python source/main.py --list          # show available studies

Add your own design study as a function below and register it in STUDIES.
"""
from __future__ import annotations
import argparse
from dataclasses import replace

import numpy as np

import pickup_rlc as prl
from pickup_network import (CoilRLC, Network, ex_single, ex_humbucker, ex_n_series,
                            cross_check_lcapy, plot_bode, peak)
from pickup_design import (Design, Pickup, Series, Parallel, Shunt, RLC,
                           volume_pot, tone_control, cable, amp_input,
                           compare, compare_impedance,
                           dummy_coil, match_dummy_N, hum_rejection)

F = np.logspace(1, 5, 2000)          # 10 Hz ... 100 kHz


# ----------------------------------------------------------------------------
# Studies
# ----------------------------------------------------------------------------
def study_coil():
    """Stage 1: self test, R/L/C report of the example coils, sweep over N."""
    prl.self_test()
    print()
    print(prl.report(prl.example_strat()))
    print()
    print(prl.report(prl.example_humbucker_coil()))
    print()
    base = prl.example_strat()
    rows = prl.sweep_N(base, range(4000, 10001, 1000))
    print(f"N sweep: {base.name}")
    print(prl.format_sweep(rows))
    prl.save_sweep_csv(rows, "sweep_N.csv")
    prl.plot_sweep(rows, "sweep_N.png", base.name)


def study_networks():
    """Stage 2: Bode plots of basic networks (cable sweep, humbucker modes)."""
    strat = CoilRLC.from_spec(prl.example_strat())
    hb = CoilRLC.from_spec(prl.example_humbucker_coil())
    for c in (strat, hb):
        print(f"{c.name}: R={c.R/1e3:.2f} kOhm, L={c.L:.3f} H, C={c.C*1e12:.1f} pF, "
              f"f0(unloaded)={c.f0/1e3:.2f} kHz")

    # single Strat coil: unloaded vs. typical load, cable sweep
    curves = {"unloaded": Network().coil("P", "out", "0", strat).solve(F, "out")}
    for Cc in (300e-12, 700e-12, 1000e-12):
        curves[f"250k vol+tone, cable {Cc*1e12:.0f} pF, amp 1M"] = \
            ex_single(strat, cable_C=Cc).solve(F, "out")
    plot_bode(F, curves, "bode_strat_cable.png", "Strat single coil - cable capacitance")

    # humbucker: two identical coils, series vs parallel, 500k pots
    ld = dict(pot=500e3, tone_pot=500e3, cable_C=500e-12)
    curves = {
        "one coil (split)": ex_single(hb, **ld).solve(F, "out"),
        "2 coils series": ex_humbucker(hb, "series", **ld).solve(F, "out"),
        "2 coils parallel": ex_humbucker(hb, "parallel", **ld).solve(F, "out"),
        "4 coils series": ex_n_series(hb, 4, **ld).solve(F, "out"),
    }
    plot_bode(F, curves, "bode_humbucker_modes.png",
              "Identical coils: split / series / parallel / 4x series")
    for lab, H in curves.items():
        fp, mp = peak(F, H)
        print(f"  {lab:18s}: peak {fp/1e3:6.2f} kHz  {mp:+6.1f} dB")

    # optional symbolic cross check
    try:
        Hs, _, err = cross_check_lcapy(hb, F[::50])
        print("lcapy cross-check, series humbucker: max rel. error vs MNA =", f"{err:.2e}")
        print("  symbolic H(s) =", Hs.simplify())
    except ImportError:
        print("lcapy not installed - symbolic cross-check skipped")


def study_design():
    """Stage 3: geometry -> network -> Bode in one step, several topologies."""
    base = prl.example_humbucker_coil()
    coil_5k = replace(base, name="HB coil N=5000")
    coil_6k = replace(base, name="HB coil N=6000", N=6000)
    standard_chain = volume_pot(500) + tone_control(500, 22) + cable(500) + amp_input(1.0)

    designs = [
        Design(Series(Pickup(coil_5k), Pickup(coil_5k)), standard_chain, "2x N=5000 series"),
        Design(Parallel(Pickup(coil_5k), Pickup(coil_5k)), standard_chain, "2x N=5000 parallel"),
        Design(Series(Pickup(coil_6k), Pickup(coil_6k)), standard_chain, "2x N=6000 series"),
        # 2 parallel pairs in series (4 coils) + extra series RC filter across the output
        Design(Series(Parallel(Pickup(coil_5k), Pickup(coil_5k)),
                      Parallel(Pickup(coil_5k), Pickup(coil_5k))),
               [Shunt(RLC(R_=47e3, C_=1e-9, mode="series"))] + standard_chain,
               "4x N=5000 (2 par pairs in series) + 47k/1nF shunt"),
        # same as the first design, but one coil without magnet/pole pieces
        Design(Series(Pickup(coil_5k), Pickup(coil_5k, air_core=True)), standard_chain,
               "N=5000 series, one coil air core"),
    ]
    compare(designs, F, "bode_design_example.png", "Geometry -> network -> Bode in one step")


def _strat_dummy_matched():
    real = prl.example_strat()
    return real, match_dummy_N(real, dummy_coil(real))


def study_dummy():
    """Hum-cancelling air-core dummy coil in series with a Strat coil."""
    real, N_match = _strat_dummy_matched()
    chain = volume_pot(250) + tone_control(250, 22) + cable(500) + amp_input(1.0)
    variants = [
        Design(Pickup(real), chain, "real coil only"),
        Design(Series(Pickup(real), Pickup(dummy_coil(real), emf=0, hum=-1)), chain,
               f"+ dummy, same winding (N={real.N})"),
        Design(Series(Pickup(real), Pickup(dummy_coil(real, N=N_match), emf=0, hum=-1)), chain,
               f"+ dummy, N={N_match} matched to N*A_eff"),
    ]
    ref = hum_rejection(variants[0])
    for v in variants:
        hr = hum_rejection(v)
        print(f"  {v.name:38s} hum/string at 50 Hz: {hr[50.0]-ref[50.0]:+7.1f} dB vs real coil only")
    compare(variants, F, "bode_dummy_string.png", "String signal: real coil + series dummy coil")
    compare(variants, F, "bode_dummy_hum.png", "Hum (uniform field): real coil + series dummy coil",
            excitation="hum")


def study_impedance():
    """Main result: input impedance seen by the amp (no sources, amp input excluded)."""
    real, N_match = _strat_dummy_matched()
    hb = prl.example_humbucker_coil()
    chain_250 = volume_pot(250) + tone_control(250, 22) + cable(500) + amp_input(1.0)
    chain_500 = volume_pot(500) + tone_control(500, 22) + cable(500) + amp_input(1.0)
    zdesigns = [
        Design(Pickup(real), [], "Strat coil alone (no pots, no cable)"),
        Design(Pickup(real), chain_250, "Strat coil + pots + cable"),
        Design(Series(Pickup(real), Pickup(dummy_coil(real, N=N_match), emf=0, hum=-1)),
               chain_250, f"Strat + dummy N={N_match} + pots + cable"),
        Design(Series(Pickup(hb), Pickup(hb)), chain_500, "HB 2 coils series + 500k pots + cable"),
        Design(Parallel(Pickup(hb), Pickup(hb)), chain_500, "HB 2 coils parallel + 500k pots + cable"),
    ]
    compare_impedance(zdesigns, F, "impedance_examples.png", "Input impedance seen by the amplifier")


STUDIES = {
    "coil": study_coil,
    "networks": study_networks,
    "design": study_design,
    "dummy": study_dummy,
    "impedance": study_impedance,
}


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Guitar pickup circuit analysis")
    ap.add_argument("studies", nargs="*",
                    help=f"studies to run (default: all): {', '.join(STUDIES)}")
    ap.add_argument("--list", action="store_true", help="list the available studies")
    args = ap.parse_args(argv)

    if args.list:
        for name, fn in STUDIES.items():
            print(f"{name:10s} {fn.__doc__.strip()}")
        return
    unknown = [s for s in args.studies if s not in STUDIES]
    if unknown:
        ap.error(f"unknown study {', '.join(unknown)}; choose from {', '.join(STUDIES)}")
    for name in args.studies or STUDIES:
        print(f"\n===== {name} =====")
        STUDIES[name]()
    print(f"\nOutputs written to {prl.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
