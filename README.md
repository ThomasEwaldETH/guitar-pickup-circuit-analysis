# Guitar pickup circuit analysis

Analytical design routine for electric-guitar pickups, from coil geometry to the
electrical behaviour of the complete guitar circuit, in plain Python (numpy, scipy,
matplotlib). No FEM, no SPICE.

The pipeline has three stages:

```
 coil geometry (wire, bobbin, turns, pole pieces / magnets)
        │   source/pickup_rlc.py
        ▼
 lumped R, L, C of one coil  (+ unloaded resonance f0)
        │   source/pickup_network.py  (nodal analysis solver)
        ▼
 network of coils in series / parallel, dummy coils, extra R/L/C filters,
 pots, tone control, cable, amp input
        │   source/pickup_design.py   (one-step design API)
        ▼
 input impedance Z_in(f) seen by the amplifier   ← main result
 (and, optionally, voltage transfer functions V_out/EMF and hum response)
```

**Main result: input impedance.** The voltage a string induces in a coil is not
known well enough to be modelled reliably. So the main quantity is the impedance the
amplifier sees when it looks into the guitar jack. All induced voltages are set to
zero, a 1 A test current is injected at the jack, and `Z_in = V / I`. This depends
only on R, L and C, which can be calculated from geometry, and it can be measured
directly with an impedance analyser or LCR meter. The magnitude peak of `|Z_in|`
and its frequency describe the resonant behaviour that gives a pickup its sound.

The older voltage-based outputs (Bode plot of `V_out / EMF`, hum response of
dummy coils) still work and are useful for comparing designs relative to each
other.

---

## Quick start

Requires Python ≥ 3.10 (developed on 3.13).

```bash
git clone https://github.com/<user>/guitar-pickup-circuit-analysis.git
cd guitar-pickup-circuit-analysis
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

python source/main.py              # run every study
python source/main.py impedance    # run only the studies you name
python source/main.py --list       # list the studies
```

`source/main.py` is the only script. It defines the studies (which coils, which
networks, which plots) and calls the modelling functions in the three modules,
which contain no runnable code of their own. It can be started from any working
directory. All plots and CSV files are written to `output/` at the repository root.
That folder is git-ignored, so the plots are not in the repository: run `main.py`
to make them.

| Study | What it does | Writes to `output/` |
| --- | --- | --- |
| `coil` | self test, R/L/C report of the example coils, sweep over N | `sweep_N.csv`, `sweep_N.png` |
| `networks` | Bode plots of basic networks, optional lcapy cross-check | `bode_strat_cable.png`, `bode_humbucker_modes.png` |
| `design` | one-step designs: series, parallel, 4 coils, extra filter, air core | `bode_design_example.png` |
| `dummy` | hum-cancelling dummy coil study | `bode_dummy_string.png`, `bode_dummy_hum.png` |
| `impedance` | **main result:** input impedance seen by the amp | `impedance_examples.png` |

To run your own design, add a `study_<name>()` function to `main.py` and register
it in the `STUDIES` dictionary.

Optional: `pip install lcapy` enables the symbolic cross-check in the `networks`
study (gives `H(s)` as a formula and compares it with the numeric solver).

---

## Repository structure

```
guitar-pickup-circuit-analysis/
├── README.md            project documentation; also the instructions for Claude Code
├── CLAUDE.md            one line that imports README.md for Claude Code
├── requirements.txt     numpy, scipy, matplotlib
├── .gitignore           ignores output/, caches, virtual environments
├── source/
│   ├── main.py            the script you run: defines and runs the studies
│   ├── pickup_rlc.py      stage 1: geometry -> R, L, C of one coil
│   ├── pickup_network.py  stage 2: netlist + nodal analysis solver, plot helpers
│   └── pickup_design.py   stage 3: one-step Design API (topology + output chain)
└── output/              generated plots and CSV (git-ignored, created on first run)
```

The modules import each other by plain module name (`import pickup_rlc as prl`).
Python adds the script's own folder to the import path, so running
`python source/main.py` works without installing anything. To import the modules
from a notebook or another script, put `source/` on `sys.path` first.

---

## Stage 1: one coil, geometry to R, L, C (`pickup_rlc.py`)

Input is a `PickupSpec`:

| Field | Meaning |
| --- | --- |
| `N` | number of turns |
| `wire` | `Wire(d_bare_mm, ins_um, eps_ins)` or `Wire.from_awg(42, "heavy_formvar")` |
| `bobbin` | `Bobbin("rect" or "round", width_mm, depth_mm, corner_radius_mm, height_mm, ...)`, the surface the first layer lies on |
| `stacks` | list of `CoreStack`: columns of `Segment`s (pole pieces, slugs, magnets) with material and length, repeated `n` times |
| `scheme` | `"standard"` (layers wound back and forth) or `"flyback"` |
| `kappa` | empirical factor on the core contribution to L (default 0.85), meant to be calibrated |
| `C_extra_pF` | extra lumped capacitance (leads, cover, baseplate), user input |
| `T_C` | temperature for the copper resistance |
| `air_core` | `True` ignores all core stacks: a pure air coil |

`analyse(spec)` returns R, L (split into air and core parts), C and f0. `report(spec)`
prints a summary, and `sweep_N(spec, N_values)` evaluates a range of turn counts with
the winding rebuilt for each N.

### Winding model
The winding is assumed to be an ideal **orthocyclic** stack: in-layer pitch d0 (wire
outer diameter), layer pitch √3/2·d0, alternate layers shifted by d0/2 and holding one
turn fewer when the space is tight. For every layer the mean turn length and enclosed
area are computed with an offset (Steiner) path around the bobbin, so rectangular
bobbins with rounded corners are handled exactly.

### R: DC resistance
Wire length is the sum of turns × mean turn length per layer. Copper resistivity is
1.72e-8 Ω·m at 20 °C with a temperature coefficient of 0.00393 /K.

### L: inductance
`L = L_air + L_core`
- `L_air`: Wheeler's multilayer formula [W28], with the rectangular coil replaced by
  a round coil of equal mean area.
- `L_core`: for each core stack, a series-reluctance composite permeability is
  computed along the stack (e.g. steel slug + ceramic magnet below the coil). It is
  then corrected for demagnetisation (prolate-ellipsoid factor by default, or the
  magnetometric factor of a uniformly magnetised cylinder) to give `mu_eff`, and
  `ΔL = kappa · n · μ0 · (mu_eff − 1) · A · N² / l_ref` with
  `l_ref = sqrt(l_coil² + D_coil²)`.
- `kappa` collects field averaging, oblong coil shape and leakage in one number.
  `calibrate_kappa(L_meas, ...)` fits it to one measured pickup.

Material permeabilities in `MATERIALS_MUR` (alnico, steel, ceramic, NdFeB, ...) are
typical small-signal values, not datasheet values.

### C: winding capacitance
Koch's (1968) layer-to-layer capacitance for orthocyclic windings [K68], combined into
one terminal capacitance with the energy method of Biela & Kolar (2008) [BK08]. The
layer capacitance is C0/3 for a standard winding and C0/4 for flyback, and an
incomplete top layer is handled with BK08 eq. (25).

### Example results
| Coil | R_dc | L | C | f0 (unloaded) |
| --- | --- | --- | --- | --- |
| Strat-style, 6 AlNiCo5 rods, AWG42, N=8000 | 5.33 kΩ | 2.084 H | 76.9 pF | 12.57 kHz |
| Humbucker coil, 6 steel slugs + ceramic bar, AWG43, N=5000 | 5.34 kΩ | 1.044 H | 140.9 pF | 13.12 kHz |

---

## Stage 2: circuit solver (`pickup_network.py`)

Each coil becomes a `CoilRLC(R, L, C)` (`CoilRLC.from_spec(spec)`), modelled as

```
   EMF ── R_dc ── L ──┬── terminal +
                      C
   terminal − ────────┴
```

i.e. the induced voltage in series with R and L, and the winding capacitance across
the terminals.

`Network` is a small netlist (`R`, `C`, `L`, `V`, `coil`) with node names as strings
and ground `"0"`. `Network.solve(f, out)` does a **modified nodal analysis** (MNA)
with numpy: one complex linear solve per frequency, all frequencies at once.
`Network.impedance(f, node)` shorts all sources and injects a 1 A test current.

The solver was checked against lcapy's symbolic solution for two coils in series
(largest relative difference 1.5e-14), and the impedance of a Strat coil with pots and
cable against a hand-derived formula (6e-15).

Helpers: `plot_bode`, `plot_impedance`, `peak`, and example networks `ex_single`,
`ex_humbucker(mode="series"|"parallel")`, `ex_n_series`, `load_chain`.

---

## Stage 3: one-step design (`pickup_design.py`)

Geometry, topology and output chain in one object:

```python
from dataclasses import replace
import pickup_rlc as prl
from pickup_design import *

coil = replace(prl.example_humbucker_coil(), N=6000)          # geometry
d = Design(
    Series(Parallel(Pickup(coil), Pickup(coil)),               # topology
           Parallel(Pickup(coil), Pickup(coil))),
    [Shunt(RLC(R_=47e3, C_=1e-9, mode="series"))]              # extra filter to ground
    + volume_pot(500) + tone_control(500, 22) + cable(500) + amp_input(1.0),
    name="4 coils, 2 parallel pairs in series")

Z = d.impedance(f)                     # what the amp sees (amp input left out)
compare_impedance([d, ...], f, "my_impedance.png")   # -> output/my_impedance.png
compare([d, ...], f, "my_bode.png")                  # V_out / EMF Bode plot
```

Building blocks (all two-terminal, nest as deep as needed):

| Block | Meaning |
| --- | --- |
| `Pickup(spec, emf=1, hum=1, air_core=None, hum_scale=1)` | one coil from geometry |
| `R(v)`, `L(v)`, `C(v)` | single parts (SI units) |
| `RLC(R_=, L_=, C_=, mode="series"/"parallel")` | combined part |
| `Series(a, b, ...)`, `Parallel(a, b, ...)` | topology |

The output chain is a list of stages in signal order, starting at the pickup node:
`Shunt(block)` connects a block to ground, `Through(block)` puts it in the signal line.
Ready-made stages: `volume_pot(R_kohm)` (fully up), `tone_control(R_kohm, C_nF, position)`,
`cable(C_pF, R_ohm)` (lumped), `amp_input(R_Mohm, C_pF)`. The amp input is an
`AmpShunt` and is left out of `Design.impedance()` unless `with_amp=True`.

### Hum-cancelling dummy coils
The idea: air-core coils (no magnet) in series with the real pickup, in the same
orientation but reversed polarity. They don't pick up the strings, but they pick up
the same background hum, which then cancels.

Modelled with two independent excitations (superposition): `emf` is the string signal
(0 for a dummy) and `hum` is the hum polarity. The hum EMF is `N·A_eff · dB/dt`, where
`A_eff` includes the flux the pole pieces concentrate (`hum_area_turns`).
`dummy_coil(real, N)`, `match_dummy_N(real, dummy)` (turns for equal N·A_eff) and
`hum_rejection(design)` support the study. For the impedance, `emf` and `hum` do not
matter; only R, L and C count.

Results for the Strat example coil (250k pots, 22 nF tone, 500 pF cable):

| Network | Hum vs. string at 50 Hz | max \|Z_in\| | at |
| --- | --- | --- | --- |
| Strat coil alone, no load | – | 5088 kΩ | 12.58 kHz |
| Strat coil + pots + cable | 0 dB (reference) | 106 kΩ | 4.67 kHz |
| + dummy, same winding (N=8000) | −9.9 dB | – | – |
| + dummy matched to N·A_eff (N=10514) | −84 dB | 107 kΩ | 3.29 kHz |

An identical dummy cancels only about 10 dB, because the alnico rods raise the real
coil's N·A_eff from about 4.15 to 6.10 m². The matched dummy cancels well at 50 Hz but
adds a lot of L and R, which pulls the resonance down. The design question is how to
get enough N·A_eff with little L (large area, few turns).

---

## Conventions (for contributors and for Claude Code)

- **Units:** SI everywhere internally (Ω, H, F, Hz, m). Geometry inputs in mm, wire
  insulation in µm, helper arguments are named with their unit (`R_kohm`, `C_nF`,
  `C_pF`, `R_Mohm`).
- **Outputs:** never write files next to the sources. Pass a bare file name to the
  plot/CSV helpers; they resolve it with `pickup_rlc.output_path()` into `output/`
  (absolute paths are used unchanged). New writers must use `output_path()` too.
- **Dependencies:** the core stays numpy + scipy + matplotlib. lcapy is optional and
  only used for cross-checks; imports of optional packages go inside functions.
  matplotlib is imported lazily with the `Agg` backend.
- **Layering:** `main.py` is the only entry point and holds no modelling code.
  `pickup_rlc` knows nothing about circuits, `pickup_network` knows nothing about
  geometry beyond `CoilRLC.from_spec`, `pickup_design` combines both. The modules
  have no `if __name__ == "__main__":` blocks. Keep it that way.
- **Verification:** any change to the solver or a formula should be checked against
  an independent result (lcapy symbolic solution, hand formula, limit case) and the
  check described in the commit message. Numbers in this README come from running
  the scripts; update them when models change.
- **Studies:** every runnable example is a `study_*` function in `main.py`.
  `python source/main.py` (all studies) is the smoke test. Keep it running end to
  end.
- **Language:** code, comments and docs in English. The project owner also writes
  German.
- **Placeholders:** values marked TYPICAL (material µr, enamel thickness, kappa) are
  not measured. Don't present results based on them as accurate in absolute terms.

---

## What is missing

### Model physics: single coil
1. **Koch capacitance formula unverified.** The integrands of the Koch factor were
   transcribed from a PDF text layer in which square roots were lost and then
   reconstructed. The structure passes plausibility checks (touching-wire limit,
   thumb rule), but it has not been checked line by line against Koch 1968 or Biela &
   Kolar 2008. Absolute C values may be off.
2. **Only ideal orthocyclic winding.** Real pickups are usually machine-wound with
   traverse patterns or scatter-wound by hand. Both have larger, irregular layer
   spacing and lower capacitance than an ideal orthocyclic stack. A scatter or random
   winding model (e.g. a fill factor or statistical layer spacing) is missing, so C is
   probably overestimated.
3. **kappa not calibrated.** The core contribution to L uses one empirical factor
   (0.85 by default). It has not been fitted to a measured pickup yet.
4. **Material data are placeholders.** µr values in `MATERIALS_MUR` and enamel
   thicknesses in `INSULATION_PRESETS_UM` are typical values, not from datasheets.
   Permeability of alnico depends on grade and magnetisation state.
5. **Demagnetisation is approximate.** The ellipsoid factor is used for cylinders, and
   the cylinder factor is the magnetometric one (an upper bound). The fluxmetric
   (midplane) factor relevant for high-µ rods (Ripka et al. 2021) is not implemented.
6. **Wheeler formula for round coils.** Rectangular bobbins are approximated by a round
   coil of equal area. A rectangular-coil formula (e.g. Grover) is missing.
7. **No eddy-current or core losses.** Eddy currents in pole pieces, magnets, metal
   covers and baseplates make R rise and L fall with frequency and damp the resonance.
   This is the largest physical effect missing from the model: predicted impedance
   peaks are probably too high and too sharp.
8. **No skin or proximity effect** in the wire. Small at audio frequencies for thin
   wire, but not checked.
9. **Extra capacitance not computed.** Lead wires, cover and baseplate capacitance are
   a manual input (`C_extra_pF`).
10. **Temperature** only affects R, not L, C or µr.

### Model physics: network
11. **No mutual inductance between coils.** All coils are treated as magnetically
    independent. The two coils of a humbucker, and a dummy coil mounted close to a real
    pickup, are coupled. Their coupling changes the total L and the resonance. Adding a
    coupling coefficient k (mutual inductance element in the MNA) is the most important
    network extension.
12. **Single lumped RLC per coil.** Higher self-resonance modes of the distributed
    winding are not modelled. Fine below the first resonance, questionable above it.
13. **String excitation unknown.** No model of the string–magnet interaction, so the
    absolute EMF (and its frequency dependence, including the pickup's aperture /
    position-on-string comb filter) is missing. This is why the impedance is the main
    result.
14. **Hum model simplified.** Assumes a uniform external field. `A_eff` with the core
    flux concentration is an estimate, and `hum_scale` is uncalibrated. Real hum fields
    have gradients, so dummy coils at a different position see a different field.
15. **Cable is lumped.** Only a capacitance (plus optional series R). Fine for normal
    cable lengths, no transmission-line model.
16. **Controls only partly modelled.** Volume pot only fully up (no voltage divider at
    other positions, no taper), tone pot as a rheostat, no treble bleed, no switching
    (pickup selector, coil split, series/parallel switch).
17. **Amplifier input simplified.** A resistor (and optional capacitor), no active
    electronics, buffers or preamps.

### Validation
18. **No comparison with measurements.** Neither R, L, C of a real coil nor a measured
    impedance curve has been compared with the model yet. Needed: one or more measured
    pickups (geometry, N, wire), measured R_dc, L, C (LCR meter) and |Z(f)| sweep.
19. **No measurement import.** No reader for impedance analyser / LCR / audio-interface
    CSV data and no overlay of measurement and model in the plots.
20. **No fitting.** No routine to fit kappa, C_extra, an eddy-current model or
    material µr to measured data.

### Software
21. **No automated tests.** `self_test()` in `pickup_rlc.py` only prints values. Needed:
    `pytest` tests with tolerances (limit cases, the lcapy cross-check, the hand formula
    for the impedance, regression values of the examples).
22. **Not a package.** No `pyproject.toml`, modules import each other by plain name and
    only work from `source/` on the path.
23. **Designs live in code.** No configuration file format (YAML/JSON/TOML) for coil
    specs and networks. Every new design means writing a `study_*` function in
    `main.py`; the command line only selects which studies run.
24. **No parameter sweeps or optimisation at network level.** `sweep_N` exists for a
    single coil, but there's no sweep or optimiser over network parameters (e.g. find
    the dummy geometry with maximal hum rejection for a given minimum resonance
    frequency).
25. **No time-domain or nonlinear simulation.** Not needed for impedance, but if wanted
    later, export to SPICE (PySpice / ngspice) would be the route.
26. **Example plots are not in the repository**, because `output/` is ignored. A
    `docs/` folder with a few committed images would let them show up in this README.

---

## References

- **[K68]** J. Koch, *Valvo Berichte* XIV(3), 99–119 (1968).
- **[BK08]** J. Biela, J. W. Kolar, "Using transformer parasitics for resonant
  converters: a review of the calculation of the stray capacitance of transformers",
  *IEEE Trans. Ind. Appl.* 44(1), 223–233 (2008).
- **[W28]** H. A. Wheeler, "Simple inductance formulas for radio coils", *Proc. IRE*
  16, 1398 (1928).
- P. Ripka et al., demagnetisation factors of cylinders, *J. Magn. Magn. Mater.* (2021).
- lcapy, symbolic linear circuit analysis in Python: https://github.com/mph-/lcapy
