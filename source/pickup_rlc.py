"""
pickup_rlc.py -- analytical R, L, C model of an electric-guitar pickup (no FEM)

R : DC resistance from wire length (mean turn length per layer).
L : air-core multilayer coil (Wheeler) + core term (pole pieces / slugs / magnet
    stack) using demagnetisation-corrected effective permeability.
C : Koch (1968) orthocyclic layer-to-layer capacitance, combined to a winding
    capacitance with the energy method of Biela & Kolar (2008), incl. an
    incomplete top layer.  The winding is ASSUMED ideal orthocyclic.

All SI units internally; geometry inputs in mm.

References (full list in README.md):
 [K68]  J. Koch, Valvo Berichte XIV(3), 99-119 (1968)
 [BK08] J. Biela, J.W. Kolar, IEEE Trans. Ind. Appl. 44(1), 223-233 (2008)
 [W28]  H.A. Wheeler, Proc. IRE 16, 1398 (1928)  (multilayer air coil)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import integrate, special

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
RHO_CU_20 = 1.72e-8          # ohm*m, annealed copper @20 degC
ALPHA_CU = 0.00393           # 1/K

# All generated files (plots, CSV) go to <repo>/output, independent of the cwd.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def output_path(path) -> Path:
    """Resolve where a generated file is written.
    A bare file name or relative path is placed under OUTPUT_DIR (created if
    needed); an absolute path is used unchanged."""
    p = Path(path)
    if not p.is_absolute():
        p = OUTPUT_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# Wire
# ----------------------------------------------------------------------------
# Enamel build per side in um -- TYPICAL placeholders, overwrite from datasheet.
INSULATION_PRESETS_UM = {"plain_enamel": 3.8, "heavy_formvar": 6.3, "poly_nylon": 7.5}


def awg_diameter_mm(awg: float) -> float:
    return 0.127 * 92.0 ** ((36.0 - awg) / 39.0)


@dataclass
class Wire:
    d_bare_mm: float                     # bare copper diameter
    ins_um: float                        # insulation thickness PER SIDE
    eps_ins: float = 3.5                 # relative permittivity of enamel

    @classmethod
    def from_awg(cls, awg, insulation="heavy_formvar", eps_ins=3.5, ins_um=None):
        t = INSULATION_PRESETS_UM[insulation] if ins_um is None else ins_um
        return cls(awg_diameter_mm(awg), t, eps_ins)

    @property
    def d_bare(self): return self.d_bare_mm * 1e-3
    @property
    def delta(self): return self.ins_um * 1e-6
    @property
    def d_out(self): return self.d_bare + 2 * self.delta
    @property
    def r_i(self): return self.d_bare / 2
    @property
    def r_0(self): return self.d_out / 2


# ----------------------------------------------------------------------------
# Bobbin / winding geometry
# ----------------------------------------------------------------------------
@dataclass
class Bobbin:
    """Surface on which layer 1 lies (outer surface of the bobbin tube)."""
    shape: str = "rect"                  # 'round' | 'rect'
    diameter_mm: float = 0.0             # round
    width_mm: float = 0.0                # rect: long side
    depth_mm: float = 0.0                # rect: short side
    corner_radius_mm: float = 0.0        # rect: outer corner radius (<= depth/2)
    height_mm: float = 10.0              # axial winding space between flanges
    max_build_mm: Optional[float] = None # optional radial limit (warning only)

    def perimeter0(self):
        if self.shape == "round":
            return math.pi * self.diameter_mm * 1e-3
        W, T, rc = self.width_mm * 1e-3, self.depth_mm * 1e-3, self.corner_radius_mm * 1e-3
        return 2 * (W + T) - (8 - 2 * math.pi) * rc

    def area0(self):
        if self.shape == "round":
            return math.pi * (self.diameter_mm * 1e-3) ** 2 / 4
        W, T, rc = self.width_mm * 1e-3, self.depth_mm * 1e-3, self.corner_radius_mm * 1e-3
        return W * T - (4 - math.pi) * rc ** 2

    # offset path (Steiner): perimeter P0 + 2*pi*s ; area A0 + P0*s + pi*s^2
    def perimeter(self, s): return self.perimeter0() + 2 * math.pi * s
    def area(self, s): return self.area0() + self.perimeter0() * s + math.pi * s * s


@dataclass
class OrthocyclicWinding:
    turns_per_layer: List[int]
    layer_center_offset: List[float]      # s_k: distance of layer-k wire centres from bobbin surface [m]
    layer_perimeter: List[float]          # mean turn length of layer k [m]
    layer_area: List[float]               # area enclosed by layer-k centre line [m^2]
    build: float                          # radial build [m]
    height: float

    @property
    def N(self): return int(sum(self.turns_per_layer))
    @property
    def n_layers(self): return len(self.turns_per_layer)
    @property
    def wire_length(self): return float(np.dot(self.turns_per_layer, self.layer_perimeter))
    @property
    def mean_turn_length(self): return self.wire_length / self.N
    @property
    def mean_offset(self): return float(np.dot(self.turns_per_layer, self.layer_offset_arr) / self.N)
    @property
    def layer_offset_arr(self): return np.array(self.layer_center_offset)
    @property
    def mean_area(self): return float(np.dot(self.turns_per_layer, self.layer_area) / self.N)


def build_orthocyclic(N: int, wire: Wire, bob: Bobbin) -> OrthocyclicWinding:
    """Ideal orthocyclic stacking: in-layer pitch d0, layer pitch sqrt(3)/2*d0,
    alternate layers shifted by d0/2 (one turn fewer if the space is too tight)."""
    d0 = wire.d_out
    H = bob.height_mm * 1e-3
    z_a = int(math.floor(H / d0 + 1e-9))
    z_b = int(math.floor((H - d0 / 2) / d0 + 1e-9))
    if z_a < 2 or z_b < 1:
        raise ValueError("winding height too small for this wire")
    tpl, off, per, area = [], [], [], []
    left, k = N, 0
    pitch = math.sqrt(3) / 2 * d0
    while left > 0:
        z = z_a if k % 2 == 0 else z_b
        n = min(z, left)
        s = d0 / 2 + k * pitch
        tpl.append(n); off.append(s)
        per.append(bob.perimeter(s)); area.append(bob.area(s))
        left -= n; k += 1
    build = d0 + (k - 1) * pitch
    if bob.max_build_mm is not None and build * 1e3 > bob.max_build_mm:
        print(f"WARNING: radial build {build*1e3:.2f} mm exceeds limit {bob.max_build_mm} mm")
    return OrthocyclicWinding(tpl, off, per, area, build, H)


# ----------------------------------------------------------------------------
# R : DC resistance
# ----------------------------------------------------------------------------
def r_dc(wire: Wire, winding: OrthocyclicWinding, T_C: float = 20.0) -> float:
    rho = RHO_CU_20 * (1 + ALPHA_CU * (T_C - 20.0))
    return rho * winding.wire_length / (math.pi * wire.d_bare ** 2 / 4)


# ----------------------------------------------------------------------------
# C : Koch orthocyclic layer capacitance + Biela/Kolar winding combination
# ----------------------------------------------------------------------------
def koch_orthocyclic_factor(r_i: float, delta: float, eps_D: float) -> float:
    """Dimensionless kappa_K with  C0 = eps0 * z * l_wm * kappa_K   (Koch 1968,
    eqs. (11),(12) in Biela & Kolar 2008, no foil between layers).

        C0 = 4 eps0 z l [ M_L + (4 delta r0 - 2 delta^2)/(eps_D (2 r0)^2) * M_D ]
        M_L = int_0^{pi/6} (cos^2 - cos*s - 1/2) / (2 [cos - q (s + 1/2)]^2) dpsi
        M_D = int_0^{pi/6} (sin^2 + cos*s)       / (2 [cos - q (s + 1/2)]^2) dpsi
        s = sqrt(cos^2 psi - 3/4),  q = 1 - delta/(2 eps_D r0),  r0 = r_i + delta

    NOTE: the integrands were transcribed from a PDF text layer in which the
    square roots were lost; the structure was reconstructed and cross-checked
    (touching-wire limit diverges, thumb-rule comparison, see self_test()).
    Verify against Koch/Biela-Kolar before relying on absolute numbers.
    """
    r0 = r_i + delta
    q = 1.0 - delta / (2.0 * eps_D * r0)

    def parts(psi):
        c = math.cos(psi)
        s = math.sqrt(max(c * c - 0.75, 0.0))
        den = 2.0 * (c - q * (s + 0.5)) ** 2
        ml = (c * c - c * s - 0.5) / den
        md = (math.sin(psi) ** 2 + c * s) / den
        return ml, md

    ML = integrate.quad(lambda p: parts(p)[0], 0.0, math.pi / 6, limit=400, epsabs=0, epsrel=1e-9)[0]
    MD = integrate.quad(lambda p: parts(p)[1], 0.0, math.pi / 6, limit=400, epsabs=0, epsrel=1e-9)[0]
    return 4.0 * (ML + (4 * delta * r0 - 2 * delta ** 2) / (eps_D * (2 * r0) ** 2) * MD)


def winding_capacitance(wire: Wire, w: OrthocyclicWinding, scheme: str = "standard") -> dict:
    """Energy-equivalent capacitance of the whole winding.

    scheme = 'standard' : layers wound back and forth, C_layer = C0/3   [BK08 III-A]
             'flyback'  : all layers same direction,      C_layer = C0/4   [BK08 III-B]
    Layer pair (i,i+1): C0_i = eps0 * z_eff * l_wm * kappa_K, z_eff = turns of the
    shorter layer (BK08 eq.(25) scaling for incomplete layers), l_wm = mean of the
    two layer turn lengths.  Transformed to the terminals with BK08 eq.(24):
        C_wdg = sum_i (N_i + N_{i+1})^2 / (sum_k N_k)^2 * C_layer,i
    """
    kap = koch_orthocyclic_factor(wire.r_i, wire.delta, wire.eps_ins)
    div = {"standard": 3.0, "flyback": 4.0}[scheme]
    n = np.array(w.turns_per_layer, float)
    per = np.array(w.layer_perimeter)
    Ntot = n.sum()
    C = 0.0
    c0_list = []
    for i in range(len(n) - 1):
        z_eff = min(n[i], n[i + 1])
        l_wm = 0.5 * (per[i] + per[i + 1])
        C0 = EPS0 * z_eff * l_wm * kap
        c0_list.append(C0)
        C += (n[i] + n[i + 1]) ** 2 / Ntot ** 2 * C0 / div
    return {"C_winding": C, "kappa_K": kap, "C0_layers": c0_list, "scheme": scheme}


# ----------------------------------------------------------------------------
# L : air coil + core (poles, slugs, magnet stack)
# ----------------------------------------------------------------------------
def wheeler_multilayer_L(N, a_mean, length, build) -> float:
    """Wheeler 1928 multilayer round coil. a, l, b in metres -> H.
    L[uH] = 0.8 a^2 N^2 / (6a + 9l + 10b), a,l,b in inches."""
    inch = 0.0254
    a, l, b = a_mean / inch, length / inch, build / inch
    return 0.8 * a * a * N * N / (6 * a + 9 * l + 10 * b) * 1e-6


def demag_ellipsoid(m: float) -> float:
    """Prolate ellipsoid, m = length/diameter (axial demag factor)."""
    if m <= 1.0 + 1e-9:
        return 1.0 / 3.0 if m >= 1 - 1e-9 else _oblate(m)
    e = math.sqrt(m * m - 1)
    return (m / e * math.log(m + e) - 1.0) / (m * m - 1.0)


def _oblate(m):
    # oblate spheroid (m<1)
    e = math.sqrt(1 - m * m)
    return (1.0 / (1 - m * m)) * (1 - m / e * math.asin(e))


def demag_cylinder_uniform(m: float) -> float:
    """Magnetometric (volume averaged) axial demag factor of a UNIFORMLY
    magnetised circular cylinder, m = length/diameter.  Surface-charge energy:
        N = [I(0) - I(l)] / (2 pi V),  I(l) = 4 pi^2 a^2 int J1(ka)^2 e^{-kl}/k^2 dk
    (I(0) = 16 pi a^3 / 3).  For high-mu rods the real midplane (fluxmetric)
    factor is smaller -- see Ripka et al., JMMM 2021 -- so treat as upper bound."""
    a, l = 0.5, m                      # diameter 1
    I0 = 16 * math.pi * a ** 3 / 3
    f = lambda k: (special.j1(k * a) ** 2) * math.exp(-k * l) / k ** 2 if k > 0 else 0.25 * a * a
    Il = 4 * math.pi ** 2 * a * a * integrate.quad(f, 0, np.inf, limit=500)[0]
    V = math.pi * a * a * l
    return (I0 - Il) / (2 * math.pi * V)


MATERIALS_MUR = {                       # small-signal / incremental, TYPICAL -- adjust!
    "air": 1.0, "nonmagnetic": 1.0, "nickel_silver": 1.0, "brass": 1.0,
    "steel_1010": 200.0, "iron_soft": 300.0, "stainless_430": 200.0,
    "alnico2": 5.0, "alnico3": 5.0, "alnico5": 4.0, "alnico8": 4.5,
    "ceramic_ferrite": 1.1, "neodymium": 1.05, "samarium_cobalt": 1.05,
}


@dataclass
class Segment:
    """One piece of a core stack, along the pickup axis."""
    material: str | float              # key of MATERIALS_MUR or a number mu_r
    length_mm: float
    area_mm2: float = 0.0              # cross-section; 0 -> use stack default
    in_coil: bool = True               # True: lies inside the coil height

    @property
    def mu_r(self):
        return MATERIALS_MUR[self.material] if isinstance(self.material, str) else float(self.material)


@dataclass
class CoreStack:
    """A column of segments on the pickup axis, repeated n times under ONE coil
    (e.g. 6 AlNiCo rods of a Strat; 6 steel slugs + bar magnet below)."""
    segments: List[Segment]
    area_mm2: float                    # default cross-section of segments (mm^2)
    n: int = 1
    shared: bool = False               # True: stack is a single body shared by all n poles
                                       # (e.g. one bar magnet under all slugs) -> contributes
                                       # only via the series path of each pole (not counted again)

    def composite(self):
        """series-reluctance composite along the stack; returns
        (mu_comp, l_tot, A_ref, D_eq) referenced to the in-coil cross-section."""
        A = [(s.area_mm2 or self.area_mm2) * 1e-6 for s in self.segments]
        l = [s.length_mm * 1e-3 for s in self.segments]
        mu = [s.mu_r for s in self.segments]
        inc = [s.in_coil for s in self.segments]
        A_ref = (sum(a * li for a, li, c in zip(A, l, inc) if c) /
                 max(sum(li for li, c in zip(l, inc) if c), 1e-12))
        R = sum(li / (m_ * a) for li, m_, a in zip(l, mu, A))
        l_tot = sum(l)
        mu_comp = l_tot / (A_ref * R)
        D_eq = math.sqrt(4 * A_ref / math.pi)
        return mu_comp, l_tot, A_ref, D_eq


def core_effective_mu(stack: CoreStack, demag: str = "ellipsoid") -> dict:
    mu_c, l_tot, A_ref, D_eq = stack.composite()
    m = l_tot / D_eq
    Nd = demag_ellipsoid(m) if demag == "ellipsoid" else demag_cylinder_uniform(m)
    mu_eff = mu_c / (1 + Nd * (mu_c - 1))
    return {"mu_comp": mu_c, "m": m, "Nd": Nd, "mu_eff": mu_eff, "A_ref": A_ref, "l_tot": l_tot}


def inductance(N: int, bob: Bobbin, w: OrthocyclicWinding, stacks: List[CoreStack],
               kappa: float = 0.85, demag: str = "ellipsoid") -> dict:
    """L = L_air + kappa * sum_stacks n * mu0 (mu_eff-1) A_ref N^2 / l_ref
    l_ref = sqrt(l_coil^2 + D_coil^2): centre field of a finite solenoid
    (D_coil = 2*a_eq, a_eq from the area enclosed by the mean turn).
    kappa : single empirical factor (field averaging, oblong coil, leakage);
            calibrate on ONE measured pickup (see calibrate_kappa)."""
    s_mean = w.mean_offset
    A_m = bob.area(s_mean)
    a_eq = math.sqrt(A_m / math.pi)
    L_air = wheeler_multilayer_L(N, a_eq, w.height, w.build)
    l_ref = math.sqrt(w.height ** 2 + (2 * a_eq) ** 2)
    L_core, info = 0.0, []
    for st in stacks:
        ce = core_effective_mu(st, demag)
        n_eff = 0 if st.shared else st.n
        dL = kappa * n_eff * MU0 * (ce["mu_eff"] - 1) * ce["A_ref"] * N ** 2 / l_ref
        L_core += dL
        info.append({**ce, "dL": dL})
    return {"L": L_air + L_core, "L_air": L_air, "L_core": L_core, "a_eq": a_eq,
            "l_ref": l_ref, "stacks": info}


def calibrate_kappa(L_meas, N, bob, w, stacks, demag="ellipsoid") -> float:
    r0 = inductance(N, bob, w, stacks, kappa=1.0, demag=demag)
    return (L_meas - r0["L_air"]) / r0["L_core"]


# ----------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------
@dataclass
class PickupSpec:
    name: str
    N: int
    wire: Wire
    bobbin: Bobbin
    stacks: List[CoreStack]
    scheme: str = "standard"
    kappa: float = 0.85
    demag: str = "ellipsoid"
    C_extra_pF: float = 0.0            # lumped extra (baseplate/cover/leads), user input
    T_C: float = 20.0
    air_core: bool = False             # True: ignore all core stacks -> pure air coil (L = Wheeler only)

    @property
    def active_stacks(self) -> List["CoreStack"]:
        return [] if self.air_core else self.stacks


def as_air_coil(p: "PickupSpec") -> "PickupSpec":
    """Same winding and bobbin, magnet/pole pieces removed."""
    return replace(p, air_core=True, name=p.name + " (air core)")


def analyse(p: PickupSpec) -> dict:
    w = build_orthocyclic(p.N, p.wire, p.bobbin)
    R = r_dc(p.wire, w, p.T_C)
    Lr = inductance(p.N, p.bobbin, w, p.active_stacks, p.kappa, p.demag)
    Cr = winding_capacitance(p.wire, w, p.scheme)
    C = Cr["C_winding"] + p.C_extra_pF * 1e-12
    f0 = 1 / (2 * math.pi * math.sqrt(Lr["L"] * C))
    return {"winding": w, "R": R, "L": Lr, "C": Cr, "C_total": C, "f0": f0}


def report(p: PickupSpec) -> str:
    r = analyse(p); w = r["winding"]; L = r["L"]; C = r["C"]
    out = [f"=== {p.name} ===",
           f"wire: bare {p.wire.d_bare_mm*1e3:.1f} um, OD {p.wire.d_out*1e6:.1f} um, eps_ins {p.wire.eps_ins}",
           f"winding: N={w.N}, layers={w.n_layers}, turns/layer {w.turns_per_layer[0]}/{w.turns_per_layer[1] if w.n_layers>1 else '-'}"
           f", build {w.build*1e3:.2f} mm, mean turn {w.mean_turn_length*1e3:.1f} mm, wire {w.wire_length:.0f} m",
           f"R_dc = {r['R']/1e3:.2f} kOhm  (T={p.T_C:.0f} C)",
           f"L    = {L['L']:.3f} H  (air {L['L_air']:.3f} + core {L['L_core']:.3f}; kappa={p.kappa})"
           + ("  [AIR CORE: core stacks ignored]" if p.air_core else "")]
    for i, s in enumerate(L["stacks"]):
        out.append(f"   stack {i}: mu_comp {s['mu_comp']:.2f}, l/d {s['m']:.2f}, N_d {s['Nd']:.4f}, "
                   f"mu_eff {s['mu_eff']:.2f}, dL {s['dL']:.3f} H")
    out += [f"C_wdg= {C['C_winding']*1e12:.1f} pF  (Koch kappa_K {C['kappa_K']:.3f}, scheme {C['scheme']}); "
            f"C_total {r['C_total']*1e12:.1f} pF",
            f"f0   = {r['f0']/1e3:.2f} kHz  (L, C_total only, no load)"]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Sweep over the number of turns N
# ----------------------------------------------------------------------------
def sweep_N(base: PickupSpec, N_values) -> List[dict]:
    """Evaluate R, L, C, f0 for every N in N_values (all other inputs of `base`
    unchanged).  Winding layers, mean turn length and build are recomputed for
    each N, so L(N), C(N), R(N) include the geometric effects, not just N^2."""
    rows = []
    for n in N_values:
        r = analyse(replace(base, N=int(n)))
        w, L, C = r["winding"], r["L"], r["C"]
        rows.append({"N": int(n), "layers": w.n_layers, "build_mm": w.build * 1e3,
                     "l_turn_mm": w.mean_turn_length * 1e3, "wire_m": w.wire_length,
                     "R_kohm": r["R"] / 1e3, "L_H": L["L"], "L_air_H": L["L_air"],
                     "L_core_H": L["L_core"], "C_wdg_pF": C["C_winding"] * 1e12,
                     "C_total_pF": r["C_total"] * 1e12, "f0_kHz": r["f0"] / 1e3})
    return rows


def format_sweep(rows: List[dict]) -> str:
    hdr = (f"{'N':>6} {'lay':>4} {'build/mm':>8} {'R/kOhm':>7} {'L/H':>7} {'L_air':>6} "
           f"{'L_core':>6} {'C_wdg/pF':>8} {'C_tot/pF':>8} {'f0/kHz':>7}")
    lines = [hdr]
    for r in rows:
        lines.append(f"{r['N']:>6d} {r['layers']:>4d} {r['build_mm']:>8.2f} {r['R_kohm']:>7.2f} "
                     f"{r['L_H']:>7.3f} {r['L_air_H']:>6.3f} {r['L_core_H']:>6.3f} "
                     f"{r['C_wdg_pF']:>8.1f} {r['C_total_pF']:>8.1f} {r['f0_kHz']:>7.2f}")
    return "\n".join(lines)


def save_sweep_csv(rows: List[dict], path: str):
    import csv
    path = output_path(path)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def plot_sweep(rows: List[dict], path: str, title: str = ""):
    """R, L, C, f0 versus N (needs matplotlib; skipped silently if missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - plot skipped"); return
    N = [r["N"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for a, key, lab in [(ax[0, 0], "R_kohm", "R_dc [kOhm]"), (ax[0, 1], "L_H", "L [H]"),
                        (ax[1, 0], "C_total_pF", "C [pF]"), (ax[1, 1], "f0_kHz", "f0 [kHz]")]:
        a.plot(N, [r[key] for r in rows], "o-"); a.set_ylabel(lab); a.grid(alpha=.3)
    ax[1, 0].set_xlabel("turns N"); ax[1, 1].set_xlabel("turns N")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(output_path(path), dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Self tests / sanity checks
# ----------------------------------------------------------------------------
def self_test():
    print("--- self test ---")
    # demag: sphere, long-rod limit, thin disk
    print("N_d ellipsoid m=1:", round(demag_ellipsoid(1.0), 4), "(1/3)")
    print("N_d ellipsoid m=10:", round(demag_ellipsoid(10), 4), "(0.0203)")
    print("N_d cyl-uniform m=0.01:", round(demag_cylinder_uniform(0.01), 3), "(->1)")
    print("N_d cyl-uniform m=1:", round(demag_cylinder_uniform(1.0), 4), "(compare with Chen et al. 2002 table - not verified here)")
    # Koch factor vs. thumb rule [BK08 eq.(6)]: C0 ~ 60 pF/m * l_wm * z for thin wires
    for awg, ins in [(42, "heavy_formvar"), (43, "heavy_formvar"), (42, "plain_enamel")]:
        wr = Wire.from_awg(awg, ins)
        k = koch_orthocyclic_factor(wr.r_i, wr.delta, wr.eps_ins)
        print(f"Koch AWG{awg} {ins}: kappa_K={k:.2f} -> C0/(z*l) = {EPS0*k*1e12:.1f} pF/m  (thumb rule 60 pF/m is a mix of orthogonal+orthocyclic; orthocyclic should be above it)")
    # Wheeler vs known: 100 turns, a=10mm, l=10mm, b=5mm
    print("Wheeler 100t a10 l10 b5 mm [uH]:", round(wheeler_multilayer_L(100, .01, .01, .005)*1e6, 2))


# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------
def example_strat() -> PickupSpec:
    rod = Segment("alnico5", 19.0, area_mm2=math.pi * 4.8 ** 2 / 4)
    return PickupSpec(
        name="Strat-style, 6 AlNiCo5 rods, AWG42 heavy formvar",
        N=8000, wire=Wire.from_awg(42, "heavy_formvar"),
        bobbin=Bobbin("rect", width_mm=52, depth_mm=6.5, corner_radius_mm=3.0, height_mm=12.5, max_build_mm=6.0),
        stacks=[CoreStack([rod], area_mm2=math.pi * 4.8 ** 2 / 4, n=6)],
        kappa=0.85, C_extra_pF=0.0)


def example_humbucker_coil() -> PickupSpec:
    """One coil of a humbucker: 6 steel slugs; bar magnet (ceramic) lies BELOW the
    coil (in_coil=False) and extends each slug magnetically (series path)."""
    slug = Segment("steel_1010", 14.0, area_mm2=math.pi * 4.8 ** 2 / 4, in_coil=True)
    magnet = Segment("ceramic_ferrite", 8.0, area_mm2=math.pi * 4.8 ** 2 / 4, in_coil=False)
    return PickupSpec(
        name="Humbucker coil, 6 steel slugs + ceramic bar magnet below",
        N=5000, wire=Wire.from_awg(43, "plain_enamel"),
        bobbin=Bobbin("rect", width_mm=70, depth_mm=7.5, corner_radius_mm=3.5, height_mm=9.0, max_build_mm=7.0),
        stacks=[CoreStack([slug, magnet], area_mm2=math.pi * 4.8 ** 2 / 4, n=6)],
        kappa=0.85)
