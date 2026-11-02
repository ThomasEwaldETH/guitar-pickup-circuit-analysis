"""
pickup_network.py -- frequency response (Bode) of networks built from pickup coils

Circuit stage after pickup_rlc.py:
  * each coil from pickup_rlc.analyse() becomes a reusable element
        EMF --- R_dc --- L ---+--- terminal +
         |                    C_total
        terminal - -----------+
    i.e. the induced voltage sits in series with R and L, C is across the terminals.
  * coils + extra R, C, L are combined in a small netlist
  * the netlist is solved by modified nodal analysis (MNA) with numpy, one
    complex linear solve per frequency -> H(jw) = V_out / EMF
  * Bode plot (magnitude in dB, phase in degrees)

Only numpy + matplotlib are needed.  lcapy is used optionally (cross_check_lcapy)
to get the same transfer function symbolically and to verify the numeric solver.

Node names are strings, ground is "0".
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import pickup_rlc as prl


# ----------------------------------------------------------------------------
# Coil element from pickup_rlc
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CoilRLC:
    R: float          # ohm
    L: float          # H
    C: float          # F (winding + extra)
    name: str = "coil"

    @classmethod
    def from_spec(cls, spec: prl.PickupSpec) -> "CoilRLC":
        r = prl.analyse(spec)
        return cls(R=r["R"], L=r["L"]["L"], C=r["C_total"], name=spec.name)

    @property
    def f0(self) -> float:
        return 1 / (2 * math.pi * math.sqrt(self.L * self.C))


# ----------------------------------------------------------------------------
# Netlist + MNA solver
# ----------------------------------------------------------------------------
@dataclass
class Network:
    elements: List[tuple] = field(default_factory=list)   # (kind, name, n+, n-, value)
    _n_int: int = 0

    # passive two-terminal elements
    def R(self, name, a, b, value): self.elements.append(("R", name, a, b, float(value))); return self
    def C(self, name, a, b, value): self.elements.append(("C", name, a, b, float(value))); return self
    def L(self, name, a, b, value): self.elements.append(("L", name, a, b, float(value))); return self

    def V(self, name, a, b, amplitude: complex = 1.0):
        """independent voltage source, V(a) - V(b) = amplitude"""
        self.elements.append(("V", name, a, b, complex(amplitude))); return self

    def coil(self, name: str, plus: str, minus: str, c: CoilRLC, emf: complex = 1.0):
        """Insert a pickup coil between `plus` and `minus`.
        emf = relative complex amplitude of the induced voltage (1 = in phase,
        -1 = reversed winding/magnet polarity, 0 = passive coil)."""
        self._n_int += 1
        x1, x2 = f"{name}.e", f"{name}.r"
        self.V(f"{name}.E", x1, minus, emf)       # EMF in series ...
        self.R(f"{name}.R", x1, x2, c.R)          # ... with R_dc ...
        self.L(f"{name}.L", x2, plus, c.L)        # ... and L
        self.C(f"{name}.C", plus, minus, c.C)     # winding capacitance across terminals
        return self

    # ------------------------------------------------------------------
    def nodes(self) -> List[str]:
        ns = []
        for _, _, a, b, _ in self.elements:
            for n in (a, b):
                if n != "0" and n not in ns:
                    ns.append(n)
        return ns

    def solve(self, f: Sequence[float], out: str, ref: str = "0",
              sources: bool = True, inject: Optional[Tuple[str, str]] = None) -> np.ndarray:
        """Complex V(out) - V(ref) for every frequency.
        sources=True : all voltage sources active (superposition is automatic).
        sources=False: all voltage sources set to 0 V (shorted).
        inject=(a, b): additionally a 1 A test current flowing into node a, out of b."""
        f = np.atleast_1d(np.asarray(f, float))
        nodes = self.nodes()
        idx = {n: i for i, n in enumerate(nodes)}
        vsrc = [e for e in self.elements if e[0] == "V"]
        n, m = len(nodes), len(vsrc)
        N = n + m
        s = 2j * np.pi * f                                   # (F,)
        A = np.zeros((len(f), N, N), complex)
        z = np.zeros(N, complex)

        def stamp_y(a, b, y):
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None: A[:, ia, ia] += y
            if ib is not None: A[:, ib, ib] += y
            if ia is not None and ib is not None:
                A[:, ia, ib] -= y; A[:, ib, ia] -= y

        k = 0
        for kind, name, a, b, v in self.elements:
            if kind == "R":   stamp_y(a, b, 1.0 / v)
            elif kind == "C": stamp_y(a, b, s * v)
            elif kind == "L": stamp_y(a, b, 1.0 / (s * v))
            elif kind == "V":
                row = n + k; k += 1
                ia, ib = idx.get(a), idx.get(b)
                if ia is not None: A[:, ia, row] += 1; A[:, row, ia] += 1
                if ib is not None: A[:, ib, row] -= 1; A[:, row, ib] -= 1
                z[row] = v if sources else 0
        if inject is not None:
            ia, ib = idx.get(inject[0]), idx.get(inject[1])
            if ia is not None: z[ia] += 1
            if ib is not None: z[ib] -= 1
        x = np.linalg.solve(A, np.broadcast_to(z, (len(f), N))[..., None])[..., 0]
        vo = x[:, idx[out]] if out != "0" else 0
        vr = x[:, idx[ref]] if ref != "0" else 0
        return vo - vr

    def impedance(self, f: Sequence[float], node: str, ref: str = "0") -> np.ndarray:
        """Impedance seen between `node` and `ref` with all sources shorted:
        Z = V / I for a 1 A test current (no induced voltage needed)."""
        return self.solve(f, node, ref, sources=False, inject=(node, ref))


# ----------------------------------------------------------------------------
# Bode helpers
# ----------------------------------------------------------------------------
def bode_data(H: np.ndarray):
    return 20 * np.log10(np.abs(H)), np.unwrap(np.angle(H)) * 180 / np.pi


def peak(f, H) -> Tuple[float, float]:
    i = int(np.argmax(np.abs(H)))
    return f[i], 20 * np.log10(abs(H[i]))


def plot_bode(f, curves: Dict[str, np.ndarray], path: str, title: str = "",
              ylabel: str = "|H| = |V_out / EMF|  [dB]"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (am, ap) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for lab, H in curves.items():
        mag, ph = bode_data(H)
        fp, mp = peak(f, H)
        am.semilogx(f, mag, label=f"{lab}  (peak {fp/1e3:.2f} kHz, {mp:+.1f} dB)")
        ap.semilogx(f, ph)
    am.set_ylabel(ylabel); am.grid(which="both", alpha=.3)
    ap.set_ylabel("phase [deg]"); ap.set_xlabel("frequency [Hz]"); ap.grid(which="both", alpha=.3)
    am.legend(fontsize=8, loc="lower left")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(prl.output_path(path), dpi=140)
    plt.close(fig)


def plot_impedance(f, curves: Dict[str, np.ndarray], path: str, title: str = ""):
    """|Z| in kOhm (log axis) and phase of Z, one curve per network."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (am, ap) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for lab, Z in curves.items():
        i = int(np.argmax(np.abs(Z)))
        am.loglog(f, np.abs(Z) / 1e3, label=f"{lab}  (max {abs(Z[i])/1e3:.0f} kOhm at {f[i]/1e3:.2f} kHz)")
        ap.semilogx(f, np.angle(Z, deg=True))
    am.set_ylabel("|Z_in|  [kOhm]"); am.grid(which="both", alpha=.3)
    ap.set_ylabel("phase of Z_in [deg]"); ap.set_xlabel("frequency [Hz]"); ap.grid(which="both", alpha=.3)
    ap.set_yticks([-90, -45, 0, 45, 90])
    am.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(prl.output_path(path), dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Example networks
# ----------------------------------------------------------------------------
def load_chain(net: Network, top: str, pot=250e3, tone_pot=250e3, tone_cap=22e-9,
               tone_pos=1.0, cable_C=500e-12, amp_R=1e6, amp_C=0.0):
    """Typical passive guitar load at node `top` (volume fully up):
    volume pot to ground, tone pot (as rheostat at `tone_pos`) + tone cap to ground,
    cable capacitance and amplifier input."""
    net.R("Rvol", top, "0", pot)
    if tone_pos is not None:
        net.R("Rtone", top, "tn", max(tone_pot * tone_pos, 1e-3))
        net.C("Ctone", "tn", "0", tone_cap)
    net.C("Ccable", top, "0", cable_C)
    net.R("Ramp", top, "0", amp_R)
    if amp_C:
        net.C("Camp", top, "0", amp_C)
    return net


def ex_single(coil: CoilRLC, **load):
    n = Network().coil("P1", "out", "0", coil)
    return load_chain(n, "out", **load)


def ex_humbucker(coil: CoilRLC, mode="series", **load):
    """Two identical coils, EMFs in phase for the string signal (reverse wound,
    reverse polarity -> string signal adds, hum cancels)."""
    n = Network()
    if mode == "series":
        n.coil("A", "out", "mid", coil).coil("B", "mid", "0", coil)
    elif mode == "parallel":
        n.coil("A", "out", "0", coil).coil("B", "out", "0", coil)
    else:
        raise ValueError(mode)
    return load_chain(n, "out", **load)


def ex_n_series(coil: CoilRLC, n_coils: int, **load):
    """n identical coils in series (generalises the humbucker; e.g. 'quad rail')."""
    n = Network()
    nodes = ["out"] + [f"k{i}" for i in range(1, n_coils)] + ["0"]
    for i in range(n_coils):
        n.coil(f"P{i+1}", nodes[i], nodes[i + 1], coil)
    return load_chain(n, "out", **load)


# ----------------------------------------------------------------------------
# Optional: symbolic cross check with lcapy
# ----------------------------------------------------------------------------
def cross_check_lcapy(coil: CoilRLC, f, pot=500e3, cable_C=500e-12, amp_R=1e6):
    """Series humbucker into pot || cable C || amp R, solved symbolically by lcapy.
    Same topology/polarity as ex_humbucker(..., 'series', tone_pos=None).
    Returns (H(s) symbolic with component names, H(j2pi f) numeric, max rel. error vs MNA)."""
    import lcapy
    net = lcapy.Circuit("""
V1 a mid s 1
RA a b RA
LA b out LA
CA out mid CA
V2 c 0 s 1
RB c d RB
LB d mid LB
CB mid 0 CB
Rv out 0 Rv
Cc out 0 Cc
Ra out 0 Ra
""")
    H = net.out.V(lcapy.s)                       # V_out(s) for unit EMFs -> transfer function
    vals = dict(RA=coil.R, RB=coil.R, LA=coil.L, LB=coil.L, CA=coil.C, CB=coil.C,
                Rv=pot, Cc=cable_C, Ra=amp_R)
    Hn = H.subs(vals)
    import sympy
    fn = sympy.lambdify(lcapy.s.sympy, Hn.sympy, "numpy")
    Hjw = np.asarray(fn(2j * np.pi * np.asarray(f)), complex)
    mna = ex_humbucker(coil, "series", pot=pot, tone_pos=None, cable_C=cable_C, amp_R=amp_R) \
        .solve(f, "out")
    err = float(np.max(np.abs(Hjw - mna) / np.abs(mna)))
    return H, Hjw, err
