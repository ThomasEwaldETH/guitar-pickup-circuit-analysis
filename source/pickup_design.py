"""
pickup_design.py -- one-step design: pickup geometry -> network -> Bode plot

    geometry (PickupSpec)  ->  R, L, C per coil (pickup_rlc.analyse)
    topology (Series / Parallel of Pickups and passive parts)
    output chain (shunt and series filter stages, cable, pots, amp)
    ->  H(jf) = V_out / EMF  ->  Bode plot

Building blocks (all two-terminal):
    Pickup(spec, emf=+1)        one coil from geometry; emf=-1 reverses polarity;
                                air_core=True drops magnet/pole pieces (air coil)
    R(v), L(v), C(v)            single parts
    RLC(R=, L=, C=, mode=)      R, L, C combined in "series" or "parallel"
    Series(a, b, ...)           blocks in series
    Parallel(a, b, ...)         blocks in parallel

A Design is a source block (containing the pickups, between node "out0" and
ground) followed by an output chain of stages:
    Shunt(block)                block from the current node to ground
    Through(block)              block in series, creates a new node
Helpers: cable(), volume_pot(), tone_control(), amp_input().

Usage examples: see main.py.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import pickup_rlc as prl
from pickup_network import CoilRLC, Network, plot_bode, plot_impedance, peak


# ----------------------------------------------------------------------------
# Two-terminal blocks
# ----------------------------------------------------------------------------
class Block:
    def stamp(self, net: Network, a: str, b: str, ids) -> None:
        raise NotImplementedError


class Ctx:
    """Counter for unique element names + the excitation being solved."""
    def __init__(self, excitation="string"):
        self._c = itertools.count(1); self.excitation = excitation
    def __iter__(self): return self
    def __next__(self): return next(self._c)


def hum_area_turns(spec: prl.PickupSpec, air_core: bool = None) -> float:
    """Effective area-turns N*A_eff [m^2] of a coil for a UNIFORM external
    (hum) field: EMF_hum = N*A_eff * dB/dt.
    A_eff = mean winding area + flux concentrated by the core rods:
            sum over stacks of n * (mu_eff - 1) * A_rod   (demag-corrected mu_eff).
    Estimate only: real hum fields are not uniform and the rod field is not
    uniform along the coil."""
    air = spec.air_core if air_core is None else air_core
    w = prl.build_orthocyclic(spec.N, spec.wire, spec.bobbin)
    A = spec.bobbin.area(w.mean_offset)
    if not air:
        for st in spec.stacks:
            if st.shared:
                continue
            ce = prl.core_effective_mu(st, spec.demag)
            A += st.n * (ce["mu_eff"] - 1) * ce["A_ref"]
    return spec.N * A


@dataclass
class Pickup(Block):
    """One coil.  Two excitations, solved separately (superposition):
      emf : string signal, relative EMF (1 = normal pickup, 0 = no magnet / dummy)
      hum : polarity of the hum EMF (+1 / -1, 0 = off).  The hum EMF itself is
            hum * N*A_eff (V per T/s), so coils of different size weigh correctly.
    A hum-cancelling dummy coil:  Pickup(spec, emf=0, hum=-1, air_core=True)."""
    spec: prl.PickupSpec
    emf: complex = 1.0
    name: Optional[str] = None
    air_core: Optional[bool] = None    # None: use spec.air_core; True/False overrides it
    hum: float = 1.0
    hum_scale: float = 1.0             # optional correction factor on N*A_eff (calibration)
    _rlc: Optional[CoilRLC] = field(default=None, repr=False)
    _na: Optional[float] = field(default=None, repr=False)

    @property
    def _spec(self):
        return self.spec if self.air_core is None else replace(self.spec, air_core=self.air_core)

    @property
    def rlc(self) -> CoilRLC:
        if self._rlc is None:
            self._rlc = CoilRLC.from_spec(self._spec)
        return self._rlc

    @property
    def area_turns(self) -> float:
        if self._na is None:
            self._na = hum_area_turns(self._spec) * self.hum_scale
        return self._na

    def source_amplitude(self, excitation: str) -> complex:
        if excitation == "string":
            return self.emf
        if excitation == "hum":
            return self.hum * self.area_turns
        raise ValueError(excitation)

    def stamp(self, net, a, b, ctx):
        exc = getattr(ctx, "excitation", "string")
        net.coil(self.name or f"P{next(ctx)}", a, b, self.rlc, self.source_amplitude(exc))


@dataclass
class R(Block):
    value: float
    def stamp(self, net, a, b, ids): net.R(f"R{next(ids)}", a, b, self.value)


@dataclass
class L(Block):
    value: float
    def stamp(self, net, a, b, ids): net.L(f"L{next(ids)}", a, b, self.value)


@dataclass
class C(Block):
    value: float
    def stamp(self, net, a, b, ids): net.C(f"C{next(ids)}", a, b, self.value)


class Series(Block):
    def __init__(self, *items: Block): self.items = items

    def stamp(self, net, a, b, ids):
        nodes = [a] + [f"n{next(ids)}" for _ in self.items[1:]] + [b]
        for it, x, y in zip(self.items, nodes[:-1], nodes[1:]):
            it.stamp(net, x, y, ids)


class Parallel(Block):
    def __init__(self, *items: Block): self.items = items

    def stamp(self, net, a, b, ids):
        for it in self.items:
            it.stamp(net, a, b, ids)


def RLC(R_: float = None, L_: float = None, C_: float = None, mode: str = "parallel") -> Block:
    """R, L and/or C combined in series or in parallel (None = part not present)."""
    parts = [p for p in (R(R_) if R_ else None, L(L_) if L_ else None, C(C_) if C_ else None) if p]
    return Parallel(*parts) if mode == "parallel" else Series(*parts)


# ----------------------------------------------------------------------------
# Output chain
# ----------------------------------------------------------------------------
@dataclass
class Shunt:
    block: Block          # from current node to ground


@dataclass
class Through:
    block: Block          # in series, current node -> new node


def cable(C_pF=500.0, R_ohm=0.0) -> List:
    """Guitar cable as lumped C (and optional series R)."""
    st = [Through(R(R_ohm))] if R_ohm else []
    return st + [Shunt(C(C_pF * 1e-12))]


def volume_pot(R_kohm=250.0) -> List:            # fully up: whole track to ground
    return [Shunt(R(R_kohm * 1e3))]


def tone_control(R_kohm=250.0, C_nF=22.0, position=1.0) -> List:
    """position 1 = tone fully open (whole pot in series with cap), 0 = closed."""
    return [Shunt(Series(R(max(R_kohm * 1e3 * position, 1.0)), C(C_nF * 1e-9)))]


@dataclass
class AmpShunt(Shunt):
    """Amplifier input; left out when computing the impedance the amp sees."""


def amp_input(R_Mohm=1.0, C_pF=0.0) -> List:
    st = [AmpShunt(R(R_Mohm * 1e6))]
    return st + ([AmpShunt(C(C_pF * 1e-12))] if C_pF else [])


# ----------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------
@dataclass
class Design:
    source: Block                          # pickups (and parts), between "out0" and ground
    chain: List = field(default_factory=list)   # Shunt / Through stages in signal order
    name: str = "design"

    def network(self, excitation: str = "string", with_amp: bool = True) -> (Network, str):
        """excitation 'string': H = V_out / EMF of one normal coil (dimensionless)
                      'hum'   : H = V_out / (dB/dt) of a uniform hum field [V/(T/s)]"""
        net, ids = Network(), Ctx(excitation)
        self.source.stamp(net, "out0", "0", ids)
        node, k = "out0", 0
        for st in self.chain:
            if isinstance(st, AmpShunt) and not with_amp:
                continue
            if isinstance(st, Shunt):
                st.block.stamp(net, node, "0", ids)
            elif isinstance(st, Through):
                k += 1
                new = f"out{k}"
                st.block.stamp(net, node, new, ids)
                node = new
            else:
                raise TypeError(st)
        return net, node

    def response(self, f, excitation: str = "string") -> np.ndarray:
        net, out = self.network(excitation)
        return net.solve(f, out)

    def impedance(self, f, with_amp: bool = False) -> np.ndarray:
        """Input impedance at the output jack: what the amplifier sees looking
        into the guitar.  All induced voltages are zero (sources shorted), so no
        EMF is needed.  with_amp=True includes the amp input itself."""
        net, out = self.network("string", with_amp)
        return net.impedance(f, out)

    def pickups(self) -> List[Pickup]:
        found = []
        def walk(b):
            if isinstance(b, Pickup): found.append(b)
            for it in getattr(b, "items", ()): walk(it)
        walk(self.source)
        return found

    def summary(self) -> str:
        lines = [f"== {self.name} =="]
        for p in self.pickups():
            c = p.rlc
            air = p.spec.air_core if p.air_core is None else p.air_core
            lines.append(f"  pickup '{p.spec.name}'{' [air core]' if air else ''} emf={p.emf:+} hum={p.hum:+} "
                         f"N*A_eff={p.area_turns:.2f} m^2: R={c.R/1e3:.2f} kOhm, "
                         f"L={c.L:.3f} H, C={c.C*1e12:.1f} pF, f0={c.f0/1e3:.2f} kHz")
        return "\n".join(lines)


def compare(designs: Sequence[Design], f=None, path="bode_design.png", title="",
            excitation: str = "string"):
    """Solve and plot several designs in one Bode plot; returns {name: H}."""
    f = np.logspace(1, 5, 2000) if f is None else f
    curves = {}
    for d in designs:
        print(d.summary())
        H = d.response(f, excitation)
        fp, mp = peak(f, H)
        print(f"  -> peak {fp/1e3:.2f} kHz, {mp:+.1f} dB")
        curves[d.name] = H
    ylab = "|V_out / EMF|  [dB]" if excitation == "string" else "|V_out / (dB/dt)|  [dB re 1 V/(T/s)]"
    plot_bode(f, curves, path, title, ylabel=ylab)
    return f, curves


def compare_impedance(designs: Sequence[Design], f=None, path="impedance.png", title="",
                      with_amp: bool = False):
    """Input impedance of several designs in one plot; returns (f, {name: Z})."""
    f = np.logspace(1, 5, 2000) if f is None else f
    curves = {}
    for d in designs:
        Z = d.impedance(f, with_amp)
        i = int(np.argmax(np.abs(Z)))
        print(f"  {d.name:44s} |Z| max {abs(Z[i])/1e3:7.1f} kOhm at {f[i]/1e3:6.2f} kHz, "
              f"|Z(100 Hz)| {abs(np.interp(100, f, Z.real) + 1j*np.interp(100, f, Z.imag))/1e3:6.2f} kOhm")
        curves[d.name] = Z
    plot_impedance(f, curves, path, title)
    return f, curves


# ----------------------------------------------------------------------------
# Hum-cancelling dummy coils
# ----------------------------------------------------------------------------
def dummy_coil(real: prl.PickupSpec, N: int = None, **changes) -> prl.PickupSpec:
    """Air-core copy of `real` (same bobbin/wire unless changed), optional other N."""
    return replace(real, name=f"dummy N={N or real.N}", N=N or real.N, air_core=True, **changes)


def match_dummy_N(real: prl.PickupSpec, dummy: prl.PickupSpec, real_hum_scale=1.0) -> int:
    """Turns of the (air-core) dummy so that its N*A_eff equals the real coil's
    -> ideal hum cancellation in a uniform field."""
    target = hum_area_turns(real) * real_hum_scale
    lo, hi = 100, 20 * real.N
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if hum_area_turns(replace(dummy, N=mid, air_core=True)) < target: lo = mid
        else: hi = mid
    return hi


def hum_rejection(design: Design, f_hum=(50.0, 150.0, 250.0)) -> Dict[float, float]:
    """|V_out(hum)| / |V_out(string)| in dB at the hum frequencies (lower = better).
    Compare designs with this; absolute value depends on the field and string level."""
    f = np.asarray(f_hum)
    return {fi: 20 * np.log10(abs(h) / abs(s)) for fi, h, s in
            zip(f, design.response(f, "hum"), design.response(f, "string"))}
