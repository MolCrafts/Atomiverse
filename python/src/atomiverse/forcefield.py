"""Translate a molpy ``ForceField`` into Atomiverse evaluators, style by style.

molpy composes a force field from LAMMPS-standard styles —
``ff.def_style(category, name, params)`` plus that style's types — and owns
typing, parameter tables and charges. Atomiverse owns the kernels. This module
is the seam: :func:`compile` reads every style of a force field against the
typed ``molpy.Frame`` it was assigned to and turns **one molpy style into one
Atomiverse evaluator**. Nothing here knows which force field it is reading:
GAFF, OPLS or a hand-built one go through the same arms.

==========================  ===========================================
molpy style                 Atomiverse evaluator
==========================  ===========================================
``atom full``               (none — masses / charges are frame data)
``pair lj/cut``             ``PairLJCut.table`` — per-type LJ, cross pairs
                            by the style's ``mixing``, LJ special weights
                            from ``ff.special_bonds``, the style's
                            ``cutoff`` and ``shift``
``pair coul/cut``           ``PairCoulCut`` — Coulomb special weights, the
                            style's ``cutoff``, ``coulomb`` / ``dielectric``
``bond harmonic``           ``BondHarmonic``
``angle harmonic``          ``AngleHarmonic``
``dihedral periodic``       ``DihedralCharmm`` (one row per nonzero term)
``dihedral charmm``         ``DihedralCharmm``
``improper cvff``           ``ImproperCvff``
``improper periodic``       ``ImproperCvff`` (molpy converts it exactly)
``improper harmonic``       ``ImproperHarmonic``
==========================  ===========================================

Any other style is refused by name — never approximated. That includes the
Ewald styles (``coul/long``, ``coul/long/pme``): Atomiverse's ``CoulLong``
takes neither the force field's Coulomb special weights nor its Coulomb
constant, so it cannot reproduce them; compose ``CoulLong`` + a k-space
solver yourself when you mean a different model.

Units come from the IR, not from assumptions: every parameter carries a
dimension in molpy's style registry (``E``, ``L``, ``Q``, ``A`` and products),
converted from the force field's declared ``units`` (``real`` or ``metal``) to
Atomiverse's eV / Å / e / rad. An angle *value* (dimension ``A``, stored in
degrees) becomes radians; a per-angle constant (``E/A^2``) is already per
radian, as the registry's own expressions (``k*(theta-theta0*(pi/180))^2``)
show.

Every evaluator is a cutoff style. A gas-phase (vacuum) calculation of one
molecule is plain composition: set the pair styles' ``cutoff`` beyond the
molecule and give the frame a box in which no periodic image comes within it::

    from atomiverse import forcefield as avff

    rc = diameter + 5.0
    for name in ("lj/cut", "coul/cut"):
        ff.get_style("pair", name).set(cutoff=rc)
    frame.box = molpy.Box.cube(diameter + rc + 2 * skin)
    system = avff.compile(ff, frame)
    opt = Minimize()
    opt.set_neighbor(NeighborList(system.cutoff, skin))
    for term in system.evaluators:
        opt.set_evaluator(term)
    opt.run(system.frame, 2000)

Placement: this module depends on molpy's force-field API and on Atomiverse's
parameter types. molpy must not depend on Atomiverse, so the translation lives
here, on the consumer's side of the dependency.
"""

from __future__ import annotations

import math
import re
from typing import Any, NamedTuple

import numpy as np

from atomiverse._binding import _cpp, require_domain

require_domain(__name__, "potentials")

#: 1 kcal/mol in eV (CODATA 2018: 4184 J / (N_A · e)).
KCAL_PER_MOL: float = 4184.0 / (6.02214076e23 * 1.602176634e-19)

#: Base-dimension factors into Atomiverse units, per molpy ``units``.
_UNITS: dict[str, dict[str, float]] = {
    "real": {"E": KCAL_PER_MOL, "L": 1.0, "Q": 1.0},
    "metal": {"E": 1.0, "L": 1.0, "Q": 1.0},
}

_PREFIX = "atv_"
_DIM_TOKEN = re.compile(r"([*/]?)\s*([A-Za-z]+|1)(?:\^(-?\d+))?")


class ForceFieldError(ValueError):
    """A molpy force field this module cannot translate exactly."""


class Compiled(NamedTuple):
    """What :func:`compile` returns.

    ``frame`` is the ``molpy.Frame`` a driver runs on (positions, masses,
    charges, the force-field atom type as integer ``atoms.type_id``, the
    topology with integer ``type_id`` kinds, the box). ``evaluators`` are the
    parameter objects, one per translated style, in the force field's style
    order. ``cutoff`` is the largest pair cutoff [Å] — build the
    ``NeighborList`` at least this long — or ``None`` without pair styles.
    """

    frame: Any
    evaluators: list[Any]
    cutoff: float | None


def compile(forcefield: Any, frame: Any) -> Compiled:  # noqa: A001 - mirrors molpy's PotentialCompiler.compile
    """Translate ``forcefield`` against the typed ``frame``, style by style.

    Parameters
    ----------
    forcefield : molpy.ff.forcefield.ForceField
        The force field the frame was typed against.
    frame : molpy.Frame
        The typed system: ``atoms`` with ``x``/``y``/``z``, ``atomic_number``,
        ``type`` (force-field atom types), ``charge`` [e] when a Coulomb style
        is present, and a box; topology blocks with their ``type`` labels, as
        ``Atomistic.to_frame()`` writes a typed molecule. It is not modified.

    Raises
    ------
    ForceFieldError
        On a style without an Atomiverse arm, two styles of one bonded
        category, a parameter the force field leaves unset, a style parameter
        the kernel cannot honour (LJ exponents other than 12-6, an unknown
        mixing rule, a Coulomb ``delta``, a CHARMM dihedral weight), a
        torsion multiplicity that is not a positive integer, or units other
        than ``real`` / ``metal``.
    KeyError
        If the frame has no box, or no ``atoms.charge`` while a Coulomb style
        is present.
    """
    import molpy
    from molpy.ff import style_registry

    units = _UNITS.get(forcefield.units)
    if units is None:
        raise ForceFieldError(
            f"force field units are {forcefield.units!r}; supported: {sorted(_UNITS)}"
        )
    specs = {(s.category, s.name): s for s in style_registry.GlobalRegistry().styles()}
    styles = list(forcefield.get_styles(molpy.ff.forcefield.Style))
    _check_styles(styles)
    if frame.box is None:
        raise KeyError("the frame has no box; the neighbor list needs one")
    if any(s.category == "pair" and s.name == "coul/cut" for s in styles) and (
        "charge" not in frame["atoms"].keys()
    ):
        raise KeyError(
            "atoms.charge is missing: assign partial charges (e.g. "
            "molpy.ff.charge.GasteigerModel or atomiverse.AM1BCCCharges) first"
        )

    work = frame.copy()
    ff = forcefield
    if any(s.category == "improper" and s.name == "periodic" for s in styles):
        ff = forcefield.to_form("improper", "cvff")
    try:
        ff.materialize_params(work, prefix=_PREFIX)
    except ValueError as error:  # molpy's refusal: a row with no type
        raise ForceFieldError(str(error)) from error

    special = forcefield.special_bonds
    special_lj, special_coul = (
        ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        if special is None
        else (tuple(float(w) for w in special[0]), tuple(float(w) for w in special[1]))
    )
    conv = _Converter(units, specs)
    atom_types = sorted({str(t) for t in np.asarray(work["atoms"]["type"])})
    out = _driver_frame(work, atom_types)
    evaluators: list[Any] = []
    cutoffs: list[float] = []
    for style in styles:
        key = (style.category, style.name)
        if key == ("atom", "full"):
            continue
        if key == ("pair", "lj/cut"):
            term, rc = _lj_cut(style, conv, atom_types, special_lj)
            cutoffs.append(rc)
        elif key == ("pair", "coul/cut"):
            term, rc = _coul_cut(style, conv, special_coul)
            cutoffs.append(rc)
        else:
            term = _bonded(style, conv, work, out)
        if term is not None:
            evaluators.append(term)
    return Compiled(out, evaluators, max(cutoffs) if cutoffs else None)


# ── Styles and units ─────────────────────────────────────────────────────

_ARMS = {
    ("atom", "full"),
    ("pair", "lj/cut"),
    ("pair", "coul/cut"),
    ("bond", "harmonic"),
    ("angle", "harmonic"),
    ("dihedral", "periodic"),
    ("dihedral", "charmm"),
    ("improper", "cvff"),
    ("improper", "periodic"),
    ("improper", "harmonic"),
}


def _check_styles(styles: list[Any]) -> None:
    seen: dict[str, str] = {}
    for style in styles:
        key = (style.category, style.name)
        if key not in _ARMS:
            raise ForceFieldError(
                f"style '{style.category} {style.name}' has no Atomiverse "
                "counterpart in this module"
            )
        if style.category in ("pair", "atom"):
            continue
        if style.category in seen:
            raise ForceFieldError(
                f"two {style.category} styles ('{seen[style.category]}', "
                f"'{style.name}') share one topology block; one style per "
                "bonded category is translated"
            )
        seen[style.category] = style.name


class _Converter:
    """Dimension-driven conversion into Atomiverse units."""

    def __init__(self, units: dict[str, float], specs: dict[tuple[str, str], Any]) -> None:
        self._units = units
        self._specs = specs

    def factor(self, dim: str) -> float:
        """Factor for a registry dimension such as ``E/L^2`` or ``E*L/Q^2``."""
        total = 1.0
        for op, base, power in _DIM_TOKEN.findall(dim.replace(" ", "")):
            if base == "1":
                continue
            p = int(power) if power else 1
            p = -p if op == "/" else p
            if base == "A":
                # An angle value is stored in degrees; a per-angle constant
                # is per radian already (the registry expressions convert
                # only the value).
                total *= (math.pi / 180.0) ** p if p > 0 else 1.0
            elif base in self._units:
                total *= self._units[base] ** p
            else:
                raise ForceFieldError(f"dimension '{dim}': unknown base '{base}'")
        return total

    def params(self, category: str, name: str, *, style_level: bool = False) -> dict[str, Any]:
        """The registry's ``ParamSpec`` (dimension, default) per parameter."""
        spec = self._specs.get((category, name))
        if spec is None:
            raise ForceFieldError(f"style '{category} {name}' is not in molpy's style registry")
        return {p.name: p for p in (spec.style_params if style_level else spec.params)}

    def dims(self, category: str, name: str, *, style_level: bool = False) -> dict[str, str]:
        return {k: p.dim for k, p in self.params(category, name, style_level=style_level).items()}

    def style_param(self, style: Any, key: str) -> float:
        dims = self.dims(style.category, style.name, style_level=True)
        value = style.params.get(key)
        if value is None:
            raise ForceFieldError(f"style '{style.category} {style.name}' states no '{key}'")
        return float(value) * self.factor(dims[key])


def _column(block: Any, key: str) -> np.ndarray:
    return np.asarray(block[key])


def _dense(block: Any, key: str, block_name: str) -> np.ndarray:
    """A materialized parameter column with every row stated, or the refusal
    naming the types that lack it."""
    if key not in block.keys():
        raise ForceFieldError(f"{block_name}: no '{key[len(_PREFIX):]}' parameter")
    values = _column(block, key)
    valid = block.validity(key)
    if valid is not None and not all(valid):
        types = _column(block, "type")
        missing = sorted({str(t) for t, ok in zip(types, valid) if not ok})
        raise ForceFieldError(
            f"{block_name}: no '{key[len(_PREFIX):]}' parameter for type(s) "
            f"{missing}; the force field does not cover this system"
        )
    return values.astype(np.float64)


def _kinds(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique parameter rows and, per input row, the index of its kind."""
    if len(rows) == 0:
        return rows, np.zeros(0, dtype=np.uint64)
    table, index = np.unique(rows, axis=0, return_inverse=True)
    return table, index.astype(np.uint64).reshape(-1)


def _driver_frame(work: Any, atom_types: list[str]) -> Any:
    """Atoms (positions, mass, charge, Z, integer type) and the box."""
    import molpy

    atoms = work["atoms"]
    index = {name: i for i, name in enumerate(atom_types)}
    block = molpy.Block()
    for key in ("x", "y", "z"):
        block.insert(key, _column(atoms, key).astype(np.float64))
    if "charge" in atoms.keys():
        block.insert("charge", _column(atoms, "charge").astype(np.float64))
    mass = _column(atoms, "mass") if "mass" in atoms.keys() else _dense(atoms, _PREFIX + "mass", "atoms")
    block.insert("mass", np.asarray(mass, dtype=np.float64))
    block.insert("atomic_number", _column(atoms, "atomic_number").astype(np.uint64))
    block.insert(
        "type_id",
        np.array([index[str(t)] for t in _column(atoms, "type")], dtype=np.uint64),
    )
    out = molpy.Frame()
    out["atoms"] = block
    out.box = work.box
    return out


# ── Pair arms ────────────────────────────────────────────────────────────


def _mix(rule: str, ei: float, si: float, ej: float, sj: float) -> tuple[float, float]:
    if rule == "arithmetic":
        return math.sqrt(ei * ej), 0.5 * (si + sj)
    if rule == "geometric":
        return math.sqrt(ei * ej), math.sqrt(si * sj)
    if rule == "sixthpower":
        s6 = 0.5 * (si**6 + sj**6)
        return 2.0 * math.sqrt(ei * ej) * si**3 * sj**3 / (si**6 + sj**6), s6 ** (1.0 / 6.0)
    raise ForceFieldError(f"pair lj/cut: unknown mixing rule {rule!r}")


def _lj_cut(style: Any, conv: _Converter, atom_types: list[str], special: tuple) -> tuple[Any, float]:
    """``pair lj/cut`` → ``PairLJCut.table`` over the frame's atom types."""
    params = style.params
    if float(params.get("n", 12.0)) != 12.0 or float(params.get("m", 6.0)) != 6.0:
        raise ForceFieldError("pair lj/cut: only the 12-6 exponents have a kernel")
    rule = str(params.get("mixing", "arithmetic"))
    shift = bool(float(params.get("shift", 0.0)))
    cutoff = conv.style_param(style, "cutoff")
    dims = conv.dims("pair", "lj/cut")
    f_eps, f_sig = conv.factor(dims["epsilon"]), conv.factor(dims["sigma"])

    diag: dict[str, tuple[float, float]] = {}
    cross: dict[frozenset, tuple[float, float]] = {}
    for t in style.get_types():
        row = (float(t.params["epsilon"]) * f_eps, float(t.params["sigma"]) * f_sig)
        if t.itom.name == t.jtom.name:
            diag[t.itom.name] = row
        else:
            cross[frozenset((t.itom.name, t.jtom.name))] = row
    missing = [name for name in atom_types if name not in diag]
    if missing:
        raise ForceFieldError(f"pair lj/cut: no like-pair row for atom type(s) {missing}")

    ti, tj, eps, sig = [], [], [], []
    for a, name_a in enumerate(atom_types):
        for b in range(a, len(atom_types)):
            name_b = atom_types[b]
            row = cross.get(frozenset((name_a, name_b)))
            if row is None:
                row = diag[name_a] if a == b else _mix(rule, *diag[name_a], *diag[name_b])
            ti.append(a)
            tj.append(b)
            eps.append(row[0])
            sig.append(row[1])
    term = _cpp.PairLJCut.table(ti, tj, eps, sig, cutoff, *special, energy_shift=shift)
    return term, cutoff


def _coul_cut(style: Any, conv: _Converter, special: tuple) -> tuple[Any, float]:
    """``pair coul/cut`` → ``PairCoulCut`` with the style's own constant."""
    if float(style.params.get("delta", 0.0)) != 0.0:
        raise ForceFieldError("pair coul/cut: a nonzero 'delta' softening has no kernel")
    dielectric = float(style.params.get("dielectric", 1.0))
    coulomb = conv.style_param(style, "coulomb") / dielectric
    cutoff = conv.style_param(style, "cutoff")
    return _cpp.PairCoulCut(cutoff, *special, coulomb=coulomb), cutoff


# ── Bonded arms ──────────────────────────────────────────────────────────

_BLOCKS = {
    "bond": ("bonds", ("atomi", "atomj")),
    "angle": ("angles", ("atomi", "atomj", "atomk")),
    "dihedral": ("dihedrals", ("atomi", "atomj", "atomk", "atoml")),
    "improper": ("impropers", ("atomi", "atomj", "atomk", "atoml")),
}


def _topology(block: Any, ends: tuple[str, ...], kinds: np.ndarray, rows: np.ndarray | None = None) -> Any:
    import molpy

    out = molpy.Block()
    for end in ends:
        col = _column(block, end).astype(np.uint64)
        out.insert(end, col if rows is None else col[rows])
    out.insert("type_id", kinds)
    return out


def _bonded(style: Any, conv: _Converter, work: Any, out: Any) -> Any | None:
    """One bonded style → one evaluator over its topology block (or ``None``
    when the frame has no terms of that category)."""
    block_name, ends = _BLOCKS[style.category]
    if block_name not in work.keys() or not work[block_name].n_rows:
        return None
    block = work[block_name]
    name = "cvff" if (style.category, style.name) == ("improper", "periodic") else style.name
    specs = conv.params(style.category, name)
    dims = {k: s.dim for k, s in specs.items()}

    def col(key: str) -> np.ndarray:
        # A parameter no type states takes the registry's declared default.
        if _PREFIX + key not in block.keys() and specs[key].default is not None:
            return np.full(block.n_rows, float(specs[key].default) * conv.factor(dims[key]))
        return _dense(block, _PREFIX + key, block_name) * conv.factor(dims[key])

    if style.category == "dihedral" and style.name == "periodic":
        return _periodic_dihedrals(block, conv.factor(dims["k"]), conv.factor(dims["phase"]), out)
    if style.category == "dihedral":  # charmm
        if np.any(col("w") != 0.0):
            raise ForceFieldError("dihedral charmm: a nonzero 1-4 weight 'w' is not translated")
        columns = [col("k"), col("periodicity"), col("phase")]
    elif style.category == "improper" and name == "cvff":
        columns = [col("k"), col("sign"), col("periodicity")]
    elif style.category == "improper":  # harmonic
        columns = [col("k"), col("chi0")]
    elif style.category == "angle":
        columns = [col("k"), col("theta0")]
    else:  # bond harmonic
        columns = [col("k"), col("r0")]
    table, kinds = _kinds(np.stack(columns, axis=1))
    out[block_name] = _topology(block, ends, kinds)
    return _make_bonded(style.category, name, table)


def _make_bonded(category: str, name: str, t: np.ndarray) -> Any:
    if category == "bond":
        return _cpp.BondHarmonic(list(t[:, 0]), list(t[:, 1]))
    if category == "angle":
        return _cpp.AngleHarmonic(list(t[:, 0]), list(t[:, 1]))
    if category == "dihedral":
        return _cpp.DihedralCharmm(list(t[:, 0]), [_multiplicity(n) for n in t[:, 1]], list(t[:, 2]))
    if name == "cvff":
        return _cpp.ImproperCvff(list(t[:, 0]), [int(d) for d in t[:, 1]], [int(n) for n in t[:, 2]])
    return _cpp.ImproperHarmonic(list(t[:, 0]), list(t[:, 1]))


def _multiplicity(n: float) -> int:
    if n < 1 or n != math.floor(n):
        raise ForceFieldError(f"dihedral multiplicity {n} is not a positive integer")
    return int(n)


def _periodic_dihedrals(block: Any, f_k: float, f_phase: float, out: Any) -> Any | None:
    """``dihedral periodic``: one CHARMM-form row per nonzero term."""
    terms = sorted(
        int(key[len(_PREFIX) + 1:])
        for key in block.keys()
        if key.startswith(_PREFIX + "k") and key[len(_PREFIX) + 1:].isdigit()
    )
    if not terms:
        raise ForceFieldError("dihedrals carry no periodic parameters")
    rows: list[tuple[float, float, float]] = []
    source: list[int] = []
    for i in terms:
        k = _column(block, f"{_PREFIX}k{i}")
        valid = block.validity(f"{_PREFIX}k{i}")  # fewer terms leave cells null
        period = _column(block, f"{_PREFIX}periodicity{i}")
        p_key = f"{_PREFIX}phase{i}"
        phase = _column(block, p_key) if p_key in block.keys() else np.zeros(len(k))
        for row in range(len(k)):
            if (valid is not None and not valid[row]) or k[row] == 0.0:
                continue
            n = float(period[row])
            _multiplicity(n)
            rows.append((float(k[row]) * f_k, n, float(phase[row]) * f_phase))
            source.append(row)
    if not rows:
        return None
    table, kinds = _kinds(np.asarray(rows, dtype=np.float64))
    out["dihedrals"] = _topology(block, _BLOCKS["dihedral"][1], kinds, np.asarray(source))
    return _make_bonded("dihedral", "periodic", table)


__all__ = ["Compiled", "ForceFieldError", "KCAL_PER_MOL", "compile"]
