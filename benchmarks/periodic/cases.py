"""PBE-family rows using the original periodic oracle decks and gates."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.framework.provenance import file_hash
from benchmarks.framework.specification import Check, Method, Participant, Row, System
from benchmarks.scf.hf import BOHR_IN_ANGSTROM

ROOT = Path(__file__).resolve().parents[2]
PBE_VARIANTS = ("PBEsol", "RevPBE", "RPBE")
ATOMIC_NUMBERS = {"H": 1, "O": 8, "Si": 14}


def _fixture(
    method: str, fixture_id: str, path: Path, spin: int
) -> tuple[System, dict]:
    """Read the geometry cards from these specific, archived input dialects."""
    deck = path.read_text()
    grid_controls = {}
    if method == "pw":
        cell_text, geometry = deck.split("CELL_PARAMETERS angstrom\n")[1].split(
            "ATOMIC_POSITIONS angstrom\n"
        )
        atoms_text = geometry.split("K_POINTS gamma")[0]
        cell = [[float(x) for x in line.split()] for line in cell_text.splitlines()]
        species_text = deck.split("ATOMIC_SPECIES\n")[1].split(
            "CELL_PARAMETERS angstrom\n"
        )[0]
        species = [line.split() for line in species_text.splitlines() if line.strip()]
        masses = {symbol: float(mass) for symbol, mass, _ in species}
        potentials = []
        assets = []
        for symbol, _, name in species:
            z = ATOMIC_NUMBERS[symbol]
            asset = path.with_name(name)
            if fixture_id == "h2o-rks-skew":
                asset = (
                    ROOT / "tests/data/pw" / name
                    if symbol == "O"
                    else ROOT
                    / "regressions/pw/pbe-family/qe-inputs/h3-uks-pbesol"
                    / name
                )
            relative = str(asset.relative_to(ROOT))
            assets.append(relative)
            if asset.suffix.lower() == ".upf":
                potentials.append((z, "upf", relative))
            else:
                _, family, functional, charge = asset.stem.split("_")
                potentials.append((z, "gth", f"{family}-{functional}-{charge}"))
    else:
        atoms_text = deck.split("UNIT bohr\n")[1].split("&END COORD")[0]
        abc = re.search(r"^\s*ABC \[bohr\] (.+)$", deck, re.MULTILINE)
        if abc is None:
            raise ValueError(f"{path}: expected archived orthorhombic CP2K cell")
        cell = [
            [float(length) * BOHR_IN_ANGSTROM if i == j else 0.0 for j in range(3)]
            for i, length in enumerate(abc[1].split())
        ]
        assets = []
        masses = {"H": 1.008, "Si": 28.0855}
        potentials = []
        mgrid = deck.split("&MGRID\n")[1].split("&END MGRID")[0]
        grid = dict(line.split() for line in mgrid.splitlines() if line.strip())
        grid_controls = {
            "cutoff_ry": float(grid["CUTOFF"]),
            "ngrids": int(grid["NGRIDS"]),
            "rel_cutoff": float(grid.get("REL_CUTOFF", "0")),
        }
        mesh = re.search(r"^\s*SCHEME MONKHORST-PACK (\d+) (\d+) (\d+)$", deck, re.MULTILINE)
        grid_controls["k_mesh"] = [int(n) for n in mesh.groups()] if mesh else [1, 1, 1]
        grid_controls["stress_kind"] = re.search(
            r"^\s*STRESS_TENSOR (ANALYTICAL|NUMERICAL)$", deck, re.MULTILINE
        )[1].title()
        grid_controls["basis_set_file"] = Path(re.search(
            r"^\s*BASIS_SET_FILE_NAME\s+(\S+)", deck, re.MULTILINE
        )[1]).name
    atoms = [line.split() for line in atoms_text.splitlines() if line.strip()]
    if len(cell) != 3 or any(len(vector) != 3 for vector in cell):
        raise ValueError(f"{path}: expected three lattice vectors")
    scale = 1.0 if method == "pw" else BOHR_IN_ANGSTROM
    system = System(
        id=fixture_id,
        label={
            "si2-rks": "Si₂ RKS Γ",
            "h2o-rks-skew": "H₂O RKS Γ, skew",
            "h3-uks-skew": "H₃ UKS Γ, skew",
            "h2-rks-ortho": "H₂ RKS Γ",
            "h2-rks-boundary": "H₂ RKS Γ across a periodic boundary",
            "h2-rks-moved": "H₂ RKS Γ after an atomic displacement",
            "h2-rks-multigrid": "H₂ RKS Γ, multigrid",
            "h3-uks-ortho": "H₃ UKS Γ",
            "h3-uks-kpoints": "H₃ UKS, two k points",
            "si4-rks-kpoints": "Displaced diamond Si₄ RKS, two k points",
            "si4-rks-kpoints-multigrid": "Displaced diamond Si₄ RKS, two k points and three grid levels",
        }[fixture_id],
        atomic_numbers=[ATOMIC_NUMBERS[atom[0]] for atom in atoms],
        positions_angstrom=[[float(x) * scale for x in atom[1:]] for atom in atoms],
        spin=spin,
    )
    return system, {
        "cell_vectors_angstrom": cell,
        "atomic_masses_amu": [masses[atom[0]] for atom in atoms],
        "nc_potentials": potentials,
        "oracle_input": str(path.relative_to(ROOT)),
        "oracle_input_sha256": file_hash(path),
        "oracle_assets": {name: file_hash(ROOT / name) for name in assets},
        **grid_controls,
    }


def method_spec(method: str) -> Method:
    pw = method == "pw"
    oracle = "qe" if pw else "cp2k"
    reference_dir = ROOT / "regressions" / method / "pbe-family"
    rows = []
    fixtures = (
        [
            ("si2-rks", 0, PBE_VARIANTS),
            ("h3-uks-skew", 1, PBE_VARIANTS),
            ("h2o-rks-skew", 0, ("PBE",)),
        ]
        if pw
        else [
            ("h2-rks-ortho", 0, PBE_VARIANTS),
            ("h3-uks-ortho", 1, ("PBE", *PBE_VARIANTS)),
            ("h2-rks-boundary", 0, ("PBEsol",)),
            ("h2-rks-moved", 0, ("PBEsol",)),
            ("h2-rks-multigrid", 0, ("PBEsol",)),
            ("si4-rks-kpoints", 0, ("PBE",)),
            ("si4-rks-kpoints-multigrid", 0, ("PBE",)),
            ("h3-uks-kpoints", 1, ("PBE",)),
        ]
    )
    for fixture_id, spin, functionals in fixtures:
        archive_id = fixture_id.removesuffix("-ortho").removesuffix("-skew")
        mixed = fixture_id == "h2o-rks-skew"
        boundary = fixture_id == "h2-rks-boundary"
        moved = fixture_id == "h2-rks-moved"
        multigrid = fixture_id.endswith("-multigrid")
        kpoints = "-kpoints" in fixture_id
        silicon = fixture_id.startswith("si4-")
        cpu_only = not pw and spin and kpoints
        for xc in functionals:
            folder = f"{archive_id}-{xc.lower()}"
            if pw:
                path = (
                    Path(__file__).with_name("fixtures") / f"{folder}.in"
                    if mixed
                    else reference_dir / "qe-inputs" / folder / "input.in"
                )
                controls = {
                    "cutoff_ry": 6.0 if spin else 12.0,
                    "tol": 1e-9,
                    "max_iter": 250 if spin else 200,
                    "fermi_dirac": not mixed,
                    "expected_electrons": 3 if spin else 8,
                    "degauss_ry": 0.02,
                    "extra_bands": 3 if spin else 2,
                    "mix_beta": 0.35 if spin else (0.3 if mixed else 0.2),
                }
            else:
                # The translated and geometry-reuse rows finish at the
                # same physical system as the archived H2 reference.
                if boundary or moved:
                    folder = "h2-rks-pbesol"
                path = (
                    Path(__file__).with_name("fixtures") / f"{folder}.inp"
                    if multigrid or kpoints
                    else reference_dir / "cp2k-inputs" / f"gpw-{folder}" / "input.inp"
                )
                controls = {
                    "tol": 1e-10,
                    "max_iter": 200 if kpoints else 100,
                    "eps_default": 1e-12 if kpoints else 1e-10,
                    "diis_depth": 8 if kpoints else 6,
                    "strain_step": 0.002,
                    "pseudopotential": "GTH-PBE-q4" if silicon else "GTH-PBE-q1",
                    "basis": "DZVP-MOLOPT-GTH" if silicon else "DZV-GTH-PADE",
                    "expected_electrons": (3 if spin else 16) if kpoints else 0,
                    "first_atom_image": [-1, -1, -1] if boundary else [0, 0, 0],
                    "first_atom_seed_shift_bohr": -0.4 if moved else 0.0,
                }
            system, geometry = _fixture(method, fixture_id, path, spin)
            if boundary or moved or multigrid or spin or xc != "PBEsol":
                system.tiers = ["acceptance", "performance"]
            checks = [
                Check(
                    observable="energy_hartree",
                    unit="Ha",
                    atol=1e-5 if pw else 5e-6,
                    justification="Original PBE-family gate for the matched total energy, including the entropy term for smeared rows.",
                ),
                Check(
                    observable="forces_hartree_per_bohr",
                    unit="Ha/Bohr",
                    atol=1e-4 if pw else 5e-5,
                    justification="Original PBE-family gate on every atom and Cartesian component.",
                ),
                Check(
                    observable="stress",
                    unit="eV/Å³" if pw else "Ha/Bohr³",
                    atol=2e-3 if pw else 5e-5,
                    justification="Original PBE-family gate on all six PW components or the three supported GPW diagonal components.",
                ),
            ]
            if boundary:
                checks += [
                    check.model_copy(update={"observable": f"translated_{check.observable}"})
                    for check in checks
                ]
            rows.append(
                Row(
                    id=f"{method}.{archive_id}.{xc.lower()}",
                    label=f"{method.upper()} {system.label} / {xc}",
                    capability=f"dft.{method}.periodic",
                    task="energy_forces_stress",
                    model={
                        "method": method,
                        "xc": xc,
                        "comparison_class": "numerical_only",
                        "oracle_version": "7.6" if pw else "2026.2",
                        "stress_components": ["xx", "yy", "zz", "xy", "xz", "yz"]
                        if pw
                        else ["xx", "yy", "zz"],
                        **controls,
                        **geometry,
                    },
                    systems=[system],
                    reference=f"{oracle}.cpu",
                    participants=[
                        Participant(
                            id="atv.cpu",
                            engine="atomiverse",
                            backend="cpu",
                            role="atv",
                            required=True,
                            entrypoint="benchmarks.periodic.adapters:atv_adapter",
                        ),
                        *(
                            []
                            if cpu_only
                            else [
                                Participant(
                                    id="atv.cuda",
                                    engine="atomiverse",
                                    backend="cuda",
                                    role="atv",
                                    entrypoint="benchmarks.periodic.adapters:atv_adapter",
                                )
                            ]
                        ),
                        Participant(
                            id=f"{oracle}.cpu",
                            engine=oracle,
                            backend="cpu",
                            role="oracle",
                            required=True,
                            entrypoint="benchmarks.periodic.adapters:oracle_adapter",
                        ),
                    ],
                    checks=checks,
                )
            )
    return Method(id=method, rows=rows)
