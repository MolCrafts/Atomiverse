#!/usr/bin/env python3
"""Read native output and offline QE evidence; never execute a scientific code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
LABELS = ["public-run", "public-forces", "public-stress", "nested-force-and-stress"]
GATES = {"energy_Ha": 1e-5, "each_force_Ha_Bohr": 1e-4, "each_stress_eV_A3": 2e-3}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(token):
    value = float(token.replace("D", "e").replace("d", "e"))
    require(math.isfinite(value), "nonfinite numeric evidence")
    return value


def load(path):
    return json.loads(Path(path).read_text())


def tokens(text):
    return re.findall(NUMBER, text)


def flat(rows):
    return [x for row in rows for x in row]


def check_reference(directory):
    directory = directory.resolve()
    manifest = load(directory / "SHA256SUMS.json")
    require(isinstance(manifest, dict) and len(manifest) >= 20, "incomplete QE manifest")
    for name, digest in manifest.items():
        path = (directory / name).resolve()
        require(directory in path.parents and path.is_file(), "missing/unsafe reference file: " + name)
        require(sha(path) == digest, "reference SHA mismatch: " + name)
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    require(files == set(manifest) | {"SHA256SUMS.json"}, "unmanifested or omitted QE reference files")
    ref = load(directory / "reference.json")
    require(ref["predeclared_gates_unchanged"] == GATES, "reference bounds changed")
    require(ref["QE"]["scientific_converged"] and ref["QE"]["process_exit_code"] == 0,
            "QE reference did not scientifically converge")
    for name, digest in ref["raw_sha256"].items():
        require(sha(directory / name) == digest, "raw reference digest mismatch: " + name)
    for source in ref["primary_QE_source"]["files"].values():
        require(sha(directory / source["local"]) == source["sha256"], "primary QE source digest mismatch")
    require(ref["primary_QE_source"]["commit"] == "9f93ddec427d2b9a45bb72d828c6d324f62fcabd",
            "unexpected primary QE release source")
    require(ref["model"]["QE_input_dft"] == "SLA+VWN+NOGX+NOGC"
            and ref["model"]["indices"] == {"exchange": 1, "correlation": 2,
                                           "gradient_exchange": 0, "gradient_correlation": 0},
            "reference is not native Slater/VWN")
    for path in [directory / "Si_GTH_PBE_q4.gth", directory / "successful/Si_GTH_PBE_q4.gth",
                 directory / "failed-dimer/Si_GTH_PBE_q4.gth"]:
        require(sha(path) == ref["potential_sha256"], "potential copy/model mismatch")
    text = (directory / "successful/qe.out").read_text()
    require("Program PWSCF v.7.6" in text and "JOB DONE." in text, "QE version/completion missing")
    printed = re.findall(r"Exchange-correlation=\s*([^\n]+)", text)
    require(len(printed) == 2 and all(x.strip() == "SLA+VWN+NOGX+NOGC" for x in printed),
            "actual printed XC is not VWN")
    converged = re.findall(r"convergence has been achieved in\s+(\d+) iterations", text)
    require(converged == [str(ref["QE"]["SCF_iterations"])]
            and "convergence NOT achieved" not in text, "QE scientific convergence absent")
    inp = (directory / "successful/input.in").read_text()
    def field(name):
        found = re.findall(r"\b" + re.escape(name) + r"\s*=\s*([^,\n/]+)", inp, re.I)
        require(len(found) == 1, "missing/duplicate QE input field: " + name)
        return found[0].strip().strip("'\"").lower()
    for key, expected in {"calculation": "scf", "input_dft": "sla+vwn+nogx+nogc",
                          "occupations": "fixed"}.items():
        require(field(key) == expected, "QE input model mismatch: " + key)
    for key, expected in {"nat": 2, "ntyp": 1, "nspin": 1, "nbnd": 4, "ecutwfc": 12,
                          "ecutrho": 48, "nr1": 24, "nr2": 28, "nr3": 24,
                          "nr1s": 24, "nr2s": 28, "nr3s": 24,
                          "conv_thr": 1e-12, "electron_maxstep": 250, "mixing_beta": .2}.items():
        require(number(field(key)) == expected, "QE input numeric mismatch: " + key)
    require(re.search(r"K_POINTS\s+gamma\s*$", inp, re.I), "QE reference is not Gamma")
    for title, expected, species in [
            ("CELL_PARAMETERS", ref["fixture"]["H_columns_A"], False),
            ("ATOMIC_POSITIONS", ref["fixture"]["positions_A"], True)]:
        start = re.search(title + r"\s+angstrom\s*\n", inp, re.I)
        require(start, "missing QE geometry section: " + title)
        rows = inp[start.end():].splitlines()[:len(expected)]
        parsed = []
        for row in rows:
            values = row.split()
            if species:
                require(values.pop(0) == "Si", "QE species mismatch")
            parsed.append([number(x) for x in values])
        require(parsed == expected, "QE geometry differs from locked fixture")
    xml = ET.parse(directory / "successful/scf-data-file-schema.xml").getroot()
    require(xml.attrib.get("Units") == "Hartree atomic units", "unrecognized QE XML units")
    out = xml.find("output")
    require(out is not None and out.findtext("convergence_info/scf_conv/convergence_achieved") == "true",
            "XML scientific convergence false")
    require(int(out.findtext("convergence_info/scf_conv/n_scf_steps")) == int(converged[0]),
            "XML/raw iteration disagreement")
    require(0 <= number(out.findtext("convergence_info/scf_conv/scf_error")) <= .5e-12,
            "XML SCF accuracy fails original reference threshold")
    values = ref["reference"]
    energy = number(out.findtext("total_energy/etot"))
    force = [number(x) for x in out.findtext("forces").split()]
    stress = [number(x) for x in out.findtext("stress").split()]
    require(energy == values["energy_Ha"] and force == flat(values["forces_Ha_Bohr"])
            and stress == flat(values["stress_tensor_Ha_Bohr3"]), "full-precision XML/JSON mismatch")
    energies = re.findall(r"!\s+total energy\s*=\s*(" + NUMBER + ")\s+Ry", text)
    require(len(energies) == 1 and abs(number(energies[0]) / 2 - energy) < 2.51e-9,
            "printed/XML energy mismatch")
    block = text.split("Forces acting on atoms")[1].split("The non-local contrib")[0]
    rows = re.findall(r"atom\s+(\d+)\s+type\s+1\s+force\s*=\s*(" + NUMBER
                      + r")\s+(" + NUMBER + r")\s+(" + NUMBER + ")", block)
    require([r[0] for r in rows] == ["1", "2"], "incomplete/duplicate total QE force rows")
    printed_force = [number(x) / 2 for row in rows for x in row[1:]]
    require(max(abs(a-b) for a, b in zip(printed_force, force)) < 2.51e-9,
            "printed/XML total-force mismatch")
    match = re.search(r"total\s+stress\s+\(Ry/bohr\*\*3\)[^\n]*\n((?:[^\n]*\n){3})", text)
    require(match, "missing full QE stress tensor")
    printed_stress = [number(x) / 2 for row in match[1].splitlines() for x in row.split()[:3]]
    require(len(printed_stress) == 9 and max(abs(a-b) for a, b in zip(printed_stress, stress)) < 2.51e-9,
            "printed/XML stress mismatch")
    hartree_ev = 4.3597447222071e-18 / 1.602176634e-19
    converted = [stress[i] * hartree_ev / .529177210903**3 for i in [0, 4, 8, 1, 2, 5]]
    require(max(abs(a-b) for a, b in zip(converted, values["stress_voigt_eV_A3"])) < 1e-15,
            "stress unit/Voigt conversion mismatch")
    header = (directory / "pins.h").read_text()
    actual_tokens = []
    for name in ["energy_ha", "forces_ha_bohr", "stress_ev_ang3"]:
        part = re.search(r"\b" + name + r"\s*=\s*([^;]+);", header)
        require(part, "missing static pin declaration: " + name)
        actual_tokens.extend(tokens(part[1]))
    expected_tokens = [repr(x) for x in [energy, *force, *values["stress_voigt_eV_A3"]]]
    require(actual_tokens == expected_tokens, "pins.h numeric tokens differ from original reference values")
    bands = ET.parse(directory / "successful/bands-data-file-schema.xml").getroot()
    ev = [number(x) for x in bands.findtext("output/band_structure/ks_energies/eigenvalues").split()]
    occ = [number(x) for x in bands.findtext("output/band_structure/ks_energies/occupations").split()]
    require(len(ev) == 6 and occ == [1, 1, 1, 1, 0, 0] and ev[4] - ev[3] > 1e-3,
            "fixed-charge six-band gap evidence invalid")
    require(ev[4] - ev[3] == ref["separate_fixed_charge_spectral_probe"]["gap_4_to_5_Ha"],
            "spectral XML/reference gap mismatch")
    failure = load(directory / "failed-dimer/status.json")
    require(failure["scientific_converged"] is False and failure["eligible_as_pin"] is False
            and "convergence NOT achieved after 250 iterations" in (directory / "failed-dimer/qe.out").read_text(),
            "failed dimer evidence lost or relabeled")
    return ref, {"manifest_files": len(manifest), "manifest_sha256": sha(directory / "SHA256SUMS.json"),
                 "XML_raw_pins_model_units_verified": True, "failed_dimer_preserved": True}


def check_k8_reference(directory):
    """Validate the executed eight-k run; primary VWN provenance stays shared."""
    directory = directory.resolve()
    gamma, gamma_integrity = check_reference(directory.parent / "qe-2026-10-28")
    manifest = load(directory / "SHA256SUMS.json")
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    require(len(manifest) == 15 and files == set(manifest) | {"SHA256SUMS.json"},
            "incomplete/unmanifested eight-k reference")
    for name, digest in manifest.items():
        path = (directory / name).resolve()
        require(directory in path.parents and path.is_file() and sha(path) == digest,
                "eight-k reference digest/path mismatch: " + name)
    ref = load(directory / "reference.json")
    require(ref["predeclared_gates_unchanged"] == GATES, "eight-k gates changed")
    shared = ref["shared_primary_provenance"]
    require(shared["package"] == "../qe-2026-10-28"
            and shared["manifest_sha256"] == gamma_integrity["manifest_sha256"]
            and shared["primary_QE_source"] == gamma["primary_QE_source"]
            and ref["model"] == gamma["model"], "eight-k primary VWN provenance mismatch")
    require(ref["potential_sha256"] == gamma["potential_sha256"]
            == sha(directory / "Si_GTH_PBE_q4.gth"), "eight-k potential differs")
    require(ref["QE"]["scientific_converged"] and ref["QE"]["process_exit_code"] == 0
            and ref["QE"]["version"] == "7.6"
            and (directory / "exit-code.txt").read_text().strip() == "0", "eight-k run failed")
    for name, digest in ref["raw_sha256"].items():
        require(name in manifest and sha(directory / name) == digest, "eight-k raw hash mismatch")
    binary = re.fullmatch(r"([a-f0-9]{64})\s+(.+)\s*", (directory / "binary.sha256").read_text().strip())
    require(binary and binary[1] == ref["QE"]["binary_sha256"] == gamma["QE"]["binary_sha256"]
            and binary[2] == ref["QE"]["binary"]
            and (directory / "binary-path.txt").read_text().strip() == binary[2], "eight-k binary provenance")
    require((directory / "host.txt").read_text().strip() == ref["QE"]["host"]
            and "QuantumESPRESSO/7.6" in (directory / "modules.txt").read_text(), "eight-k environment")
    require(load(directory / "input-sha256.json") == {
        name: sha(directory / name) for name in ["input.in", "Si_GTH_PBE_q4.gth"]}, "eight-k input capture mismatch")
    inp, text = [(directory / n).read_text() for n in ["input.in", "qe.out"]]
    def field(name):
        found = re.findall(r"\b" + re.escape(name) + r"\s*=\s*([^,\n/]+)", inp, re.I)
        require(len(found) == 1, "eight-k missing/duplicate input field: " + name)
        return found[0].strip().strip("'\"").lower()
    for key, value in {"calculation": "scf", "input_dft": "sla+vwn+nogx+nogc",
                       "occupations": "fixed", "nosym": ".true.", "noinv": ".true.",
                       "diagonalization": "david"}.items():
        require(field(key) == value, "eight-k input model mismatch: " + key)
    for key, value in {"nat": 2, "ntyp": 1, "nspin": 1, "nbnd": 4, "ibrav": 0,
                       "ecutwfc": 12, "ecutrho": 48, "nr1": 24, "nr2": 28, "nr3": 24,
                       "nr1s": 24, "nr2s": 28, "nr3s": 24, "conv_thr": 1e-12,
                       "electron_maxstep": 250, "mixing_beta": .2, "diago_thr_init": 1e-12}.items():
        require(number(field(key)) == value, "eight-k input numeric mismatch: " + key)
    fixture = ref["fixture"]
    for key in ["H_columns_A", "positions_A"]:
        require(fixture[key] == gamma["fixture"][key], "eight-k locked geometry differs")
    for title, expected, species in [("CELL_PARAMETERS", fixture["H_columns_A"], False),
                                      ("ATOMIC_POSITIONS", fixture["positions_A"], True)]:
        start = re.search(title + r"\s+angstrom\s*\n", inp, re.I)
        require(start, "eight-k geometry section missing")
        rows = [r.split() for r in inp[start.end():].splitlines()[:len(expected)]]
        if species:
            require(all(r.pop(0) == "Si" for r in rows), "eight-k species")
        require([[number(x) for x in r] for r in rows] == expected, "eight-k input geometry")
    require(re.search(r"ATOMIC_SPECIES\s*\nSi\s+28\.0855\s+Si_GTH_PBE_q4\.gth\s*\n", inp),
            "eight-k species/potential declaration")
    start = re.search(r"K_POINTS\s+crystal\s*\n8\s*\n", inp, re.I)
    require(start, "eight-k explicit crystal mesh missing")
    points = [[a, b, c] for a in [0, .5] for b in [0, .5] for c in [0, .5]]
    rows = [[number(x) for x in r.split()] for r in inp[start.end():].splitlines() if r.strip()]
    require(rows == [p + [.125] for p in points] and fixture["k_mesh"] == 2
            and fixture["k_crystal_points"] == points and fixture["k_weights"] == [.125] * 8,
            "eight-k input points/weights changed")
    require("Program PWSCF v.7.6" in text and "JOB DONE." in text
            and "convergence NOT achieved" not in text, "eight-k raw scientific completion")
    require(re.findall(r"Number of MPI processes:\s*(\d+)", text) == ["1"]
            and re.findall(r"Threads/MPI process:\s*(\d+)", text) == ["1"]
            and ref["QE"]["MPI_ranks"] == ref["QE"]["OpenMP_threads"] == 1,
            "eight-k native resource provenance")
    require([s.strip() for s in re.findall(r"Exchange-correlation=\s*([^\n]+)", text)]
            == ["SLA+VWN+NOGX+NOGC"] * 2, "eight-k printed model is not VWN")
    require(re.findall(r"convergence has been achieved in\s+(\d+) iterations", text) == ["32"]
            and ref["QE"]["SCF_iterations"] == 32, "eight-k scientific convergence/steps")
    xml = ET.parse(directory / "data-file-schema.xml").getroot()
    require(xml.get("Units") == "Hartree atomic units"
            and xml.find("general_info/creator").get("VERSION") == "7.6", "eight-k XML release/units")
    out = xml.find("output")
    require(out.findtext("convergence_info/scf_conv/convergence_achieved") == "true"
            and out.findtext("convergence_info/scf_conv/n_scf_steps") == "32"
            and number(out.findtext("convergence_info/scf_conv/scf_error")) == ref["QE"]["scf_error_Ha"]
            and 0 <= ref["QE"]["scf_error_Ha"] <= .5e-12, "eight-k XML convergence")
    require(xml.findtext("input/dft/functional") == "SLA+VWN+NOGX+NOGC"
            and out.findtext("dft/functional") == "XC-001I-002I-000I-000I-000I-000I"
            and xml.findtext("input/bands/occupations") == "fixed"
            and number(xml.findtext("input/bands/nbnd")) == 4
            and all(xml.findtext("input/spin/" + k) == "false" for k in ["lsda", "noncolin", "spinorbit"])
            and all(xml.findtext("input/symmetry_flags/" + k) == "true" for k in ["nosym", "noinv"]),
            "eight-k XML model/occupations")
    basis = out.find("basis_set")
    require(basis.findtext("gamma_only") == "false" and number(basis.findtext("ecutwfc")) == 6
            and number(basis.findtext("ecutrho")) == 24 and basis.findtext("ngm") == "1393",
            "eight-k XML basis")
    require(xml.findtext("input/basis/gamma_only") == "false"
            and number(xml.findtext("input/basis/ecutwfc")) == 6
            and number(xml.findtext("input/basis/ecutrho")) == 24
            and number(xml.findtext("input/electron_control/conv_thr")) == .5e-12
            and number(xml.findtext("input/electron_control/mixing_beta")) == .2
            and number(xml.findtext("input/electron_control/max_nstep")) == 250
            and number(xml.findtext("input/electron_control/diago_thr_init")) == 1e-12,
            "eight-k XML input cutoffs/convergence settings")
    for key in ["fft_grid", "fft_smooth"]:
        require(basis.find(key).attrib == dict(zip(["nr1", "nr2", "nr3"], ["24", "28", "24"])),
                "eight-k XML FFT shape")
    bohr, hartree_ev = .529177210903, 4.3597447222071e-18 / 1.602176634e-19
    require(shared["units"] == {"Hartree_eV": hartree_ev, "Bohr_A": bohr}, "eight-k conversion constants")
    for location in ["input/atomic_structure", "output/atomic_structure"]:
        structure = xml.find(location)
        atoms = structure.findall("atomic_positions/atom")
        require(structure.get("nat") == "2" and [a.get("name") for a in atoms] == ["Si", "Si"]
                and [a.get("index") for a in atoms] == ["1", "2"], "eight-k XML species")
        actual = [[number(v) for v in a.text.split()] for a in atoms]
        cell = [[number(v) for v in structure.findtext("cell/a" + str(i)).split()] for i in [1, 2, 3]]
        for source, expected in [(actual, fixture["positions_A"]), (cell, fixture["H_columns_A"])]:
            require(all(len(row) == 3 for row in source)
                    and all(abs(a * bohr - b) < 5e-15 * max(1, abs(b))
                        for a, b in zip(flat(source), flat(expected))), "eight-k XML geometry/Bohr conversion")
    reciprocal = [[number(v) for v in basis.findtext("reciprocal_lattice/b" + str(i)).split()] for i in [1, 2, 3]]
    cart = [[sum(p[d] * reciprocal[d][a] for d in range(3)) for a in range(3)] for p in points]
    band = out.find("band_structure")
    require(band.findtext("nks") == "8" and band.findtext("nbnd") == "4"
            and number(band.findtext("nelec")) == 8 and band.findtext("occupations_kind") == "fixed"
            and all(band.findtext(k) == "false" for k in ["lsda", "noncolin", "spinorbit"]), "eight-k XML band/spin shape")
    for location in ["input/k_points_IBZ", "output/band_structure/starting_k_points"]:
        group = xml.find(location)
        kp = group.findall("k_point")
        require(group.findtext("nk") == "8" and len(kp) == 8
                and all(number(v.get("weight")) == .125 for v in kp), "eight-k geometric weights")
        coordinates = [[number(x) for x in v.text.split()] for v in kp]
        require(all(len(row) == 3 for row in coordinates)
                and max(abs(a-b) for a, b in zip(flat(coordinates), flat(cart))) < 2e-15,
                "eight-k XML Cartesian/crystal mapping")
    states = band.findall("ks_energies")
    require(len(states) == 8 and [int(s.findtext("npw")) for s in states] == fixture["QE_wave_PW_each_k"], "eight-k actual PW counts")
    for i, state in enumerate(states):
        require(number(state.find("k_point").get("weight")) == .25
                and [number(v) for v in state.findtext("occupations").split()] == [1] * 4
                and len([number(v) for v in state.findtext("eigenvalues").split()]) == 4, "eight-k spin-weight/occupation convention")
        coordinate = [number(v) for v in state.findtext("k_point").split()]
        require(len(coordinate) == 3 and max(abs(a-b) for a, b in zip(coordinate, cart[i])) < 2e-15,
                "eight-k output wavevector mismatch")
    printed_k = re.findall(r"k\(\s*(\d+)\)\s*=\s*\(([^)]+)\),\s*wk\s*=\s*(" + NUMBER + ")", text)
    require(len(printed_k) == 16 and [int(v[0]) for v in printed_k] == list(range(1, 9)) * 2
            and all(number(v[2]) == .25 for v in printed_k), "eight-k printed k mesh/weights")
    for group, expected in [(printed_k[:8], cart), (printed_k[8:], points)]:
        coordinates = [[number(x) for x in v[1].split()] for v in group]
        require(all(len(row) == 3 for row in coordinates)
                and max(abs(a-b) for a, b in zip(flat(coordinates), flat(expected))) < 5.1e-8,
                "eight-k printed wavevectors")
    energy = number(out.findtext("total_energy/etot"))
    force, stress = [[number(v) for v in out.findtext(k).split()] for k in ["forces", "stress"]]
    values = ref["reference"]
    require(len(force) == 6 and len(stress) == 9 and energy == values["energy_Ha"]
            and force == flat(values["forces_Ha_Bohr"]) and stress == flat(values["stress_tensor_Ha_Bohr3"]), "eight-k XML/reference values")
    eprint = re.findall(r"!\s+total energy\s*=\s*(" + NUMBER + ")\s+Ry", text)
    require(text.count("Forces acting on atoms") == 1, "eight-k total-force section missing/duplicate")
    total_force_block = text.split("Forces acting on atoms")[1].split("The non-local contrib")[0]
    fprint = re.findall(r"atom\s+(\d+)\s+type\s+1\s+force\s*=\s*(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER + ")", total_force_block)
    sprint = re.search(r"total\s+stress\s+\(Ry/bohr\*\*3\)[^\n]*\n((?:[^\n]*\n){3})", text)
    require(len(eprint) == 1 and abs(number(eprint[0]) / 2 - energy) < 2.51e-9
            and [v[0] for v in fprint] == ["1", "2"] and sprint, "eight-k printed totals incomplete")
    require(max(abs(number(a)/2-b) for a, b in zip([x for row in fprint for x in row[1:]], force)) < 2.51e-9
            and max(abs(number(a)/2-b) for a, b in zip([x for row in sprint[1].splitlines() for x in row.split()[:3]], stress)) < 2.51e-9,
            "eight-k printed/XML force/stress mismatch")
    # Preserve the k8 header's explicit precomputed unit-factor convention.
    converted = [stress[i] * (hartree_ev / bohr**3) for i in [0, 4, 8, 1, 2, 5]]
    require(converted == values["stress_voigt_eV_A3"], "eight-k stress units/Voigt")
    header = directory.parent / "k8_pins.h"
    require(ref["pins_header"] == "../k8_pins.h" and sha(header) == ref["pins_header_sha256"], "eight-k header digest")
    actual = []
    for name in ["energy_ha", "forces_ha_bohr", "stress_ev_ang3"]:
        match = re.search(r"\b" + name + r"\s*=\s*([^;]+);", header.read_text())
        require(match, "eight-k pin missing")
        actual.extend(tokens(match[1]))
    require(actual == [repr(v) for v in [energy, *force, *converted]], "eight-k pin tokens differ from XML")
    return ref, {"manifest_files": len(manifest), "manifest_sha256": sha(directory / "SHA256SUMS.json"),
                 "XML_raw_pins_model_units_verified": True, "shared_Gamma_provenance_verified": True,
                 "eight_points_and_weight_conventions_verified": True}


def native_scope(text):
    passed = re.findall(r"^PASS PW_ANALYTIC_STRESS (.*)$", text, re.M)
    require(len(passed) == 1, "require one unique complete native PASS marker; split MPI rank logs")
    old = re.fullmatch(r"(GammaRKS_fixed_LSDA_Si2|K8RKS_fixed_LSDA_Si2(?:_rotation_z90)?) "
                       r"public=3 nested_combined=1 external=QE7.6_Slater_VWN5 E_Ha=(" + NUMBER
                       + r") gates_Ha_HaBohr_eVA3=1e-5,1e-4,2e-3", passed[0])
    if old:
        return {"mesh": 1 if old[1].startswith("Gamma") else 2,
                "rotated": int(old[1].endswith("rotation_z90")), "energy": number(old[2]),
                "legacy_Gamma": old[1].startswith("Gamma")}
    new = re.fullmatch(r"RKS_fixed_LSDA_Si2 public=3 nested_combined=1 external=QE7.6_Slater_VWN5 "
                       r"mesh=([12]) rotated=([01]) E_Ha=(" + NUMBER
                       + r") gates_Ha_HaBohr_eVA3=1e-5,1e-4,2e-3", passed[0])
    require(new, "native scope/gates mismatch")
    return {"mesh": int(new[1]), "rotated": int(new[2]), "energy": number(new[3]), "legacy_Gamma": False}


def check_native(log, reference, *, rank=None, ranks=None):
    text = log.read_text()
    require(not re.search(r"^FAIL PW_ANALYTIC_STRESS", text, re.M), "native failure marker present")
    scope = native_scope(text)
    require((rank is None) == (ranks is None), "supply both rank and ranks")
    if ranks is not None:
        require(ranks > 0 and 0 <= rank < ranks, "invalid per-rank identity")
    if scope["mesh"] == 2 and reference["fixture"].get("k_mesh", 1) != 2:
        reference, _ = check_k8_reference(Path(__file__).resolve().parent / "qe-k8-2026-10-28")
    require(reference["fixture"].get("k_mesh", 1) == scope["mesh"], "native/reference mesh mismatch")
    pattern = (r"^SCF_AUDIT (\S+) solves=(\d+) builds=(\d+) destroys=(\d+) guard_builds=(\d+) "
               r"guard_destroys=(\d+) iterations=(\d+) polls=(\d+) residual_first=(" + NUMBER
               + r") residual_last=(" + NUMBER + r")$")
    audits = re.findall(pattern, text, re.M)
    require([a[0] for a in audits] == LABELS, "require exactly four unique complete SCF audits in order")
    parsed = []
    for row in audits:
        require(all(int(x) == 1 for x in row[1:6]), "SCF lifecycle count is not exactly one")
        iterations, polls = int(row[6]), int(row[7])
        first, last = number(row[8]), number(row[9])
        require(iterations == polls and iterations > 2 and first > 1e-10 and 0 <= last < 1e-10,
                "incomplete collector or nonphysical residual trajectory")
        parsed.append({"label": row[0], "SCF_solves": 1, "iterations": iterations, "polls": polls,
                       "residual_first": first, "residual_last": last})
    require(len({a["iterations"] for a in parsed}) == 1, "run/derivatives/nesting changed reference iterations")
    energy = scope["energy"]
    rows = re.findall(r"^PUBLIC_FORCE_HaBohr atom=(\d+) (" + NUMBER + ") (" + NUMBER
                      + ") (" + NUMBER + r")$", text, re.M)
    require([r[0] for r in rows] == ["0", "1"], "native total-force data incomplete/duplicate")
    forces = [[number(x) for x in row[1:]] for row in rows]
    stresses = re.findall(r"^PUBLIC_STRESS_eVA3 ([^\n]+)$", text, re.M)
    require(len(stresses) == 1 and len(stresses[0].split()) == 6, "native stress data incomplete/duplicate")
    stress = [number(x) for x in stresses[0].split()]
    expected = dict(reference["reference"])
    if scope["rotated"]:
        expected["forces_Ha_Bohr"] = [[-f[1], f[0], f[2]] for f in expected["forces_Ha_Bohr"]]
        s = expected["stress_voigt_eV_A3"]
        expected["stress_voigt_eV_A3"] = [s[1], s[0], s[2], -s[3], -s[5], s[4]]
    errors = {"energy_Ha": abs(energy - expected["energy_Ha"]),
              "each_force_Ha_Bohr": [abs(a-b) for a, b in zip(flat(forces), flat(expected["forces_Ha_Bohr"]))],
              "each_stress_eV_A3": [abs(a-b) for a, b in zip(stress, expected["stress_voigt_eV_A3"])]}
    require(errors["energy_Ha"] <= GATES["energy_Ha"], "native energy exceeds original QE gate")
    for name in ["each_force_Ha_Bohr", "each_stress_eV_A3"]:
        require(max(errors[name]) <= GATES[name], "native " + name + " exceeds original QE gate")
    result = {"audits": parsed, "energy_Ha": energy, "forces_Ha_Bohr": forces,
              "stress_eV_A3": stress, "absolute_errors": errors, "gates": GATES}
    band_rows = re.findall(r"^BAND_AUDIT (\S+) local_k=(\d+) global_k=(\d+) Davidson=(\d+) "
                           r"extract=(\d+) assembly=(\d+) legacy=(\d+)$", text, re.M)
    require(len(re.findall(r"^BAND_AUDIT\b", text, re.M)) == len(band_rows), "malformed BAND_AUDIT")
    if band_rows:
        require([r[0] for r in band_rows] == LABELS, "require exactly four ordered BAND_AUDIT records")
        band_audits = []
        for row, scf in zip(band_rows, parsed):
            local, global_k, davidson, extract, assembly, legacy = map(int, row[1:])
            derivative = row[0] != "public-run"
            stress_request = derivative and row[0] != "public-forces"
            require(global_k == scope["mesh"] ** 3 and 0 <= local <= global_k
                    and extract == int(derivative) and assembly == int(stress_request) and legacy == 0
                    and davidson == local * (scf["iterations"] + int(derivative)), "incorrect exact local band budget")
            if ranks is None:
                require(local == global_k, "distributed band audit requires explicit rank and ranks")
            else:
                require(local == global_k // ranks + int(rank < global_k % ranks), "band audit rank ownership mismatch")
            band_audits.append(dict(zip(["label", "local_k", "global_k", "Davidson", "extract", "assembly", "legacy"],
                                        [row[0], local, global_k, davidson, extract, assembly, legacy])))
        result["band_audits"] = band_audits
    if not scope["legacy_Gamma"] or band_rows or ranks is not None:
        result["execution_scope"] = {"mesh": scope["mesh"], "rotated": scope["rotated"],
                                      "rank": rank, "ranks": ranks,
                                      "exact_band_budget_measured": bool(band_rows),
                                      "log_scope": "one process/rank only; merged MPI logs rejected"}
    return result


def inventory(path):
    data = load(path)
    if isinstance(data, dict) and isinstance(data.get("files"), dict):
        data = data["files"]
    require(isinstance(data, dict) and data and all(isinstance(k, str) and isinstance(v, str)
            and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in data.items()), "invalid source hash inventory")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["log", "source-before", "source-after", "binary", "hardware", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binaries-sha256", type=Path)
    parser.add_argument("--rank", type=int, help="MPI rank owning this single log")
    parser.add_argument("--ranks", type=int, help="MPI world size; supply with --rank")
    args = parser.parse_args()
    gamma_package = Path(__file__).resolve().parent / "qe-2026-10-28"
    k8_package = Path(__file__).resolve().parent / "qe-k8-2026-10-28"
    package = gamma_package
    inputs = [args.log, args.source_before, args.source_after, args.binary, args.hardware]
    if args.binaries_sha256:
        inputs.append(args.binaries_sha256)
    output = args.output.resolve()
    if output in [p.resolve() for p in inputs] + [k8_package.parent / "k8_pins.h"] or any(p == output or p in output.parents for p in [gamma_package, k8_package]):
        print("unsafe output path would overwrite input/reference evidence", file=sys.stderr)
        return 1
    result = {"schema": "pw-analytic-stress-scoped-validation-v1", "backend": "cpu",
              "formal_source": False, "formal_build": False, "formal_source_build": False,
              "scope": "per-rank public/nested fixed LSDA Gamma/eight-k plus exact-QE gates; not broad formal source/build qualification",
              "collector": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}}
    try:
        scope = native_scope(args.log.read_text())
        if scope["mesh"] == 2:
            package = k8_package
            reference, integrity = check_k8_reference(package)
        else:
            reference, integrity = check_reference(package)
        result["reference_integrity"] = integrity
        result["native"] = check_native(args.log, reference, rank=args.rank, ranks=args.ranks)
        before, after = inventory(args.source_before), inventory(args.source_after)
        require(before == after and load(args.source_before) == load(args.source_after),
                "source-before and source-after inventories differ")
        result["source_inventory"] = {"equal": True, "files": len(before),
                                      "before_sha256": sha(args.source_before), "after_sha256": sha(args.source_after)}
        with args.binary.open("rb") as executable:
            require(executable.read(4) == b"\x7fELF", "native binary is not an ELF executable")
        digest = sha(args.binary)
        if args.binaries_sha256:
            rows = re.findall(r"^([0-9a-f]{64})\s+\*?(.+)$", args.binaries_sha256.read_text(), re.M)
            matches = [h for h, p in rows if Path(p).resolve() == args.binary.resolve()
                       or (args.binaries_sha256.parent / p).resolve() == args.binary.resolve()]
            require(matches == [digest], "native binary differs from supplied frozen binary inventory")
        hardware = args.hardware.read_text()
        require(hardware.strip(), "hardware evidence empty")
        result["binary"] = {"path": str(args.binary.resolve()), "sha256": digest}
        result["hardware"] = {"path": str(args.hardware.resolve()), "sha256": sha(args.hardware), "raw": hardware}
        result["main_log"] = {"path": str(args.log.resolve()), "sha256": sha(args.log)}
        provenance = gamma_package / "math-provenance"
        result["prior_mathematical_evidence"] = {
            "attribution": "Earlier three-tolerance/72 fixed-basis SCF receipts keep their own source/binary hashes; not attributed to current binary.",
            "files": {p.name: sha(p) for p in provenance.iterdir() if p.is_file()}}
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "fail"
        result["error"] = str(error)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"].upper(), output, result.get("error", ""))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
