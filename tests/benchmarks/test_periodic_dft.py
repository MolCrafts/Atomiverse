"""Parser checks using synthetic output, never scientific acceptance pins."""

import pytest

from benchmarks.periodic.adapters import atv_observables, cp2k_observables
from benchmarks.periodic.bench_gpw import SPEC
from benchmarks.scf.hf import ScientificFailure


def test_cp2k_uses_unstrained_energy_and_every_force_row():
    system = SPEC.rows[3].systems[0]
    output = """
 CP2K| version string: CP2K version 2026.2
 Number of electrons: 2
 Number of electrons: 1
 *** SCF run converged in 12 steps ***
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -1.0
 ******************* NUMERICAL STRESS END *******************
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -2.0
 FORCES| Atomic forces [hartree/bohr]
 FORCES|      1  1.0  2.0  3.0  3.74
 FORCES|      2  4.0  5.0  6.0  8.77
 FORCES|      3  7.0  8.0  9.0 13.93
 FORCES| Sum 12.0 15.0 18.0
 STRESS| Numerical stress tensor [bar]
 STRESS|      x  1.0 10.0 20.0
 STRESS|      y 10.0  2.0 30.0
 STRESS|      z 20.0 30.0  3.0
 STRESS| 1/3 Trace 2.0
 PROGRAM ENDED AT
"""
    version, data = cp2k_observables(output, system, stress_kind="Numerical")
    assert version == "2026.2"
    assert data["energy_hartree"] == -2
    assert data["forces_hartree_per_bohr"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    xx, yy, zz = data["stress"]
    assert xx > 0
    assert [yy / xx, zz / xx] == pytest.approx([2, 3])
    analytical = output[output.index(" CP2K|"):].replace(
        "NUMERICAL STRESS END", "analytical calculation"
    ).replace("Numerical stress tensor", "Analytical stress tensor")
    analytical = analytical.replace(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -1.0", ""
    )
    assert cp2k_observables(analytical, system, stress_kind="Analytical")[1] == data
    with pytest.raises(ScientificFailure, match="all ordered atoms"):
        cp2k_observables(output.replace("FORCES|      2", "FORCES|      3"), system,
                         stress_kind="Numerical")


def test_boundary_h2_has_cuda_and_reference_checks_for_both_images():
    row = next(row for row in SPEC.rows if row.id == "gpw.h2-rks-boundary.pbesol")
    assert {p.id for p in row.participants} == {"atv.cpu", "atv.cuda", "cp2k.cpu"}
    assert row.model["first_atom_image"] == [-1, -1, -1]
    assert {c.observable for c in row.checks} == {
        "energy_hartree", "forces_hartree_per_bohr", "stress",
        "translated_energy_hartree", "translated_forces_hartree_per_bohr",
        "translated_stress",
    }
    record = "-1.2 .2 .1 .3 -.2 -.1 -.3 .004 -.005 .006"
    data = atv_observables(
        f"ATV_PERIODIC_RESULT {record}\nATV_PERIODIC_TRANSLATED {record}\n",
        row, row.systems[0],
    )
    assert data["translated_forces_hartree_per_bohr"] == [[.2, .1, .3], [-.2, -.1, -.3]]
    assert data["translated_stress"] == [.004, -.005, .006]


@pytest.mark.parametrize("index, observable", [(0, "energy_hartree"),
                                                (6, "forces_hartree_per_bohr"),
                                                (9, "stress")])
def test_boundary_h2_rejects_disagreement_between_images(index, observable):
    row = next(row for row in SPEC.rows if row.id == "gpw.h2-rks-boundary.pbesol")
    record = "-1.2 .2 .1 .3 -.2 -.1 -.3 .004 -.005 .006"
    shifted = record.split()
    shifted[index] = str(float(shifted[index]) + .001)
    with pytest.raises(ScientificFailure, match=f"translation changed {observable}"):
        atv_observables(
            f"ATV_PERIODIC_RESULT {record}\n"
            f"ATV_PERIODIC_TRANSLATED {' '.join(shifted)}\n",
            row, row.systems[0],
        )


def test_same_grid_displacement_rejects_stale_observables():
    row = next(row for row in SPEC.rows if row.id == "gpw.h2-rks-moved.pbesol")
    assert row.model["first_atom_seed_shift_bohr"] == -0.4
    initial = "-1.2 .2 .1 .3 -.2 -.1 -.3 .004 -.005 .006"
    changed = "-1.3 .25 .1 .3 -.25 -.1 -.3 .003 -.005 .006"
    output = f"ATV_PERIODIC_INITIAL {initial}\nATV_PERIODIC_RESULT {changed}\n"
    assert atv_observables(output, row, row.systems[0])["energy_hartree"] == -1.3
    with pytest.raises(ScientificFailure, match="Displaced geometry did not change"):
        atv_observables(output.replace(changed, initial), row, row.systems[0])


def test_multigrid_h2_reads_controls_from_cp2k_input():
    row = next(row for row in SPEC.rows if row.id == "gpw.h2-rks-multigrid.pbesol")
    assert row.model["ngrids"] == 3
    assert row.model["rel_cutoff"] == 96
    assert row.model["cutoff_ry"] == 384
    assert row.model["stress_components"] == ["xx", "yy", "zz"]
    assert {p.id for p in row.participants} == {"atv.cpu", "atv.cuda", "cp2k.cpu"}


def test_si_bloch_row_declares_full_mesh_and_analytical_reference_stress():
    row = next(row for row in SPEC.rows if row.id == "gpw.si4-rks-kpoints.pbe")
    assert row.model["k_mesh"] == [1, 1, 2]
    assert row.model["ngrids"] == 1
    assert row.model["expected_electrons"] == 16
    assert row.model["basis_set_file"] == "BASIS_MOLOPT"
    assert row.model["stress_kind"] == "Analytical"


def test_si_bloch_multigrid_row_runs_on_both_backends_and_keeps_the_full_mesh():
    row = next(row for row in SPEC.rows if row.id == "gpw.si4-rks-kpoints-multigrid.pbe")
    assert row.model["k_mesh"] == [1, 1, 2]
    assert row.model["ngrids"] == 3
    assert 0 < row.model["rel_cutoff"] < row.model["cutoff_ry"]
    assert {p.id for p in row.participants} == {"atv.cpu", "atv.cuda", "cp2k.cpu"}


def test_h3_bloch_uks_row_uses_a_full_mesh_on_cpu():
    row = next(row for row in SPEC.rows if row.id == "gpw.h3-uks-kpoints.pbe")
    assert row.systems[0].spin == 1
    assert row.model["expected_electrons"] == 3
    assert row.model["k_mesh"] == [1, 1, 2]
    assert row.model["ngrids"] == 1
    assert row.model["stress_kind"] == "Analytical"
    assert {p.id for p in row.participants} == {"atv.cpu", "cp2k.cpu"}
