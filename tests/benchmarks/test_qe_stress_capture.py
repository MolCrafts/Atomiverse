"""Oracle-output parsing only; these synthetic numbers are not physics pins."""

import importlib

import pytest

from benchmarks.periodic.adapters import qe_observables
from benchmarks.periodic.bench_pw import SPEC

capture = importlib.import_module("tools.qe_oracle.inputs.pw-spin-smear-stress.capture")


def test_verbose_qe_force_components_cannot_be_mistaken_for_total_forces():
    output = """
     Program PWSCF v.7.6 starts
     convergence has been achieved in 30 iterations
!    total energy = -2.5 Ry
     total magnetization = 0.9 Bohr mag/cell
     smearing contrib. (-TS) = -0.004 Ry
     Forces acting on atoms (cartesian axes, Ry/au):

     atom 1 type 1 force = 0.1 0.2 0.3
     atom 2 type 1 force = -0.1 -0.2 -0.3
     The local contribution to forces
     atom 1 type 1 force = 10 20 30
     atom 2 type 1 force = -10 -20 -30
     Total force = 0.5
     total stress (Ry/bohr**3) (kbar) P=0
     1 4 5 0 0 0
     4 2 6 0 0 0
     5 6 3 0 0 0
     JOB DONE.
"""
    result = capture.parse(output)
    assert result["converged"]
    assert result["forces_ry_bohr"] == [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    assert result["energy_ry"] == -2.5
    assert result["spin_nontrivial"] and result["entropy_nontrivial"]
    # The Voigt map must take xy,xz,yz, independently of row flattening.
    stress = result["stress_voigt_ev_ang3"]
    assert [v / stress[0] for v in stress] == pytest.approx([1, 2, 3, 4, 5, 6])
    _, observed = qe_observables(output, SPEC.rows[0].systems[0], fermi_dirac=True)
    assert observed["energy_hartree"] == -1.25
    assert observed["forces_hartree_per_bohr"] == [
        [0.05, 0.1, 0.15],
        [-0.05, -0.1, -0.15],
    ]
    assert observed["stress"] == pytest.approx(stress)


def test_unfinished_external_output_cannot_claim_convergence_or_stress():
    result = capture.parse("Program PWSCF v.7.6\n! total energy = -2 Ry\n")
    assert not result["converged"]
    assert result["stress_tensor_ry_bohr3"] is None
    assert result["forces_ry_bohr"] == []
