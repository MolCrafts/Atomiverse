"""AM1-BCC's molecular input is read-only and charges follow graph atom order."""

from __future__ import annotations

import gc

import atomiverse as atv
import molpy as mp
import numpy as np
import pytest


def _tetrahedron(atomic_number: int, formal_charge: int = 0) -> mp.Atomistic:
    """The central atom is deliberately between hydrogen rows."""
    mol = mp.Atomistic()
    mol.props["source"] = "hand-drawn tetrahedron"
    h = 0.63 if atomic_number == 6 else 0.59
    sites = [
        (1, h, h, h),
        (1, -h, -h, h),
        (atomic_number, 0.0, 0.0, 0.0),
        (1, -h, h, -h),
        (1, h, -h, -h),
    ]
    atoms = [
        mol.add_atom(
            atomic_number=z,
            x=x,
            y=y,
            z=z_pos,
            label=f"site-{i}",
            type="keep-my-type",
            charge=0.125 * i,
            formal_charge=formal_charge if i == 2 else 0,
        )
        for i, (z, x, y, z_pos) in enumerate(sites)
    ]
    for i in (0, 1, 3, 4):
        mol.add_bond(atoms[2], atoms[i], bond_type=1, bond_number=1, label=f"bond-{i}")
    return mol


def _snapshot(mol: mp.Atomistic):
    return (
        dict(mol.props),
        [dict(atom) for atom in mol.atoms],
        [dict(bond) for bond in mol.bonds],
    )


@pytest.mark.parametrize("correction", ["bcc", "abcg2"])
def test_methane_charges_follow_graph_order_without_modifying_chemistry(correction):
    mol = _tetrahedron(6)
    before = _snapshot(mol)
    model = atv.AM1BCCCharges(method=atv.AM1(charge=0), correction=correction)

    charges = model.assign(mol)

    assert isinstance(charges, np.ndarray)
    assert charges.dtype == np.float64
    assert charges.shape == (5,)
    assert np.isfinite(charges).all()
    assert _snapshot(mol) == before
    assert charges[2] < 0.0
    assert np.all(charges[[0, 1, 3, 4]] > 0.0)
    # Tetrahedral methane has a single hydrogen equivalence class.
    np.testing.assert_allclose(charges[[0, 1, 3, 4]], charges[0], rtol=0, atol=1e-12)
    assert abs(float(charges.sum())) < 1e-10


def test_charged_graph_keeps_formal_and_existing_partial_charges():
    mol = _tetrahedron(7, formal_charge=1)
    before = _snapshot(mol)
    model = atv.AM1BCCCharges(method=atv.AM1(charge=1), correction="bcc")

    charges = model.assign(mol)

    assert _snapshot(mol) == before
    assert np.isfinite(charges).all()
    assert abs(float(charges.sum()) - 1.0) < 1e-10
    np.testing.assert_allclose(charges[[0, 1, 3, 4]], charges[0], rtol=0, atol=1e-12)


def test_returned_charges_outlive_the_model_and_are_not_its_work_buffer():
    model = atv.AM1BCCCharges(method=atv.AM1(charge=0), correction="bcc")
    first = model.assign(_tetrahedron(6))
    saved = first.copy()
    second = model.assign(_tetrahedron(6))
    assert not np.shares_memory(first, second)
    first[0] = 123.0
    np.testing.assert_array_equal(second, saved)
    del model
    gc.collect()
    np.testing.assert_array_equal(second, saved)


def test_unknown_correction_is_refused():
    with pytest.raises(ValueError, match="correction|bcc|abcg2"):
        atv.AM1BCCCharges(correction="gaff2")


def test_frame_is_not_accepted_as_a_molecular_charge_input():
    model = atv.AM1BCCCharges(correction="bcc")
    with pytest.raises(TypeError, match="Atomistic|graph|molecule"):
        model.assign(_tetrahedron(6).to_frame())


def test_generic_graph_is_not_an_atomistic_charge_input():
    model = atv.AM1BCCCharges(correction="bcc")
    with pytest.raises(TypeError, match="Atomistic|graph|molecule"):
        model.assign(mp.MolGraph())


def test_missing_coordinates_are_refused_without_mutation():
    mol = mp.Atomistic()
    atoms = [mol.add_atom(atomic_number=1) for _ in range(2)]
    mol.add_bond(*atoms, bond_type=1, bond_number=1)
    before = _snapshot(mol)
    with pytest.raises(ValueError, match="coordinate|position|x"):
        atv.AM1BCCCharges(correction="bcc").assign(mol)
    assert _snapshot(mol) == before


def test_a_coordinate_missing_on_only_one_atom_is_not_replaced_by_zero():
    mol = mp.Atomistic()
    first = mol.add_atom(atomic_number=1, x=0.0, y=0.0, z=0.0)
    second = mol.add_atom(atomic_number=1, x=0.74, y=0.0)
    mol.add_bond(first, second, bond_type=1, bond_number=1)
    before = _snapshot(mol)
    with pytest.raises(ValueError, match="coordinate|position|z"):
        atv.AM1BCCCharges(correction="bcc").assign(mol)
    assert _snapshot(mol) == before


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coordinates_are_refused(coordinate):
    mol = mp.Atomistic()
    first = mol.add_atom(atomic_number=1, x=0.0, y=0.0, z=0.0)
    second = mol.add_atom(atomic_number=1, x=coordinate, y=0.0, z=0.0)
    mol.add_bond(first, second, bond_type=1, bond_number=1)
    with pytest.raises(ValueError, match="finite|coordinate|position"):
        atv.AM1BCCCharges(correction="bcc").assign(mol)


def test_multiatom_graph_without_connectivity_is_refused_without_mutation():
    mol = mp.Atomistic()
    mol.add_atom(atomic_number=1, x=0.0, y=0.0, z=0.0)
    mol.add_atom(atomic_number=1, x=0.74, y=0.0, z=0.0)
    before = _snapshot(mol)
    with pytest.raises(ValueError, match="bond|connect"):
        atv.AM1BCCCharges(correction="bcc").assign(mol)
    assert _snapshot(mol) == before
