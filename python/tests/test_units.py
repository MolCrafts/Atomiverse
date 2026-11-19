"""Public data boundaries preserve physical meaning without Frame methods."""
import numpy as np
import pytest
from _domains import skip_module_unless
skip_module_unless("ATV_BUILD_MD")
import molpy
from atomiverse import units
from atomiverse.md import MD, FixEfield
from atomiverse.md.integrator import VelocityVerlet


def frame():
    result=molpy.Frame()
    atoms=molpy.Block()
    for axis in "xyz":
        atoms.insert(axis,np.array([1.,2.]))
    atoms.insert("mass",np.array([1.,1.]))
    atoms.insert("charge",np.array([1.,-1.]))
    result["atoms"]=atoms
    result.box=molpy.Box.cube(20.)
    units.annotate(result,"real")
    return result


def test_copy_conversion_preserves_source():
    source=frame()
    converted=units.convert(source,"si")
    assert source.meta["units_style"]=="real"
    np.testing.assert_array_equal(np.asarray(source["atoms"]["x"]),[1.,2.])
    np.testing.assert_allclose(np.asarray(converted["atoms"]["x"]),[1.e-10,2.e-10])
    assert converted.meta["units_style"]=="si"


def test_lj_reference_is_explicit_and_round_trips():
    source=frame()
    with pytest.raises(ValueError,match="reference"):
        units.convert(source,"lj")
    ref=units.ReducedReference("argon",6.6335209e-26,3.4e-10,1.654017e-21)
    reduced=units.convert(source,"lj",reference=ref)
    assert reduced.meta["units_lj_identity"]=="argon"
    restored=units.convert(reduced,"real")
    np.testing.assert_allclose(np.asarray(restored["atoms"]["x"]),[1.,2.])


def test_run_import_and_writeback_adapt_automatically():
    source=frame()
    run=MD(units="metal")
    run.set_integrator(VelocityVerlet(0.))
    run.set_evaluator(FixEfield(1.,0.,0.))
    run.run(source,0)
    assert source.meta["units_style"]=="real"
    np.testing.assert_allclose(np.asarray(source["atoms"]["fx"]),[23.060547830619,-23.060547830619])


def test_from_style_exposes_the_runs_thermal_scale():
    reduced=units.from_style("lj")
    metal=units.from_style("metal")
    assert reduced.style == "lj"
    assert reduced.thermal_energy(2.) == 2.
    assert metal.style == "metal"
    assert metal.thermal_energy(300.) == pytest.approx(300.*metal.boltzmann_constant)
    assert metal.boltzmann_constant == pytest.approx(8.617333262145e-5)
