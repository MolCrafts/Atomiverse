"""Structural tests for the typed CPU Minimize Python surface."""

from __future__ import annotations

import pytest

import atomiverse as av
from atomiverse.minimizer import ForceConverged, LBFGS, Minimize, MinimizeResult
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut


def make_minimize() -> Minimize:
    opt = Minimize(
        convergence=ForceConverged(
            ftol=0.05,
            rms_ftol=0.03,
            max_dtol=0.002,
            rms_dtol=0.001,
            etol=1e-5,
        )
    )
    opt.add(LBFGS(memory=5, alpha0=0.01, max_ls=10))
    opt.on("force").add(NeighborList(8.5, 0.3), order=-10).add(
        LJCut(0.0104, 3.4, 8.5)
    )
    return opt


def test_canonical_identity_exports() -> None:
    assert av.Minimize is Minimize
    assert av.LBFGS is LBFGS
    assert av.ForceConverged is ForceConverged
    assert av.MinimizeResult is MinimizeResult


def test_typed_composition_builds() -> None:
    opt = make_minimize()
    opt.build()
    assert "built" in repr(opt)


def test_wrong_phase_fails_before_build() -> None:
    with pytest.raises(ValueError, match="Unknown minimizer phase"):
        Minimize().on("md.force")


def test_gpu_is_explicitly_deferred() -> None:
    with pytest.raises(ValueError, match="(?i)gpu|cuda"):
        Minimize(device="cuda")


def test_run_returns_control_result_and_mutates_same_frame(argon_frame: object) -> None:
    opt = make_minimize()
    result = opt.run(argon_frame, 1)
    assert isinstance(result, MinimizeResult)
    assert result.iterations == 1
    assert isinstance(result.converged, bool)
    assert isinstance(result.stop_reason, str)
