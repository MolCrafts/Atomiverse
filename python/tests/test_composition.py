"""Registration and ownership contracts; no scientific solve is run."""

import sys

import pytest

from _domains import needs


#: Constructor arguments per Gaussian method; the identity is mandatory.
_MEAN_FIELD_ARGS = {
    "RHF": {"basis": "STO-3G"},
    "UHF": {"basis": "STO-3G", "n_alpha": 1, "n_beta": 1},
    "RKS": {"basis": "STO-3G", "xc": "PBE"},
    "UKS": {"basis": "STO-3G", "n_alpha": 1, "n_beta": 1},
}


@pytest.mark.parametrize("workflow", ["md", "minimize"])
@pytest.mark.parametrize("method", sorted(_MEAN_FIELD_ARGS))
@needs("ATV_BUILD_MD", "ATV_BUILD_MINIMIZER", "ATV_BUILD_HF")
def test_mean_field_node_registration_keeps_owner(workflow, method):
    import atomiverse as av

    driver = av.MD(dt=0.25) if workflow == "md" else av.Minimize()
    if workflow == "minimize":
        driver.set_optimizer(av.LBFGS())
    evaluator = getattr(av, method)(**_MEAN_FIELD_ARGS[method])
    before = sys.getrefcount(evaluator)
    driver.set_evaluator(evaluator)
    assert sys.getrefcount(evaluator) > before
    del evaluator
    driver.build()


@pytest.mark.parametrize("workflow", ["md", "minimize"])
@needs("ATV_BUILD_MD", "ATV_BUILD_MINIMIZER", "ATV_BUILD_SEMIEMPIRICAL")
def test_dftb_component_registration_keeps_owner(workflow):
    import atomiverse as av

    driver = av.MD(dt=0.25) if workflow == "md" else av.Minimize()
    if workflow == "minimize":
        driver.set_optimizer(av.LBFGS())
    evaluator = av.DFTB2(av.DftbSkSet.mio)
    before = sys.getrefcount(evaluator)
    driver.set_evaluator(evaluator)
    assert sys.getrefcount(evaluator) > before
    del evaluator
    driver.build()


@needs("ATV_BUILD_SEMIEMPIRICAL")
def test_dftb_evaluate_registration_is_a_valid_component():
    """DFTB2 is bound with a Component base; that upcast must be exact."""
    import atomiverse as av

    evaluation = av.Evaluate()
    evaluator = av.DFTB2(av.DftbSkSet.mio)
    evaluation.set_evaluator(evaluator)
    del evaluator
    evaluation.build()
