"""Registration and ownership contracts; no scientific solve is run."""

import sys

import pytest

from _domains import needs


@pytest.mark.parametrize("workflow", ["md", "minimize"])
@pytest.mark.parametrize("method", ["RHF", "UHF", "RKS", "UKS"])
@needs("ATV_BUILD_MD", "ATV_BUILD_MINIMIZER", "ATV_BUILD_HF")
def test_mean_field_node_registration_keeps_owner(workflow, method):
    import atomiverse as av

    driver = av.MD(dt=0.25) if workflow == "md" else av.Minimize()
    if workflow == "minimize":
        driver.set_optimizer(av.LBFGS())
    evaluator = getattr(av, method)()
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
    evaluator = av.DFTB2()
    before = sys.getrefcount(evaluator)
    driver.set_evaluator(evaluator)
    assert sys.getrefcount(evaluator) > before
    del evaluator
    driver.build()
