"""Driver entry points — one-shot evaluations that need no loop of their own.

Canonical deep path: ``atomiverse.driver.single_point``; top-level
``atomiverse.single_point`` is the same function. Its C++ counterpart is
``cpu::single_point`` (``cpu/evaluate/single_point.h``), the helper that
runs one pass of the ``Evaluate`` driver preset.
"""

from atomiverse._binding import _cpp, has_domain, require_cpp


def single_point(frame: object, evaluator: object, device: str = "cpu") -> None:
    """Compute energy (and any evaluator-specific properties) for one fixed
    geometry — a *single point*, as opposed to a trajectory or an
    optimization.

    Energy lands in the frame's typed ``meta`` map [eV]; charge-capable
    evaluators (AM1) write per-atom charges [e] back into the same
    ``molpy.Frame``. The evaluator is borrowed — inspect
    ``evaluator.last_result()`` afterwards for solver-internal quantities.

    A method's public ``run(frame)`` entry owns its standalone solve (Gaussian
    SCF, PW and DFTB2). Evaluator components use the C++ ``Evaluate`` preset.
    AM1Evaluator retains its MPI single-point task; generic components reject
    launcher mode when no task payload can represent them.

    Parameters
    ----------
    frame : molpy.Frame
        System state, updated in place in engine units (eV and Å).
    evaluator : method or evaluator component
        A method with ``run(frame)``, or a native evaluator Component.
        Forces follow that object's run policy; use ``forces(frame)`` when
        the method's standalone run is energy-only.
    device : str, optional
        Execution backend, PyTorch-style. v1 accepts only ``"cpu"``; any
        other value raises ``ValueError`` before any compute.

    Raises
    ------
    ValueError
        If *device* is anything but ``"cpu"``, or (SCF path) if the method's
        spin multiplicity cannot split this frame's electron count.
    TypeError
        If *evaluator* has neither a public run entry nor a native Component.
    RuntimeError
        If the frame is missing a required column or carries a wrong dtype.
    """
    if device != "cpu":
        raise ValueError(
            f"device={device!r}: CUDA bindings are not yet available; "
            "only device='cpu' is supported"
        )
    require_cpp()
    # Methods own their standalone Frame entry. This also covers PW and
    # DFTB2 without a second per-family list of supported methods.
    run = getattr(evaluator, "run", None)
    if callable(run):
        run(frame)
        return
    # Preserve AM1's existing MPI task transport. Generic components have
    # no serialized task payload and the native entry rejects launcher use.
    if has_domain("am1") and isinstance(evaluator, _cpp.AM1Evaluator):
        _cpp.single_point(frame, evaluator)
    elif isinstance(evaluator, _cpp._Component):
        _cpp._single_point_component(frame, evaluator)
    else:
        raise TypeError(
            f"single_point cannot run {type(evaluator).__name__}: "
            "expected a method with run(frame) or an evaluator component"
        )


__all__ = ["single_point"]
