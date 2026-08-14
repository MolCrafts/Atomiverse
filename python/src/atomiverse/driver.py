"""Driver entry points — one-shot evaluations that need no loop of their own.

Canonical deep path: ``atomiverse.driver.single_point``; top-level
``atomiverse.single_point`` is the same function. Its C++ counterpart is
``cpu::single_point`` (``cpu/evaluate/single_point.h``), the helper that
runs one pass of the ``Evaluate`` driver preset.
"""

from atomiverse._binding import _cpp


def single_point(frame: object, evaluator: object, device: str = "cpu") -> None:
    """Compute energy (and any evaluator-specific properties) for one fixed
    geometry — a *single point*, as opposed to a trajectory or an
    optimization.

    Energy lands in the frame's typed ``meta`` map [eV]; charge-capable
    evaluators (AM1) write per-atom charges [e] back into the same
    ``molrs.Frame``. The evaluator is borrowed — inspect
    ``evaluator.last_result()`` afterwards for solver-internal quantities.

    Two dispatch paths, one call shape. A component-style evaluator such as
    ``AM1Evaluator`` is run by the C++ ``cpu::single_point`` helper, which
    drives one pass of the ``Evaluate`` driver over it. A Gaussian SCF method
    (``RHF`` / ``UHF`` / ``RKS`` / ``UKS``) is already a driver in its own
    right — it owns the self-consistent-field loop — so it is dispatched
    straight to ``evaluator.run(frame)``. Nothing is returned either way;
    results are read off the frame or off the evaluator.

    Parameters
    ----------
    frame : molrs.Frame
        System state (``atoms`` block with ``x, y, z, atomic_number``; the
        ``atomic_number`` column [uint] carries Z). Results land as typed
        metadata read back as plain values, for example
        ``frame.meta["energy"]`` is a ``float`` [eV].
    evaluator : atomiverse.AM1Evaluator or atomiverse.RHF/UHF/RKS/UKS
        The evaluator component. ``AM1Evaluator`` runs through the C++
        ``Evaluate`` pass; a Gaussian SCF method owns its own SCF loop, so
        it is driven by ``evaluator.run(frame)`` instead (read the Hartree
        energy / loop state off the method afterwards with ``energy()`` /
        ``converged()`` / ``iterations()``). Neither path computes forces.
    device : str, optional
        Execution backend, PyTorch-style. v1 accepts only ``"cpu"``; any
        other value raises ``ValueError`` before any compute.

    Raises
    ------
    ValueError
        If *device* is anything but ``"cpu"``, or (SCF path) if the method's
        spin multiplicity cannot split this frame's electron count.
    RuntimeError
        If the frame is missing a required column or carries a wrong dtype.
    """
    if device != "cpu":
        raise ValueError(
            f"device={device!r}: CUDA bindings are not yet available; "
            "only device='cpu' is supported"
        )
    # A Gaussian SCF method is its own driver loop, not an Evaluate
    # component. Imported here so this module stays importable on an
    # extension built without ATV_BUILD_HF (then __all__ is empty).
    from atomiverse import scf

    scf_methods = tuple(getattr(scf, name) for name in scf.__all__)
    if scf_methods and isinstance(evaluator, scf_methods):
        evaluator.run(frame)
        return
    _cpp.single_point(frame, evaluator)


__all__ = ["single_point"]
