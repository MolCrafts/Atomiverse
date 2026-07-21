"""Driver entry points (mirrors ``cpu/driver/``).

Canonical deep path: ``atomiverse.driver.single_point``; top-level
``atomiverse.single_point`` is the same function.
"""

from atomiverse._binding import _cpp


def single_point(frame: object, evaluator: object, device: str = "cpu") -> None:
    """Evaluate *frame* once with *evaluator* (``cpu::single_point``).

    One ``Evaluate`` pass: energy lands in the frame's typed ``meta`` map [eV];
    charge-capable evaluators (AM1) write per-atom charges [e] back into the
    same ``molrs.Frame``. The evaluator is borrowed — inspect
    ``evaluator.last_result()`` afterwards for solver-internal quantities.

    Parameters
    ----------
    frame : molrs.Frame
        System state (``atoms`` block with ``x, y, z, type``; for
        semiempirical methods ``type`` is the atomic number Z). Results use
        ``molrs.MetaValue`` entries, for example
        ``frame.meta["energy"].value`` with dtype ``"f64"``.
    evaluator : atomiverse.AM1Evaluator
        The evaluator component (v1: AM1 only; SCF/PW single-point
        wrappers are future specs).
    device : str, optional
        Execution backend, PyTorch-style. v1 accepts only ``"cpu"``; any
        other value raises ``ValueError`` before any compute.
    """
    if device != "cpu":
        raise ValueError(
            f"device={device!r}: CUDA bindings are not yet available; "
            "only device='cpu' is supported"
        )
    _cpp.single_point(frame, evaluator)


__all__ = ["single_point"]
