"""Driver entry points — one-shot evaluations that need no loop of their own.

Canonical deep path: ``atomiverse.driver.single_point``; top-level
``atomiverse.single_point`` is the same function. Its C++ counterpart is
``cpu::single_point`` (``cpu/evaluate/single_point.h``), the helper that
runs one pass of the ``Evaluate`` driver preset.
"""

from atomiverse._binding import _cpp, has_domain, require_cpp, backend_type


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
    AM1 keeps its MPI single-point task; generic components reject launcher
    mode when no task payload can represent them.

    Parameters
    ----------
    frame : molpy.Frame
        System state, updated in place in engine units (eV and Å).
    evaluator : method or evaluator component
        A method with ``run(frame)``, or a native evaluator Component.
        Forces follow that object's run policy; use ``forces(frame)`` when
        the method's standalone run is energy-only.
    device : str, optional
        Execution backend, ``"cpu"`` or ``"cuda"``. The method must use
        the same backend; unavailable bindings fail before compute.

    Raises
    ------
    ValueError
        If the requested backend is unavailable or mismatched, if the method's
        ``n_alpha + n_beta`` does not equal this frame's electron count, or if
        an NDDO method (AM1/RM1/MNDO/PM3) finds atoms closer than 0.6 x their
        covalent-radius sum (the message names the pairs).
    TypeError
        If *evaluator* has neither a public run entry nor a native Component.
    RuntimeError
        If the frame is missing a required column or carries a wrong dtype.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown device {device!r}; expected 'cpu' or 'cuda'")
    require_cpp()
    method_device = getattr(evaluator, "device", "cpu")
    if method_device != device:
        raise ValueError(f"Method device={method_device!r} does not match requested device={device!r}")
    if device == "cuda":
        evaluation = Evaluate(device="cuda")
        evaluation.set_evaluator(evaluator)
        evaluation.run(frame)
        return
    # Preserve AM1's existing single-point/MPI task behavior, including forces.
    if has_domain("am1") and isinstance(evaluator, _cpp.AM1):
        _cpp.single_point(frame, evaluator)
        return
    if has_domain("am1") and isinstance(evaluator, (_cpp.RM1, _cpp.MNDO, _cpp.PM3)):
        Evaluate(device="cpu").set_evaluator(evaluator).run(frame)
        return
    # Methods own their standalone Frame entry. This also covers PW and
    # DFTB2 without a second per-family list of supported methods.
    run = getattr(evaluator, "run", None)
    if callable(run):
        run(frame)
        return
    if isinstance(evaluator, _cpp._Component):
        _cpp._single_point_component(frame, evaluator)
    else:
        raise TypeError(
            f"single_point cannot run {type(evaluator).__name__}: "
            "expected a method with run(frame) or an evaluator component"
        )


class Evaluate:
    """One native energy/force evaluation with borrowed, kept-alive methods."""
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._cpp = backend_type(device, "_Evaluate", "_CudaEvaluate")()
        self._built = False

    def set_evaluator(self, method: object):
        # A classical parameter mirror has no device. A borrowed electronic
        # method does, and it has to match this driver.
        method_device = getattr(method, "device", None)
        if method_device is not None and method_device != self.device:
            raise ValueError("Evaluator and driver must use the same device")
        self._cpp.set_evaluator(method)
        self._built = False
        return self

    def set_neighbor(self, component: object):
        """Install the neighbor list. Coulomb and Lennard-Jones read it."""
        self._cpp.set_neighbor(component)
        self._built = False
        return self

    def build(self) -> None:
        self._cpp.build()
        self._built = True

    def run(self, frame: object) -> None:
        if not self._built:
            self.build()
        self._cpp.run(frame)


__all__ = ["single_point", "Evaluate"]
