"""CPU geometry minimization with typed phases and explicit loop policies."""

from __future__ import annotations

from atomiverse._binding import _cpp

_PHASE_MAP: dict[str, object] = {
    "search_direction": _cpp.MinimizePhase.search_direction,
    "line_search": _cpp.MinimizePhase.line_search,
    "force": _cpp.MinimizePhase.force,
    "convergence_check": _cpp.MinimizePhase.convergence_check,
    "end_of_step": _cpp.MinimizePhase.end_of_step,
}

_DRIVER_DISPATCH: dict[type, str] = {
    _cpp.LBFGS: "add_lbfgs",
}

_PHASE_DISPATCH: dict[type, str] = {
    _cpp.PairLJCut: "add_pair_lj_cut",
    _cpp.NeighborList: "add_neighbor_list",
}


class PhaseBuilder:
    """Fluent builder returned by :meth:`Minimize.on`."""

    def __init__(self, cpp_builder: object) -> None:
        self._cpp = cpp_builder

    def add(self, component: object, order: int = 0) -> PhaseBuilder:
        method_name = _PHASE_DISPATCH.get(type(component))
        if method_name is None:
            supported = ", ".join(t.__name__ for t in _PHASE_DISPATCH)
            raise TypeError(
                f"Cannot add {type(component).__name__} to a minimizer phase. "
                f"Supported: {supported}"
            )
        getattr(self._cpp, method_name)(component, order)
        return self


class Minimize:
    """Geometry-minimization driver for the CPU backend.

    The optimizer is a component and convergence is an independent handler.
    ``run`` updates the borrowed ``molrs.Frame`` in place and returns only
    loop-control metadata.
    """

    def __init__(
        self,
        convergence: object | None = None,
        device: str = "cpu",
    ) -> None:
        if device != "cpu":
            raise ValueError(
                f"device={device!r}: GPU minimization is deferred; "
                "only device='cpu' is supported"
            )
        if convergence is None:
            convergence = _cpp.ForceConverged()
        if type(convergence) is not _cpp.ForceConverged:
            raise TypeError("convergence must be a ForceConverged policy")
        self.device = device
        self._cpp = _cpp.Minimize(convergence)
        self._built = False

    def add(self, component: object) -> Minimize:
        """Add an optimizer component with its default typed bindings."""
        method_name = _DRIVER_DISPATCH.get(type(component))
        if method_name is None:
            supported = ", ".join(t.__name__ for t in _DRIVER_DISPATCH)
            raise TypeError(
                f"Cannot add {type(component).__name__} directly. "
                f"Supported optimizer components: {supported}"
            )
        getattr(self._cpp, method_name)(component)
        self._built = False
        return self

    def on(self, phase: str) -> PhaseBuilder:
        """Return a builder for one minimizer-specific phase."""
        cpp_phase = _PHASE_MAP.get(phase)
        if cpp_phase is None:
            valid = ", ".join(f'"{p}"' for p in _PHASE_MAP)
            raise ValueError(f"Unknown minimizer phase {phase!r}. Valid: {valid}")
        self._built = False
        return PhaseBuilder(self._cpp.on(cpp_phase))

    def build(self) -> None:
        """Compile the execution plan (called automatically by ``run``)."""
        self._cpp.build()
        self._built = True

    def run(self, frame: object, max_iter: int) -> object:
        """Optimize ``frame`` for at most ``max_iter`` iterations."""
        if not self._built:
            self.build()
        return self._cpp.run(frame, max_iter)

    def __repr__(self) -> str:
        state = "built" if self._built else "not built"
        return f"Minimize({state})"


LBFGS = _cpp.LBFGS
ForceConverged = _cpp.ForceConverged
MinimizeResult = _cpp.MinimizeResult

__all__ = ["Minimize", "LBFGS", "ForceConverged", "MinimizeResult"]
