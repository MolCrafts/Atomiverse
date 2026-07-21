"""I/O recorders (mirrors ``cpu/recorder/``).

Canonical deep path: ``atomiverse.recorder.Thermo``; top-level
``atomiverse.Thermo`` is the same class.
"""

from atomiverse._binding import _cpp

#: Thermodynamic output at ``end_of_step`` — ``cpu::Thermo(freq [steps])``.
Thermo = _cpp.Thermo

__all__ = ["Thermo"]
