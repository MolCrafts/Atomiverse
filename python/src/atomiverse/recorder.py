"""I/O recorders (mirrors ``cpu/record/``).

Canonical deep paths: ``atomiverse.recorder.Thermo`` and
``atomiverse.recorder.XYZ``; the top-level ``atomiverse.Thermo`` /
``atomiverse.XYZ`` are the same classes.

Prefer ``MD.set_thermo(N)`` for the common LAMMPS-style cadence; construct
``Thermo(freq)`` only when passing through ``set_record`` or the phase
escape hatch. ``XYZ(filename, freq)`` always goes through ``set_record``.

Both names are unconditional: the wheel forces ``ATV_BUILD_IO=ON`` through
the CMake module DAG, so there is no capability probe here (unlike the
compile-optional ``KSpacePME`` / ``PW``).
"""

from atomiverse._binding import _cpp

#: Thermodynamic output at ``end_of_step`` — ``cpu::Thermo(freq [steps])``.
Thermo = _cpp.Thermo

#: XYZ trajectory output at ``end_of_step`` —
#: ``cpu::XyzRecorder(filename, freq [steps])``.
XYZ = _cpp.XYZ

__all__ = ["Thermo", "XYZ"]
