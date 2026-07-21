/// @file cpu/task/launcher_gate.h
/// @brief LauncherGate -- "am I hosted by the MPI launcher?" probe
///        (mpi-launcher-embedded-python-04).
///
/// The launcher executable and the `_atomiverse` extension module each link
/// their own copy of the static atomiverse library, so a C++ global set on
/// the executable side is invisible to the bindings side. The only process-
/// wide state that crosses that boundary is the environment and the shared
/// libmpi: the launcher's `PythonHost` therefore publishes the marker
/// `ATV_LAUNCHER=<its own PID>` before `Py_Initialize`, and the bindings'
/// dispatch seam probes it here.
///
/// `active()` ⇔ marker present ∧ marker PID == getpid() ∧ MPI initialized
/// ∧ world size > 1.
///
/// **Never throws; inconsistency degrades to inactive.** A child process
/// spawned from the user script inherits the marker but has a new PID —
/// that shape is SILENTLY inactive (a multiprocessing pool must not print
/// one warning per worker). Marker matching our own PID while MPI is not
/// initialized is a genuinely inconsistent host; still inactive, but with a
/// one-time stderr warning for diagnosability.
#pragma once

namespace ATV {
namespace cpu {
namespace task {

/// @brief Probe for launcher hosting; constructed at the dispatch seam.
class LauncherGate {
  public:
    /// @brief Read the marker + MPI state; never throws.
    LauncherGate() noexcept;

    /// @brief Whether task dispatch through the launcher protocol applies.
    /// @return True only when this process IS the marker's writer AND a
    ///         multi-rank MPI world is up.
    bool active() const noexcept { return active_; }

  private:
    bool active_ = false;
};

} // namespace task
} // namespace cpu
} // namespace ATV
