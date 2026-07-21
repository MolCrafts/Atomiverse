/// @file app/atomiverse/interpreter_locator.h
/// @brief InterpreterLocator -- runtime discovery of the Python interpreter
///        the launcher embeds (mpi-launcher-embedded-python-05).
///
/// The launcher binary may be installed into any venv/conda/module
/// environment; the interpreter it embeds must be THAT environment's, not
/// the build machine's. Resolution order (validating constructor, no
/// fallback to build-time paths):
///   1. `ATV_PYTHON` (path to a python executable) -- the explicit override
///      for exotic layouts;
///   2. the first executable `python3` on `PATH` (an activated venv/conda
///      env or an Lmod-loaded Python puts its interpreter first);
///   3. a second pass for `python`;
///   4. nothing found -> `std::runtime_error` naming the search order and
///      the `ATV_PYTHON` hint.
///
/// Deliberately `Python.h`-free and free of process spawning: pure path
/// resolution + string comparison, so it is unit-testable in `tests/`
/// (zero third-party reads). The live-interpreter version probe and the
/// `PyConfig` calls live in `PythonHost`.
#pragma once

#include <string>

namespace ATV {
namespace app {

/// @brief Resolves which Python interpreter the launcher embeds.
class InterpreterLocator {
  public:
    /// @brief Production resolution: reads `ATV_PYTHON` and `PATH` from
    ///        the environment.
    /// @throws std::runtime_error per the resolution contract above.
    InterpreterLocator();

    /// @brief Injection constructor (the test seam).
    /// @param atv_python_override Value of ATV_PYTHON; nullptr = unset.
    /// @param path_env Value of PATH (':'-separated); nullptr = unset.
    /// @throws std::runtime_error per the resolution contract above.
    InterpreterLocator(const char* atv_python_override, const char* path_env);

    /// @brief The resolved interpreter executable.
    /// @return Absolute or PATH-joined path exactly as resolved.
    const std::string& executable() const noexcept { return executable_; }

    /// @brief ABI guard: the launcher links one libpython minor version and
    ///        can only host interpreters of the same minor.
    /// @param probed_major_minor "MAJOR.MINOR" of the resolved interpreter
    ///        (probed live by the caller).
    /// @param built_major_minor "MAJOR.MINOR" this binary was built against.
    /// @throws std::runtime_error naming BOTH versions and the ATV_PYTHON
    ///         hint when they differ.
    static void check_abi(const std::string& probed_major_minor,
                          const std::string& built_major_minor);

  private:
    std::string executable_;
};

} // namespace app
} // namespace ATV
