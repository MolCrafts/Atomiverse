/// @file app/atomiverse/python_host.h
/// @brief PythonHost -- rank 0's embedded CPython interpreter.
///
/// Owns the one interpreter of the launcher process model: publishes the
/// PID-qualified `ATV_LAUNCHER` marker BEFORE `Py_Initialize` (the bindings'
/// `LauncherGate` reads it at the dispatch seam), then runs the user script
/// as `__main__` via `Py_RunMain`. Only the launcher executable links
/// libpython; libatomiverse and `_atomiverse.so` are untouched.
///
/// Interpreter discovery (04 contract): the BUILD-TIME Python — CMake's
/// `Python3_EXECUTABLE` is compiled in as the interpreter identity, so
/// `PyConfig`'s path computation resolves that environment's stdlib and
/// site-packages (a venv's `pyvenv.cfg` included). Runtime discovery /
/// relocation is chain link 05.
#pragma once

#include <string>

namespace ATV {
namespace app {

/// @brief Embeds CPython and runs one script as `__main__`.
class PythonHost {
  public:
    /// @brief Validate nothing further; publish the launcher marker.
    /// @param script Path to the Python script (validated by LauncherArgs).
    /// @param argc Count of the script's argv (script itself included).
    /// @param argv The script's argv — `argv[0]` is the script path, the
    ///             rest are its own arguments (becomes `sys.argv`).
    PythonHost(std::string script, int argc, char** argv);

    /// @brief Initialize the interpreter, run the script, finalize.
    /// @return The script's process exit code (nonzero on Python
    ///         exception; CPython prints the traceback itself).
    int run();

  private:
    std::string script_;
    int argc_;
    char** argv_;
};

} // namespace app
} // namespace ATV
