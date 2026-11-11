/// @file app/atomiverse/python_host.cpp
/// @brief Embedded-CPython host (contract in python_host.h).

#include "python_host.h"

#include <Python.h>

// POSIX getpid/setenv/popen: the launcher process model is Linux-only by charter.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "interpreter_locator.h"

namespace ATV {
namespace app {

PythonHost::PythonHost(std::string script, int argc, char** argv)
    : script_(std::move(script)), argc_(argc), argv_(argv) {
    // The marker MUST land before Py_Initialize: the bindings' LauncherGate
    // (in the extension module's copy of the library) reads the environment
    // lazily at the first run() — the value is OUR pid, so a child process
    // inheriting the environment stays silently serial (PID mismatch).
    const std::string pid = std::to_string(static_cast<long>(::getpid()));
    ::setenv("ATV_LAUNCHER", pid.c_str(), /*overwrite=*/1);
}

namespace {

/// Probe the resolved interpreter's MAJOR.MINOR once (rank 0, before
/// Py_Initialize). One popen; the launcher is Linux-only by charter.
std::string probe_major_minor(const std::string& python) {
    const std::string cmd =
        python + " -c \"import sys;print('%d.%d'%sys.version_info[:2])\"";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr)
        throw std::runtime_error("cannot probe interpreter '" + python + "'");
    char buf[32] = {};
    const bool got = std::fgets(buf, sizeof(buf), pipe) != nullptr;
    ::pclose(pipe);
    if (!got)
        throw std::runtime_error("interpreter '" + python
                                 + "' did not report its version");
    std::string v(buf);
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r'))
        v.pop_back();
    return v;
}

} // namespace

int PythonHost::run() {
    // Runtime interpreter discovery (chain link 05): the environment the
    // launcher runs in — an activated venv/conda env or ATV_PYTHON —
    // decides which Python is embedded. PyConfig's getpath then computes
    // sys.prefix / site-packages from THAT interpreter (pyvenv.cfg
    // honoured). No fallback to build-time paths.
    const InterpreterLocator locator;
    const std::string built = std::to_string(PY_MAJOR_VERSION) + "."
                              + std::to_string(PY_MINOR_VERSION);
    InterpreterLocator::check_abi(probe_major_minor(locator.executable()),
                                  built);

    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    PyStatus status = PyConfig_SetBytesString(&config, &config.executable,
                                              locator.executable().c_str());
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        throw std::runtime_error("PyConfig executable setup "
                                 "failed");
    }
    status =
        PyConfig_SetBytesString(&config, &config.run_filename, script_.c_str());
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        throw std::runtime_error("PyConfig script setup failed");
    }
    // The script's own argv (sys.argv = [script, args...]); parse_argv
    // stays off — the launcher consumed its argv already.
    config.parse_argv = 0;
    status = PyConfig_SetBytesArgv(&config, argc_, argv_);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        throw std::runtime_error("PyConfig argv setup failed");
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        if (PyStatus_IsExit(status))
            return status.exitcode;
        throw std::runtime_error(
            std::string("Python initialization failed: ")
            + (status.err_msg != nullptr ? status.err_msg : "unknown"));
    }

    // Runs the script as __main__, prints any traceback, finalizes the
    // interpreter, and returns the process exit code. Driver destructors
    // inside the script run during interpreter teardown here — before the
    // caller broadcasts SHUTDOWN and finalizes MPI (the Backend::shutdown-
    // before-MPI_Finalize ordering holds by construction).
    return Py_RunMain();
}

} // namespace app
} // namespace ATV
