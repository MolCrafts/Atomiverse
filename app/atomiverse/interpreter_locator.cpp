/// @file app/atomiverse/interpreter_locator.cpp
/// @brief Interpreter resolution (contract in interpreter_locator.h).

#include "interpreter_locator.h"

// POSIX access(X_OK): the launcher process model is Linux-only by charter.
#include <unistd.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ATV {
namespace app {

namespace {

bool is_executable(const std::string& path) {
    return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

/// One left-to-right pass over a ':'-separated PATH for `name`;
/// empty segments are skipped. Returns "" when not found.
std::string search_path(const std::string& path_env, const std::string& name) {
    std::size_t begin = 0;
    while (begin <= path_env.size()) {
        const std::size_t end = path_env.find(':', begin);
        const std::string dir = path_env.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!dir.empty()) {
            const std::string candidate = dir + "/" + name;
            if (is_executable(candidate))
                return candidate;
        }
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return {};
}

} // namespace

InterpreterLocator::InterpreterLocator()
    : InterpreterLocator(std::getenv("ATV_PYTHON"), std::getenv("PATH")) {}

InterpreterLocator::InterpreterLocator(const char* atv_python_override,
                                       const char* path_env) {
    if (atv_python_override != nullptr) {
        const std::string overridden = atv_python_override;
        if (!is_executable(overridden))
            throw std::runtime_error(
                "ATV_PYTHON='" + overridden
                + "' does not name an executable interpreter");
        executable_ = overridden;
        return;
    }

    const std::string path = path_env != nullptr ? path_env : "";
    executable_ = search_path(path, "python3");
    if (executable_.empty())
        executable_ = search_path(path, "python");
    if (executable_.empty())
        throw std::runtime_error(
            "no Python interpreter found (search order: "
            "ATV_PYTHON override, then python3 on PATH, then python on "
            "PATH). Activate the environment the wheel was installed "
            "into, or set ATV_PYTHON to its interpreter.");
}

void InterpreterLocator::check_abi(const std::string& probed_major_minor,
                                   const std::string& built_major_minor) {
    if (probed_major_minor != built_major_minor)
        throw std::runtime_error(
            "interpreter Python " + probed_major_minor
            + " does not match the libpython " + built_major_minor
            + " this launcher was built against (same minor version "
              "required). Point ATV_PYTHON at a Python "
            + built_major_minor + " interpreter.");
}

} // namespace app
} // namespace ATV
