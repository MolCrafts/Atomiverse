/// @file app/atomiverse/launcher_args.h
/// @brief LauncherArgs -- validated command line of the `atomiverse` launcher.
///
/// Header-only validating constructor (never a `parse_X()` free function):
/// construction succeeds iff argv carries exactly one readable script path.
/// Under mpirun every rank receives identical argv, so validation fails (or
/// passes) deterministically on every rank — a usage error exits nonzero
/// everywhere with no hang.
#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

namespace ATV {
namespace app {

/// @brief Validated launcher command line: `atomiverse <script.py>`.
class LauncherArgs {
  public:
    /// @brief Validate argv into a script path.
    /// @param argc Argument count from main.
    /// @param argv Argument vector from main.
    /// @throws std::runtime_error when the script argument is missing or
    ///         the file does not exist / is not readable.
    LauncherArgs(int argc, char** argv) {
        if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0')
            throw std::runtime_error(
                "usage: atomiverse <script.py>  (the Python script to run "
                "on rank 0)");
        script_ = argv[1];
        std::ifstream probe(script_);
        if (!probe.good())
            throw std::runtime_error("atomiverse: cannot read script '"
                                     + script_ + "'");
    }

    /// @brief The validated script path.
    /// @return Path exactly as given on the command line.
    const std::string& script() const noexcept { return script_; }

  private:
    std::string script_;
};

} // namespace app
} // namespace ATV
