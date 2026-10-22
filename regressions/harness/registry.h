/// @file regressions/harness/registry.h
/// @brief Case registry of the single `atv_regressions` executable.
///
/// A case is one method (`regressions/<domain>/<method>/`). Each section
/// source defines `run()` (or `run(argc, argv)`) inside its own namespace and
/// registers it with `ATV_REGRESSION("<domain>/<method>", "<section>", ns::run)`.
/// ctest runs one case per test: `atv_regressions --case <domain>/<method>`.
#pragma once

#include <string>
#include <vector>

namespace atv_reg {

/// @brief Entry point of a section that reads its forwarded argv.
using Entry = int (*)(int argc, char** argv);
/// @brief Entry point of a section without arguments.
using PlainEntry = int (*)();

/// @brief One registered section; exactly one entry is set.
struct Section {
    std::string case_name;      ///< "domain/method".
    std::string name;           ///< Section name.
    Entry entry = nullptr;      ///< Body taking argv.
    PlainEntry plain = nullptr; ///< Body without arguments.
};

/// @brief Directory of a section source, for inputs that sit next to it.
/// @param file `__FILE__` of the section source.
/// @return The directory part of `file`.
inline std::string case_dir(const char* file) {
    const std::string path = file;
    return path.substr(0, path.find_last_of('/'));
}

/// @brief All sections linked into the executable, in registration order.
std::vector<Section>& registry();

/// @brief Static registration helper used by `ATV_REGRESSION`.
struct Registrar {
    /// @brief Register a section taking the forwarded argv.
    /// @param case_name "domain/method".
    /// @param section Section name.
    /// @param entry Section body.
    Registrar(const char* case_name, const char* section, Entry entry);
    /// @brief Register a section without arguments.
    /// @param case_name "domain/method".
    /// @param section Section name.
    /// @param entry Section body.
    Registrar(const char* case_name, const char* section, PlainEntry entry);
};

} // namespace atv_reg

#define ATV_REG_CAT2_(a, b) a##b
#define ATV_REG_CAT_(a, b) ATV_REG_CAT2_(a, b)
/// @brief Register `fn` as section `section` of case `case_name`.
#define ATV_REGRESSION(case_name, section, fn)                                 \
    static const ::atv_reg::Registrar ATV_REG_CAT_(atv_reg_registrar_,         \
                                                   __COUNTER__) {              \
        case_name, section, fn                                                 \
    }
