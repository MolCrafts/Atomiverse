/// @file regressions/harness/main.cpp
/// @brief `atv_regressions --case <domain/method> [--section <name>] [args…]`
///        | `--list`.
///
/// A case runs its sections in name order and fails when any section fails.
/// Sections listed in `ATV_REG_BOTH_SECTIONS` (':'-separated) get
/// `--device=both`. A section that relaunches its own executable (mpiexec
/// children) passes only its own arguments; case and section travel in the
/// environment (`ATV_REG_CASE`, `ATV_REG_SECTION`).
#include "registry.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace atv_reg {

std::vector<Section>& registry() {
    static std::vector<Section> sections;
    return sections;
}

Registrar::Registrar(const char* case_name, const char* section, Entry entry) {
    registry().push_back({case_name, section, entry, nullptr});
}

Registrar::Registrar(const char* case_name, const char* section,
                     PlainEntry entry) {
    registry().push_back({case_name, section, nullptr, entry});
}

} // namespace atv_reg

namespace {

std::vector<std::string> split(const char* text, char sep) {
    std::vector<std::string> out;
    if (!text)
        return out;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, sep);)
        if (!item.empty())
            out.push_back(item);
    return out;
}

int run_section(const atv_reg::Section& s, char* argv0,
                const std::vector<char*>& extra) {
    setenv("ATV_REG_SECTION", s.name.c_str(), 1);
    if (s.plain)
        return s.plain();
    std::vector<std::string> owned;
    const auto both = split(std::getenv("ATV_REG_BOTH_SECTIONS"), ':');
    if (std::find(both.begin(), both.end(), s.name) != both.end())
        owned.push_back("--device=both");
    std::vector<char*> args{argv0};
    for (auto& a : owned)
        args.push_back(a.data());
    for (char* a : extra)
        args.push_back(a);
    args.push_back(nullptr);
    return s.entry(static_cast<int>(args.size()) - 1, args.data());
}

} // namespace

int main(int argc, char** argv) {
    const auto& all = atv_reg::registry();
    if (argc == 2 && std::strcmp(argv[1], "--list") == 0) {
        for (const auto& s : all)
            std::printf("%s %s\n", s.case_name.c_str(), s.name.c_str());
        return 0;
    }
    std::string case_name, section;
    std::vector<char*> extra;
    bool child = false;
    if (argc >= 3 && std::strcmp(argv[1], "--case") == 0) {
        case_name = argv[2];
        int i = 3;
        if (argc >= 5 && std::strcmp(argv[3], "--section") == 0) {
            section = argv[4];
            i = 5;
        }
        for (; i < argc; ++i)
            extra.push_back(argv[i]);
        setenv("ATV_REG_CASE", case_name.c_str(), 1);
    } else if (const char* inherited = std::getenv("ATV_REG_CASE")) {
        child = true;
        case_name = inherited;
        if (const char* s = std::getenv("ATV_REG_SECTION"))
            section = s;
        for (int i = 1; i < argc; ++i)
            extra.push_back(argv[i]);
    } else {
        std::fprintf(stderr,
                     "usage: %s --case <domain/method> [--section <name>] "
                     "[args...] | --list\n",
                     argv[0]);
        return 2;
    }

    std::vector<const atv_reg::Section*> chosen;
    for (const auto& s : all)
        if (s.case_name == case_name && (section.empty() || s.name == section))
            chosen.push_back(&s);
    if (chosen.empty()) {
        std::fprintf(stderr, "unknown regression '%s%s%s' (see --list)\n",
                     case_name.c_str(), section.empty() ? "" : " ",
                     section.c_str());
        return 2;
    }
    if (child)
        return run_section(*chosen.front(), argv[0], extra);
    std::sort(chosen.begin(), chosen.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });

    struct Outcome {
        std::string name;
        int rc;
        double seconds;
    };
    std::vector<Outcome> outcomes;
    for (const auto* s : chosen) {
        std::printf("\n=== %s :: %s ===\n", case_name.c_str(), s->name.c_str());
        std::fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();
        const int rc = run_section(*s, argv[0], extra);
        const std::chrono::duration<double> dt =
            std::chrono::steady_clock::now() - t0;
        outcomes.push_back({s->name, rc, dt.count()});
    }
    int failed = 0;
    std::printf("\n=== %s summary ===\n", case_name.c_str());
    for (const auto& o : outcomes) {
        std::printf("  %-4s %7.1f s  %s\n", o.rc == 0 ? "PASS" : "FAIL",
                    o.seconds, o.name.c_str());
        failed += o.rc != 0;
    }
    std::printf("%d/%zu sections failed\n", failed, outcomes.size());
    return failed == 0 ? 0 : 1;
}
