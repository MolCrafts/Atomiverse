/// @file src/core/log.cpp
/// @brief Owner of the logging registry: sinks, rank gating, named subsystem
/// loggers, the structured-report logger, and `log_init` configuration.
///
/// The public macro/accessor surface lives in `core/log.h`; stateful logic is
/// here so there is a single owner of the sink set and category map. Loggers
/// are created once and mutated in place on re-init, so cached `logger*`
/// (the call-site statics behind ATV_*_CAT) stay valid across `log_init`.

#include "atomiverse/core/log.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ATV {
namespace {

constexpr const char* kNormalPattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v";
constexpr const char* kReportPattern = "%v"; // raw: clean tables / CSV rows

/// Single owner of all logging state. File-local singleton.
struct Registry {
    std::mutex mtx;
    bool initialized = false;
    LogConfig cfg;

    std::vector<spdlog::sink_ptr> normal_sinks;  // shared by default + category
    std::vector<spdlog::sink_ptr> report_sinks;  // console only (raw pattern)
    std::vector<spdlog::sink_ptr> metrics_sinks; // metrics file only (raw)

    std::shared_ptr<spdlog::logger> default_logger;
    std::shared_ptr<spdlog::logger> report_logger;
    std::shared_ptr<spdlog::logger> metrics_logger;
    std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> categories;

    static Registry& get() {
        static Registry r;
        return r;
    }

    /// Level for a category, honoring per-category overrides.
    spdlog::level::level_enum level_for(const std::string& name) const {
        for (const auto& [cat, lvl] : cfg.categories)
            if (cat == name)
                return lvl;
        return cfg.level;
    }

    /// Non-rank-0 loggers with no per-rank file sink are turned fully off so
    /// logging is a single boolean check (zero formatting cost).
    bool silent_rank() const {
        return cfg.mpi_rank != 0 && cfg.per_rank_dir.empty()
               && !(cfg.console && !cfg.console_rank0_only);
    }

    enum class Kind { Normal, Report, Metrics };

    void apply_to(spdlog::logger& lg, const std::string& name, Kind kind) {
        switch (kind) {
        case Kind::Normal:
            lg.sinks() = normal_sinks;
            lg.set_pattern(kNormalPattern);
            lg.set_level(silent_rank() ? spdlog::level::off : level_for(name));
            break;
        case Kind::Report:
            lg.sinks() = report_sinks;
            lg.set_pattern(kReportPattern);
            lg.set_level(cfg.mpi_rank == 0 ? spdlog::level::info
                                           : spdlog::level::off);
            break;
        case Kind::Metrics:
            lg.sinks() = metrics_sinks;
            lg.set_pattern(kReportPattern);
            lg.set_level((cfg.mpi_rank == 0 && !metrics_sinks.empty())
                             ? spdlog::level::info
                             : spdlog::level::off);
            break;
        }
        lg.flush_on(spdlog::level::info);
    }

    void rebuild_sinks() {
        normal_sinks.clear();
        report_sinks.clear();
        metrics_sinks.clear();

        const bool console_here =
            cfg.console && (cfg.mpi_rank == 0 || !cfg.console_rank0_only);
        if (console_here)
            normal_sinks.push_back(
                std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        if (!cfg.logfile.empty() && cfg.mpi_rank == 0)
            normal_sinks.push_back(
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                    cfg.logfile, /*truncate=*/true));
        if (!cfg.per_rank_dir.empty()) {
            const std::string path = cfg.per_rank_dir + "/atv.rank"
                                     + std::to_string(cfg.mpi_rank) + ".log";
            normal_sinks.push_back(
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                    path, /*truncate=*/true));
        }

        // Human tables: rank-0 console only. Machine CSV: rank-0 file only.
        if (cfg.mpi_rank == 0) {
            report_sinks.push_back(
                std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            if (!cfg.metrics_file.empty())
                metrics_sinks.push_back(
                    std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                        cfg.metrics_file, /*truncate=*/true));
        }
    }

    void ensure_lazy_default() {
        // Called when a logger is requested before log_init: rank-0 console,
        // info -- matches the historical "safe to skip init" behavior.
        if (initialized || default_logger)
            return;
        rebuild_sinks();
        default_logger = std::make_shared<spdlog::logger>(
            "atv", normal_sinks.begin(), normal_sinks.end());
        apply_to(*default_logger, "atv", Kind::Normal);
        report_logger = std::make_shared<spdlog::logger>(
            "atv.report", report_sinks.begin(), report_sinks.end());
        apply_to(*report_logger, "atv.report", Kind::Report);
        metrics_logger = std::make_shared<spdlog::logger>(
            "atv.metrics", metrics_sinks.begin(), metrics_sinks.end());
        apply_to(*metrics_logger, "atv.metrics", Kind::Metrics);
    }
};

spdlog::level::level_enum parse_level(const std::string& s,
                                      spdlog::level::level_enum fallback) {
    auto lv = spdlog::level::from_str(s);
    // from_str returns `off` for unknown strings; only accept real matches.
    if (lv == spdlog::level::off && s != "off")
        return fallback;
    return lv;
}

void apply_env(LogConfig& cfg) {
    if (const char* v = std::getenv("ATV_LOG_LEVEL"))
        cfg.level = parse_level(v, cfg.level);
    if (const char* v = std::getenv("ATV_LOG_FILE"))
        cfg.logfile = v;
    if (const char* v = std::getenv("ATV_LOG_PER_RANK_DIR"))
        cfg.per_rank_dir = v;
    if (std::getenv("ATV_LOG_CONSOLE_ALL"))
        cfg.console_rank0_only = false;
    if (const char* v = std::getenv("ATV_METRICS_FILE"))
        cfg.metrics_file = v;
    if (const char* v = std::getenv("ATV_LOG_CATEGORIES")) {
        // "comm=trace,scf=warn"
        std::stringstream ss(v);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            auto eq = tok.find('=');
            if (eq == std::string::npos)
                continue;
            cfg.categories.emplace_back(
                tok.substr(0, eq), parse_level(tok.substr(eq + 1), cfg.level));
        }
    }
}

} // namespace

spdlog::logger& logger() {
    Registry& r = Registry::get();
    std::lock_guard<std::mutex> lk(r.mtx);
    r.ensure_lazy_default();
    return *r.default_logger;
}

spdlog::logger& report_logger() {
    Registry& r = Registry::get();
    std::lock_guard<std::mutex> lk(r.mtx);
    r.ensure_lazy_default();
    return *r.report_logger;
}

spdlog::logger& metrics_logger() {
    Registry& r = Registry::get();
    std::lock_guard<std::mutex> lk(r.mtx);
    r.ensure_lazy_default();
    return *r.metrics_logger;
}

spdlog::logger* cat_logger(const char* name) {
    Registry& r = Registry::get();
    std::lock_guard<std::mutex> lk(r.mtx);
    r.ensure_lazy_default();
    auto it = r.categories.find(name);
    if (it != r.categories.end())
        return it->second.get();
    const std::string nm = std::string("atv.") + name;
    auto lg = std::make_shared<spdlog::logger>(nm, r.normal_sinks.begin(),
                                               r.normal_sinks.end());
    r.apply_to(*lg, name, Registry::Kind::Normal);
    auto [ins, ok] = r.categories.emplace(name, std::move(lg));
    return ins->second.get();
}

void log_init(const LogConfig& cfg_in) {
    Registry& r = Registry::get();
    std::lock_guard<std::mutex> lk(r.mtx);
    r.cfg = cfg_in;
    apply_env(r.cfg);
    r.rebuild_sinks();

    // Create or update in place (preserve logger object identity so cached
    // pointers from ATV_*_CAT call sites stay valid).
    if (!r.default_logger)
        r.default_logger = std::make_shared<spdlog::logger>(
            "atv", r.normal_sinks.begin(), r.normal_sinks.end());
    r.apply_to(*r.default_logger, "atv", Registry::Kind::Normal);

    if (!r.report_logger)
        r.report_logger = std::make_shared<spdlog::logger>(
            "atv.report", r.report_sinks.begin(), r.report_sinks.end());
    r.apply_to(*r.report_logger, "atv.report", Registry::Kind::Report);

    if (!r.metrics_logger)
        r.metrics_logger = std::make_shared<spdlog::logger>(
            "atv.metrics", r.metrics_sinks.begin(), r.metrics_sinks.end());
    r.apply_to(*r.metrics_logger, "atv.metrics", Registry::Kind::Metrics);

    for (auto& [name, lg] : r.categories)
        r.apply_to(*lg, name, Registry::Kind::Normal);

    r.initialized = true;
}

bool log_initialized() {
    Registry& r = Registry::get();
    std::lock_guard<std::mutex> lk(r.mtx);
    return r.initialized;
}

} // namespace ATV
