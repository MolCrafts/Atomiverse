/// @file core/log.h
/// @brief Project-wide logging via spdlog.
///
/// CUDA-free header -- do NOT include from .cuh or .cu files.
/// Components in .cu files that need logging should delegate to .cpp helpers.
///
/// Usage:
/// @code
///   #include "atomiverse/core/log.h"
///   ATV_INFO("Simulation started: {} atoms, {} steps", n_atoms, n_steps);
///   ATV_DEBUG("Component '{}' built", name);
///   ATV_INFO_CAT("comm", "halo rebuilt: {} ghosts", n);   // subsystem logger
///   ATV_MD_INFO("stage {} done", id);                      // sugar for "md"
/// @endcode
///
/// Optional initialization (call once at program start). The Backend also
/// auto-initializes rank-aware logging from the MPI rank/size it detects, so
/// applications rarely need to call this directly:
/// @code
///   ATV::log_init({.level = spdlog::level::info,
///                  .mpi_rank = r, .mpi_size = n,
///                  .logfile = "run.log"});
/// @endcode
///
/// Rank gating: by default only rank 0 emits to the console; all ranks can
/// write per-rank files via `per_rank_dir`. Structured timing/metrics output
/// goes through `report_logger()` (raw pattern) -- see core/report.h.
#pragma once

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif

#include <chrono>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ATV {

using LogClock = std::chrono::steady_clock;

/// @brief Elapsed wall time in milliseconds.
inline double elapsed_ms(LogClock::time_point start,
                         LogClock::time_point end = LogClock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/// @brief Full logging configuration. Env vars (read in `log_init`) override
/// these fields: ATV_LOG_LEVEL, ATV_LOG_FILE, ATV_LOG_PER_RANK_DIR,
/// ATV_LOG_CONSOLE_ALL, ATV_METRICS_FILE, ATV_LOG_CATEGORIES.
struct LogConfig {
    spdlog::level::level_enum level = spdlog::level::info; ///< default/global
    int mpi_rank = 0;
    int mpi_size = 1;
    bool console = true;            ///< colored console sink
    bool console_rank0_only = true; ///< gate console to rank 0
    std::string logfile;            ///< "" = none; rank-0 file sink
    std::string per_rank_dir; ///< "" = none; <dir>/atv.rank<N>.log (all ranks)
    std::string metrics_file; ///< "" = none; CSV/JSON sink on report logger
    /// Per-category level overrides, e.g. {{"comm", trace}, {"scf", warn}}.
    std::vector<std::pair<std::string, spdlog::level::level_enum>> categories;
};

/// @brief The default ("atv") logger. Created lazily with rank-0 console/info
/// defaults if `log_init` has not run.
spdlog::logger& logger();

/// @brief A named subsystem logger (e.g. "md", "scf", "comm", "timer"),
/// sharing the configured sinks. Lazily created and cached; the returned
/// pointer is stable for the process lifetime.
spdlog::logger* cat_logger(const char* name);

/// @brief The structured-report logger: raw `%v` pattern, **rank-0 console
/// only**. Used by core/report.h to print aligned human-readable tables.
/// @see report.h
spdlog::logger& report_logger();

/// @brief The metrics logger: raw `%v` pattern, writes **only** to the
/// configured `metrics_file` (rank 0); a no-op sink set when unset. Used by
/// core/report.h `Csv` to emit machine-readable rows without polluting the
/// console. @see report.h
spdlog::logger& metrics_logger();

/// @brief Full initialization. Env vars override `cfg` fields. Idempotent;
/// an explicit call wins over the Backend's auto-init.
void log_init(const LogConfig& cfg);

/// @brief True once any `log_init` overload has run. The Backend uses this to
/// avoid clobbering an explicit application-level `log_init`.
bool log_initialized();

} // namespace ATV

// -- Convenience macros (compile to nothing when spdlog strips the level) ----

#define ATV_TRACE(...) SPDLOG_LOGGER_TRACE(&ATV::logger(), __VA_ARGS__)
#define ATV_DEBUG(...) SPDLOG_LOGGER_DEBUG(&ATV::logger(), __VA_ARGS__)
#define ATV_INFO(...) SPDLOG_LOGGER_INFO(&ATV::logger(), __VA_ARGS__)
#define ATV_WARN(...) SPDLOG_LOGGER_WARN(&ATV::logger(), __VA_ARGS__)
#define ATV_ERROR(...) SPDLOG_LOGGER_ERROR(&ATV::logger(), __VA_ARGS__)
#define ATV_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(&ATV::logger(), __VA_ARGS__)

// -- Category (named subsystem) macros ---------------------------------------
// The subsystem logger pointer is resolved once per call site (function-local
// static) so repeated logging never re-looks-up the registry. spdlog's
// compile-time level elision (SPDLOG_ACTIVE_LEVEL) still applies.
#define ATV_LOG_CAT_IMPL_(SPDLOG_MACRO, cat, ...)                              \
    do {                                                                       \
        static ::spdlog::logger* _atv_cat_lg = ::ATV::cat_logger(cat);         \
        SPDLOG_MACRO(_atv_cat_lg, __VA_ARGS__);                                \
    } while (0)

#define ATV_TRACE_CAT(cat, ...)                                                \
    ATV_LOG_CAT_IMPL_(SPDLOG_LOGGER_TRACE, cat, __VA_ARGS__)
#define ATV_DEBUG_CAT(cat, ...)                                                \
    ATV_LOG_CAT_IMPL_(SPDLOG_LOGGER_DEBUG, cat, __VA_ARGS__)
#define ATV_INFO_CAT(cat, ...)                                                 \
    ATV_LOG_CAT_IMPL_(SPDLOG_LOGGER_INFO, cat, __VA_ARGS__)
#define ATV_WARN_CAT(cat, ...)                                                 \
    ATV_LOG_CAT_IMPL_(SPDLOG_LOGGER_WARN, cat, __VA_ARGS__)
#define ATV_ERROR_CAT(cat, ...)                                                \
    ATV_LOG_CAT_IMPL_(SPDLOG_LOGGER_ERROR, cat, __VA_ARGS__)

// Per-subsystem sugar (optional; the _CAT form is the primitive).
#define ATV_MD_INFO(...) ATV_INFO_CAT("md", __VA_ARGS__)
#define ATV_MD_DEBUG(...) ATV_DEBUG_CAT("md", __VA_ARGS__)
#define ATV_SCF_INFO(...) ATV_INFO_CAT("scf", __VA_ARGS__)
#define ATV_SCF_DEBUG(...) ATV_DEBUG_CAT("scf", __VA_ARGS__)
#define ATV_COMM_INFO(...) ATV_INFO_CAT("comm", __VA_ARGS__)
#define ATV_COMM_DEBUG(...) ATV_DEBUG_CAT("comm", __VA_ARGS__)
