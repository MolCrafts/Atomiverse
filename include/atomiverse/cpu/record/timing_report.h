/// @file cpu/record/timing_report.h
/// @brief Cross-rank reduction + rendering of TimerRegistry data into a
/// LAMMPS-`timing breakdown`-comparable table (console) and CSV (metrics file).
///
/// Lives in `cpu/md` because it knows the 8 LAMMPS-aligned MD timing
/// categories and their names (`StepTiming`). The registry/data model is
/// domain-agnostic (`core/timer.h`); output is routed through the log module
/// (`core/report.h`).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "atomiverse/core/report.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/driver/step_timing.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Per-section cross-rank stats (mirrors LAMMPS min/avg/max + %var).
struct SectionStat {
    double min_s = 0.0;
    double avg_s = 0.0;
    double max_s = 0.0;
    double pct_var = 0.0;   ///< 100*(max-avg)/avg -- load imbalance
    double pct_total = 0.0; ///< 100*avg/loop_avg
};

/// @brief Reduced timing report for one stage/run.
struct ReducedReport {
    std::array<SectionStat, StepTiming::kNumCategories> sections{};
    double loop_min = 0.0, loop_avg = 0.0, loop_max = 0.0;
    int nlocal_min = 0, nlocal_max = 0;
    double nlocal_avg = 0.0;
    int nghost_min = 0, nghost_max = 0;
    double nghost_avg = 0.0;
    std::uint64_t steps = 0;
    int ranks = 1;
};

/// @brief Reduce a registry's per-category totals + loop wall + nlocal across
/// ranks (MPI_COMM_WORLD when MPI is initialized; single-rank otherwise) into
/// a `ReducedReport`. Uses the batched min/max/sum allreduce idiom.
ReducedReport reduce_timing(const TimerRegistry& reg);

/// @brief Emit the human-readable per-section table (console, rank 0) through
/// the report logger.
void emit_timing_table(const ReducedReport& rep, std::string_view stage);

/// @brief Emit one CSV row (schema identical to StepTiming::csv_header()) to
/// the given Csv sink (metrics file, rank 0).
void emit_timing_csv(const ReducedReport& rep, report::Csv& csv,
                     std::string_view stage);

// ── Memory breakdown (built-in driver summary, peer of the timing table) ─────

/// @brief Process peak resident set size [kB] (Linux `/proc/self/status`
/// VmHWM). 0 if unavailable. The true high-water mark, not the current RSS.
std::size_t read_peak_rss_kb();

/// @brief Cross-rank-reduced memory breakdown for one run/stage. Categories
/// mirror the timing taxonomy plus an `Atoms` bucket for per-atom storage;
/// each `cat_bytes` entry is SUMMED across ranks (total footprint), while the
/// peak-RSS stats are per-rank min/avg/max + sum.
struct MemReport {
    static constexpr int kNumCat =
        6; ///< Atoms, Pair, Bonded, Neigh, Comm, Other
    static constexpr std::array<std::string_view, kNumCat> kNames{
        "Atoms", "Pair", "Bonded", "Neigh", "Comm", "Other"};
    std::array<std::size_t, kNumCat>
        cat_bytes{};                 ///< per-category, summed over ranks
    std::size_t accounted_bytes = 0; ///< Σ cat_bytes (AV-attributed resident)
    double rss_sum_mb = 0.0;         ///< Σ per-rank peak RSS
    double rss_min_mb = 0.0, rss_avg_mb = 0.0, rss_max_mb = 0.0;
    int ranks = 1;
};

/// @brief Reduce per-rank `(component-name, owned-bytes)` pairs (from
/// `Backend::collect_component_memory`) + this rank's peak RSS into a
/// `MemReport`. Names are classified into categories (`[atoms]`→Atoms,
/// `[domain]`→Comm, else `StepTiming::classify`). MPI-summed when initialized.
MemReport
reduce_memory(const std::vector<std::pair<std::string, std::size_t>>& local,
              std::size_t peak_rss_kb);

/// @brief Emit the human-readable memory table (console, rank 0): per-category
/// total + %RSS, the AV-accounted total, the unaccounted remainder
/// (runtime / MPI / allocator), and per-rank peak-RSS min/avg/max.
void emit_memory_table(const MemReport& rep, std::string_view stage);

/// @brief Diagnostic: emit per-component (per-name) max-over-ranks wall time —
/// splits the category rollup (e.g. Comm) into its constituent components
/// (halo_forward / halo_reverse / owner_exchange). Rank 0 only; intended for
/// scaling investigations. Component registration order is identical across
/// ranks, so the per-component totals reduce elementwise.
void emit_per_component(const TimerRegistry& reg, std::string_view stage);

} // namespace md
} // namespace cpu
} // namespace ATV
