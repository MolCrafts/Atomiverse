/// @file cpu/recorder/timing_report.cpp
/// @brief Cross-rank reduction + rendering of TimerRegistry data.

#include "atomiverse/cpu/record/timing_report.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

namespace {
constexpr std::size_t kN = StepTiming::kNumCategories; // 8 LAMMPS sections
} // namespace

ReducedReport reduce_timing(const TimerRegistry& reg) {
    ReducedReport rep;
    rep.steps = reg.steps();

    // Local per-category totals (+ loop wall) packed into one buffer.
    const std::vector<TimerStat> rollup = reg.rollup_by_category();
    std::array<double, kN + 1> local{};
    for (std::size_t c = 0; c < kN; ++c)
        local[c] = (c < rollup.size()) ? rollup[c].total_s : 0.0;
    local[kN] = reg.loop_seconds();

    std::array<double, kN + 1> vmin = local, vmax = local, vsum = local;
    int nl = reg.nlocal() < 0 ? 0 : reg.nlocal();
    int nl_min = nl, nl_max = nl;
    long long nl_sum = nl;
    int ng = reg.nghost() < 0 ? 0 : reg.nghost();
    int ng_min = ng, ng_max = ng;
    long long ng_sum = ng;
    int ranks = 1;

#ifdef ATV_HAS_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        if (ranks > 1) {
            MPI_Allreduce(local.data(), vmin.data(), kN + 1, MPI_DOUBLE,
                          MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(local.data(), vmax.data(), kN + 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(local.data(), vsum.data(), kN + 1, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&nl, &nl_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nl, &nl_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            long long nll = nl;
            MPI_Allreduce(&nll, &nl_sum, 1, MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&ng, &ng_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&ng, &ng_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            long long ngl = ng;
            MPI_Allreduce(&ngl, &ng_sum, 1, MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
        }
    }
#endif

    rep.ranks = ranks;
    rep.loop_min = vmin[kN];
    rep.loop_avg = vsum[kN] / ranks;
    rep.loop_max = vmax[kN];
    rep.nlocal_min = nl_min;
    rep.nlocal_max = nl_max;
    rep.nlocal_avg = static_cast<double>(nl_sum) / ranks;
    rep.nghost_min = ng_min;
    rep.nghost_max = ng_max;
    rep.nghost_avg = static_cast<double>(ng_sum) / ranks;

    for (std::size_t c = 0; c < kN; ++c) {
        SectionStat s;
        s.min_s = vmin[c];
        s.avg_s = vsum[c] / ranks;
        s.max_s = vmax[c];
        s.pct_var = s.avg_s > 0.0 ? 100.0 * (s.max_s - s.avg_s) / s.avg_s : 0.0;
        s.pct_total = rep.loop_avg > 0.0 ? 100.0 * s.avg_s / rep.loop_avg : 0.0;
        rep.sections[c] = s;
    }
    return rep;
}

void emit_timing_table(const ReducedReport& rep, std::string_view stage) {
    std::string title = "Timing breakdown [" + std::string(stage)
                        + "]  ranks=" + std::to_string(rep.ranks)
                        + "  steps=" + std::to_string(rep.steps);
    report::Table t(title, {{"Section", 9, report::Align::Left},
                            {"min(s)"},
                            {"avg(s)"},
                            {"max(s)"},
                            {"%varavg"},
                            {"%total"}});
    for (std::size_t c = 0; c < kN; ++c) {
        const auto& s = rep.sections[c];
        t.row({std::string(StepTiming::kNames[c]), report::fmt(s.min_s),
               report::fmt(s.avg_s), report::fmt(s.max_s),
               report::fmt(s.pct_var, 1), report::fmt(s.pct_total, 1)});
    }
    t.row({"Loop", report::fmt(rep.loop_min), report::fmt(rep.loop_avg),
           report::fmt(rep.loop_max), "", "100.0"});
    t.emit();

    // Load-balance line as a second one-row table. `ghost/owned` is the
    // halo width in atoms: it is what a slab-band or hop-count regression
    // moves, and it moves there long before it is visible in Comm seconds.
    report::Table lb("Load balance (atoms/rank)", {{"nlocal_min", 12},
                                                   {"nlocal_avg", 12},
                                                   {"nlocal_max", 12},
                                                   {"nghost_min", 12},
                                                   {"nghost_avg", 12},
                                                   {"nghost_max", 12},
                                                   {"ghost/owned", 12}});
    const double ratio =
        rep.nlocal_avg > 0.0 ? rep.nghost_avg / rep.nlocal_avg : 0.0;
    lb.row({std::to_string(rep.nlocal_min), report::fmt(rep.nlocal_avg, 1),
            std::to_string(rep.nlocal_max), std::to_string(rep.nghost_min),
            report::fmt(rep.nghost_avg, 1), std::to_string(rep.nghost_max),
            report::fmt(ratio, 2)});
    lb.emit();
}

void emit_timing_csv(const ReducedReport& rep, report::Csv& csv,
                     std::string_view stage) {
    // Schema identical to StepTiming::csv_header():
    // stage,ranks,steps,total_s,nlocal_min,nlocal_max,nlocal_avg,<6>_s,<6>_pct
    std::string r =
        std::string(stage) + "," + std::to_string(rep.ranks) + ","
        + std::to_string(rep.steps) + "," + report::fmt(rep.loop_avg, 6) + ","
        + std::to_string(rep.nlocal_min) + "," + std::to_string(rep.nlocal_max)
        + "," + report::fmt(rep.nlocal_avg, 6);
    for (std::size_t c = 0; c < kN; ++c)
        r += "," + report::fmt(rep.sections[c].avg_s, 6);
    for (std::size_t c = 0; c < kN; ++c)
        r += "," + report::fmt(rep.sections[c].pct_total, 6);
    csv.row(r);
}

// ── Memory breakdown ─────────────────────────────────────────────────────

std::size_t read_peak_rss_kb() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "VmHWM: %zu kB", &kb) == 1)
            break;
    }
    std::fclose(f);
    return kb;
}

namespace {
// Map a component name to a memory category index into MemReport::kNames
// {Atoms, Pair, Bonded, Neigh, Comm, Other}. Timing Kspace → Pair bucket
// (electrostatics footprint); Output/Modify → Other.
int mem_category(const std::string& name) {
    if (name == "[atoms]")
        return 0; // Atoms
    if (name == "[domain]")
        return 4; // Comm
    switch (StepTiming::classify(name)) {
    case StepTiming::Pair:
    case StepTiming::Kspace:
        return 1;
    case StepTiming::Bond:
        return 2;
    case StepTiming::Neigh:
        return 3;
    case StepTiming::Comm:
        return 4;
    default:
        return 5; // Output/Modify/Other
    }
}
} // namespace

MemReport
reduce_memory(const std::vector<std::pair<std::string, std::size_t>>& local,
              std::size_t peak_rss_kb) {
    MemReport rep;
    std::array<double, MemReport::kNumCat> cat{};
    for (const auto& [name, bytes] : local)
        cat[static_cast<std::size_t>(mem_category(name))] +=
            static_cast<double>(bytes);

    double rss_mb = static_cast<double>(peak_rss_kb) / 1024.0;
    std::array<double, MemReport::kNumCat> csum = cat;
    double rss_min = rss_mb, rss_max = rss_mb, rss_sum = rss_mb;
    int ranks = 1;
#ifdef ATV_HAS_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        if (ranks > 1) {
            MPI_Allreduce(cat.data(), csum.data(), MemReport::kNumCat,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&rss_mb, &rss_min, 1, MPI_DOUBLE, MPI_MIN,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&rss_mb, &rss_max, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&rss_mb, &rss_sum, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
    }
#endif
    rep.ranks = ranks;
    std::size_t acc = 0;
    for (int c = 0; c < MemReport::kNumCat; ++c) {
        rep.cat_bytes[static_cast<std::size_t>(c)] =
            static_cast<std::size_t>(csum[static_cast<std::size_t>(c)]);
        acc += rep.cat_bytes[static_cast<std::size_t>(c)];
    }
    rep.accounted_bytes = acc;
    rep.rss_sum_mb = rss_sum;
    rep.rss_min_mb = rss_min;
    rep.rss_avg_mb = rss_sum / ranks;
    rep.rss_max_mb = rss_max;
    return rep;
}

void emit_memory_table(const MemReport& rep, std::string_view stage) {
    const double gb = 1024.0 * 1024.0 * 1024.0;
    const double rss_sum_b = rep.rss_sum_mb * 1024.0 * 1024.0;
    const double acc = static_cast<double>(rep.accounted_bytes);
    // "Allocated" = the capacity AV's data structures requested; the % column
    // is share of that AV total (NOT of RSS — allocated capacity includes
    // reserved-but-untouched slack, so it can exceed resident RSS).
    report::Table t(
        "Memory breakdown [" + std::string(stage) + "]  ranks="
            + std::to_string(rep.ranks) + "  (AV-allocated, summed over ranks)",
        {{"Section", 14, report::Align::Left}, {"alloc(GB)"}, {"%AV"}});
    for (int c = 0; c < MemReport::kNumCat; ++c) {
        const double b =
            static_cast<double>(rep.cat_bytes[static_cast<std::size_t>(c)]);
        t.row({std::string(MemReport::kNames[static_cast<std::size_t>(c)]),
               report::fmt(b / gb, 3),
               report::fmt(acc > 0 ? 100.0 * b / acc : 0.0, 1)});
    }
    t.row({"AV total", report::fmt(acc / gb, 3), "100.0"});
    t.emit();

    // Resident footprint vs AV-allocated. When RSS > AV alloc, the remainder is
    // runtime / MPI library state / allocator overhead / libs (not AV data) —
    // the dominant term at high rank counts. When AV alloc > RSS, AV reserved
    // capacity it has not yet touched (lazy pages).
    report::Table pr("Peak RSS (resident) vs AV-allocated ["
                         + std::string(stage) + "]",
                     {{"metric", 18, report::Align::Left},
                      {"sum(GB)"},
                      {"per-rank avg(MB)"}});
    pr.row({"peak RSS (resident)", report::fmt(rss_sum_b / gb, 3),
            report::fmt(rep.rss_avg_mb, 1)});
    pr.row({"AV-allocated", report::fmt(acc / gb, 3),
            report::fmt(rep.ranks ? acc / gb * 1024.0 / rep.ranks : 0.0, 1)});
    const double delta = rss_sum_b - acc;
    pr.row({delta >= 0 ? "runtime/MPI/libs" : "AV reserved (untouched)",
            report::fmt((delta >= 0 ? delta : -delta) / gb, 3),
            report::fmt((delta >= 0 ? delta : -delta) / gb * 1024.0
                            / (rep.ranks ? rep.ranks : 1),
                        1)});
    pr.row({"RSS min/max (MB)",
            report::fmt(rep.rss_min_mb, 1) + " / "
                + report::fmt(rep.rss_max_mb, 1),
            ""});
    pr.emit();
}

void emit_per_component(const TimerRegistry& reg, std::string_view stage) {
    const auto& stats = reg.per_component();
    const auto& keys = reg.keys();
    const int n = static_cast<int>(stats.size());
    std::vector<double> local(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        local[static_cast<std::size_t>(i)] =
            stats[static_cast<std::size_t>(i)].total_s;
    std::vector<double> gmax = local;
    int rank = 0;
#ifdef ATV_HAS_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        int ranks = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ranks > 1 && n > 0)
            MPI_Allreduce(local.data(), gmax.data(), n, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
    }
#endif
    if (rank != 0)
        return;
    report::Table t("Per-component [" + std::string(stage)
                        + "] (max-over-ranks s)",
                    {{"component", 22, report::Align::Left},
                     {"category", 9, report::Align::Left},
                     {"calls"},
                     {"max_s"}});
    for (int i = 0; i < n; ++i) {
        const auto& k = keys[static_cast<std::size_t>(i)];
        const char* cat =
            (k.category >= 0
             && k.category < static_cast<int>(StepTiming::kNumCategories))
                ? StepTiming::kNames[static_cast<std::size_t>(k.category)]
                      .data()
                : "?";
        t.row({k.component.empty() ? "(scoped)" : k.component, cat,
               std::to_string(stats[static_cast<std::size_t>(i)].calls),
               report::fmt(gmax[static_cast<std::size_t>(i)], 6)});
    }
    t.emit();

    // Scoped sub-regions (ATV_PROFILE_SCOPE), rank 0's local values. Keys
    // can differ across ranks (an FFT-subgroup member times regions a
    // non-member never enters), so no cross-rank reduction is attempted —
    // the table is a rank-0 drill-down, not a collective section.
    if (!reg.scoped().empty()) {
        std::vector<std::pair<std::string, TimerStat>> rows(
            reg.scoped().begin(), reg.scoped().end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        report::Table st(
            "Scoped regions [" + std::string(stage) + "] (rank 0 s)",
            {{"region", 22, report::Align::Left}, {"calls"}, {"rank0_s"}});
        for (const auto& [key, stat] : rows)
            st.row({key, std::to_string(stat.calls),
                    report::fmt(stat.total_s, 6)});
        st.emit();
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
