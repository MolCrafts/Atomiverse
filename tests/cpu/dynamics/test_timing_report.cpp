/// @file tests/cpu/md/test_timing_report.cpp
/// @brief TimerRegistry mechanics, cross-rank reduce (single rank), and the
/// log-module report path (Table render + CSV via metrics file sink).

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "atomiverse/core/log.h"
#include "atomiverse/core/report.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/record/timing_report.h"

using namespace ATV;
using cpu::md::StepTiming;

namespace {
int handle(TimerRegistry& r, const char* name, int phase) {
    return r.register_op(TimerKey{
        name, phase, static_cast<int>(StepTiming::classify(name)), ""});
}
} // namespace

TEST(TimerRegistry, RecordRollupReset) {
    TimerRegistry r;
    r.set_enabled(true);
    const int h_lj = handle(r, "lj", 2);              // Pair
    const int h_bond = handle(r, "harmonic_bond", 2); // Bond
    const int h_vv = handle(r, "velocity_verlet", 0); // Modify
    const int h_pp = handle(r, "pppm", 2);            // Kspace

    r.record(h_lj, 1.0);
    r.record(h_lj, 3.0); // Pair total 4.0, 2 calls, min1 max3
    r.record(h_bond, 0.5);
    r.record(h_vv, 0.25);
    r.record(h_pp, 2.0);
    r.set_loop(5.0, 1234, 5678);
    r.tick_step();

    const auto roll = r.rollup_by_category();
    // Rollup is sized to (max registered category id + 1), not full enum.
    ASSERT_GT(roll.size(), static_cast<std::size_t>(StepTiming::Modify));
    EXPECT_DOUBLE_EQ(roll[StepTiming::Pair].total_s, 4.0);
    EXPECT_EQ(roll[StepTiming::Pair].calls, 2u);
    EXPECT_DOUBLE_EQ(roll[StepTiming::Pair].min_s, 1.0);
    EXPECT_DOUBLE_EQ(roll[StepTiming::Pair].max_s, 3.0);
    EXPECT_DOUBLE_EQ(roll[StepTiming::Bond].total_s, 0.5);
    EXPECT_DOUBLE_EQ(roll[StepTiming::Modify].total_s, 0.25);
    EXPECT_DOUBLE_EQ(roll[StepTiming::Kspace].total_s, 2.0);
    EXPECT_DOUBLE_EQ(r.loop_seconds(), 5.0);
    EXPECT_EQ(r.nlocal(), 1234);
    EXPECT_EQ(r.steps(), 1u);

    r.reset();
    EXPECT_EQ(r.steps(), 0u);
    EXPECT_DOUBLE_EQ(r.loop_seconds(), 0.0);
    // handles preserved across reset
    r.record(h_lj, 2.0);
    EXPECT_DOUBLE_EQ(r.rollup_by_category()[StepTiming::Pair].total_s, 2.0);
}

TEST(TimerRegistry, ScopedTimerRecordsOnlyWhenEnabled) {
    TimerRegistry off;
    {
        ScopedTimer st(&off, "region", 2, 0);
    }
    EXPECT_TRUE(off.scoped().empty());

    TimerRegistry on;
    on.set_enabled(true);
    {
        ScopedTimer st(&on, "region", 2, 0);
        volatile double x = 0;
        for (int i = 0; i < 100000; ++i)
            x += i; // burn a little time
        (void)x;
    }
    ASSERT_EQ(on.scoped().count("region"), 1u);
    EXPECT_EQ(on.scoped().at("region").calls, 1u);
}

TEST(TimerReport, ReduceSingleRank) {
    TimerRegistry r;
    r.set_enabled(true);
    r.record(handle(r, "lj", 2), 4.0);            // Pair
    r.record(handle(r, "harmonic_bond", 2), 1.0); // Bond
    r.set_loop(5.0, 100, 250);
    r.tick_step();

    const auto rep = cpu::md::reduce_timing(r);
    EXPECT_EQ(rep.ranks, 1);
    // Ghost rows are reduced beside owned rows: halo over-forwarding is
    // visible here before it is visible as Comm seconds.
    EXPECT_EQ(rep.nlocal_min, 100);
    EXPECT_EQ(rep.nlocal_max, 100);
    EXPECT_EQ(rep.nghost_min, 250);
    EXPECT_EQ(rep.nghost_max, 250);
    EXPECT_DOUBLE_EQ(rep.nghost_avg, 250.0);
    EXPECT_DOUBLE_EQ(rep.sections[StepTiming::Pair].avg_s, 4.0);
    EXPECT_DOUBLE_EQ(rep.sections[StepTiming::Pair].min_s, 4.0);
    EXPECT_DOUBLE_EQ(rep.sections[StepTiming::Pair].max_s, 4.0);
    EXPECT_NEAR(rep.sections[StepTiming::Pair].pct_total, 80.0, 1e-9);
    EXPECT_NEAR(rep.sections[StepTiming::Bond].pct_total, 20.0, 1e-9);
    EXPECT_DOUBLE_EQ(rep.loop_avg, 5.0);
    EXPECT_EQ(rep.nlocal_min, 100);
    EXPECT_EQ(rep.nlocal_max, 100);
}

TEST(Report, TableRendersHeadersAndRows) {
    report::Table t("My Title", {{"Section", 8, report::Align::Left}, {"avg"}});
    t.row({"Pair", "1.234"});
    const std::string s = t.to_string();
    EXPECT_NE(s.find("My Title"), std::string::npos);
    EXPECT_NE(s.find("Section"), std::string::npos);
    EXPECT_NE(s.find("Pair"), std::string::npos);
    EXPECT_NE(s.find("1.234"), std::string::npos);
}

TEST(Report, CsvRoutesThroughMetricsFile) {
    const std::string path = "atv_test_metrics.csv";
    std::remove(path.c_str());
    LogConfig lc;
    lc.metrics_file = path;
    log_init(lc); // rank 0 default -> metrics sink active

    {
        report::Csv csv(StepTiming::csv_header());
        csv.row(
            "stage1,1,10,0.5,100,100,100,0.4,0.05,0.05,0,0,0,80,10,10,0,0,0");
    }
    metrics_logger().flush();

    std::ifstream in(path);
    ASSERT_TRUE(in.good());
    std::string header, row;
    std::getline(in, header);
    std::getline(in, row);
    EXPECT_EQ(header, StepTiming::csv_header());
    EXPECT_EQ(row.rfind("stage1,1,10,", 0), 0u); // starts with
    std::remove(path.c_str());
}
