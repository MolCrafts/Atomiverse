/// @file tests/cuda/async/test_async_record_correctness.cu
/// @brief Async vs sync recorder CSV field-by-field parity gate.
///
/// **Not registered under ctest** (tester contract: no e2e under `tests/`).
/// This TU shells out via `system()` to the full `polyethylene_tension`
/// binary — re-home under `benchmarks/` or `regressions/` when that path is
/// rebuilt. Source kept as a reference harness only.
///
/// Runs `polyethylene_tension --backend cuda --mode stage:2 --scale 0.001`
/// twice with the same seed: once with the default AsyncRecordBase path
/// (env `ATV_TEST_RECORD_SYNC=0`) and once with the synchronous D2H path
/// (env `ATV_TEST_RECORD_SYNC=1`). Compares every numeric column in the
/// resulting thermo.csv files using a relative tolerance — async/sync
/// paths must produce equivalent rows within `|x_a - x_s| <= 1e-6 *
/// max(|x_s|, 1.0)`. Row-count difference between the two CSVs is
/// allowed up to 1 (async path may drop a single trailing row at exit
/// boundaries; ac-005 declares this explicitly).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#ifndef ATV_PE_TENSION_BIN
#error                                                                         \
    "ATV_PE_TENSION_BIN must be defined to the polyethylene_tension binary path"
#endif

namespace {

/// @brief Run the polyethylene_tension benchmark in a fresh out-dir.
/// @param out_dir Where the benchmark writes thermo.csv.
/// @param sync_mode Value passed via `ATV_TEST_RECORD_SYNC` env var
///        (0 = async path, 1 = forced synchronous D2H path).
/// @return system() exit code.
int run_pe_tension(const std::string& out_dir, int sync_mode) {
    std::filesystem::create_directories(out_dir);
    std::string cmd = "ATV_TEST_RECORD_SYNC=" + std::to_string(sync_mode) + " "
                      + std::string(ATV_PE_TENSION_BIN)
                      + " --backend cuda --mode stage:2 --scale 0.001"
                      + " --seed 12345 --print-every 1" + " --quiet --out-dir "
                      + out_dir + " > /dev/null 2>&1";
    return std::system(cmd.c_str());
}

/// @brief Parse a thermo.csv into rows of numeric columns. Skips the header.
/// @param path CSV file path.
/// @return Vector of row vectors. Empty vector if the file cannot be opened.
std::vector<std::vector<double>> load_csv(const std::string& path) {
    std::vector<std::vector<double>> rows;
    std::ifstream in(path);
    if (!in.is_open())
        return rows;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first) {
            first = false;
            continue; // header
        }
        if (line.empty())
            continue;
        std::vector<double> cols;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                cols.push_back(std::stod(cell));
            } catch (...) {
                // non-numeric cell: skip silently.
            }
        }
        rows.push_back(std::move(cols));
    }
    return rows;
}

} // namespace

TEST(CudaAsyncRecordCorrectness, AsyncMatchesSync) {
    const std::string async_dir =
        (std::filesystem::temp_directory_path() / "atv_pe_async").string();
    const std::string sync_dir =
        (std::filesystem::temp_directory_path() / "atv_pe_sync").string();
    std::filesystem::remove_all(async_dir);
    std::filesystem::remove_all(sync_dir);

    ASSERT_EQ(run_pe_tension(async_dir, /*sync_mode=*/0), 0)
        << "polyethylene_tension (async path) failed to run";
    ASSERT_EQ(run_pe_tension(sync_dir, /*sync_mode=*/1), 0)
        << "polyethylene_tension (sync path) failed to run";

    auto async_rows = load_csv(async_dir + "/thermo.csv");
    auto sync_rows = load_csv(sync_dir + "/thermo.csv");

    ASSERT_FALSE(async_rows.empty()) << "async thermo.csv has no rows";
    ASSERT_FALSE(sync_rows.empty()) << "sync thermo.csv has no rows";

    // ac-005: row-count difference must be <= 1.
    const int da = static_cast<int>(async_rows.size());
    const int ds = static_cast<int>(sync_rows.size());
    EXPECT_LE(std::abs(da - ds), 1)
        << "async vs sync row count differs by more than 1: async=" << da
        << " sync=" << ds;

    const std::size_t n_compare = std::min(async_rows.size(), sync_rows.size());
    int compared_cells = 0;
    for (std::size_t i = 0; i < n_compare; ++i) {
        const auto& a = async_rows[i];
        const auto& s = sync_rows[i];
        ASSERT_EQ(a.size(), s.size())
            << "row " << i << " column count mismatch";
        for (std::size_t c = 0; c < a.size(); ++c) {
            // Treat NaN/Inf as "both runs blew up identically" -- the
            // test gates field equivalence between async and sync, not
            // physical stability. If both are non-finite the row carries
            // no signal; if exactly one is non-finite that is a mismatch.
            const bool a_finite = std::isfinite(a[c]);
            const bool s_finite = std::isfinite(s[c]);
            if (!a_finite && !s_finite)
                continue;
            ASSERT_TRUE(a_finite && s_finite)
                << "row " << i << " col " << c
                << " finiteness mismatch: async=" << a[c] << " sync=" << s[c];
            const double tol = 1e-6 * std::max(std::abs(s[c]), 1.0);
            EXPECT_NEAR(a[c], s[c], tol)
                << "row " << i << " col " << c << " async=" << a[c]
                << " sync=" << s[c];
            ++compared_cells;
        }
    }
    ASSERT_GT(compared_cells, 0)
        << "no finite cells compared -- both runs produced all-NaN output, "
           "test cannot prove parity";
}
