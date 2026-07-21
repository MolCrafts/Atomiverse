/// @file tests/cuda/test_pair_lj_no_d2h.cu
/// @brief Static-grep gate for `cuda-md-review-cleanup-03-launch-purity`
///        ac-001 / ac-003: `cuda::PairLJCut::compute()` must contain
///        zero per-step `cudaStreamSynchronize` and zero D2H copies on
///        the hot path.
///
/// Runs at test-time (not build-time) so the grep is automatic without
/// adding a custom CMake target. Reads the source file as plain text
/// and asserts the forbidden patterns do not appear. The single
/// remaining D2H — a one-time `cudaMemcpyAsync` + `cudaStreamSynchronize`
/// gated on `pair_capacity_host_ == 0` in build()-equivalent first-launch
/// path — is allowed because it fires exactly once per simulation
/// (cached afterward).

#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <sstream>
#include <string>

namespace {

/// Strip C/C++ comments before searching so wording like
/// "the previous cudaStreamSynchronize was removed" inside a comment
/// does not trip the grep gate. Mirrors the helper in
/// `test_driver_no_per_phase_sync.cu`.
std::string strip_comments(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    bool in_line = false;
    bool in_block = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        char c = src[i];
        if (in_line) {
            if (c == '\n') {
                in_line = false;
                out.push_back('\n');
            }
            continue;
        }
        if (in_block) {
            if (c == '*' && i + 1 < src.size() && src[i + 1] == '/') {
                in_block = false;
                ++i;
            }
            continue;
        }
        if (c == '/' && i + 1 < src.size()) {
            if (src[i + 1] == '/') {
                in_line = true;
                ++i;
                continue;
            }
            if (src[i + 1] == '*') {
                in_block = true;
                ++i;
                continue;
            }
        }
        out.push_back(c);
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        ADD_FAILURE() << "Cannot open " << path
                      << " — set ATV_SOURCE_DIR or place this test next to "
                         "the source tree.";
        return {};
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// Pull `PairLJCut::compute()` out of the source so the grep is scoped
/// to the hot-path function (build()-equivalent first-launch path is
/// allowed to sync).
std::string extract_compute_body(const std::string& src) {
    // Match from `void PairLJCut::compute(` to the matching closing brace.
    // Heuristic: find function signature line, then count braces.
    auto sig = src.find("void PairLJCut::compute(");
    if (sig == std::string::npos)
        return src; // fall back to whole file
    auto brace = src.find('{', sig);
    if (brace == std::string::npos)
        return src;
    int depth = 1;
    std::size_t i = brace + 1;
    while (i < src.size() && depth > 0) {
        if (src[i] == '{')
            ++depth;
        else if (src[i] == '}')
            --depth;
        ++i;
    }
    return src.substr(sig, i - sig);
}

} // namespace

#ifndef ATV_SOURCE_DIR
#define ATV_SOURCE_DIR "."
#endif

// ── ac-001: no cudaStreamSynchronize OUTSIDE the first-launch cache path ──

TEST(PairLJNoD2H, ComputeHasZeroStreamSyncOnHotPath) {
    const std::string path =
        std::string(ATV_SOURCE_DIR) + "/src/cuda/potentials/pair_lj_cut.cu";
    const std::string body =
        strip_comments(extract_compute_body(read_file(path)));
    ASSERT_FALSE(body.empty()) << "compute() body extraction failed";

    // The first-launch cache path (gated by `pair_capacity_host_ == 0`)
    // contains a one-time `cudaStreamSynchronize` for the capacity D2H.
    // Strip that gated block from the body before checking; any remaining
    // `cudaStreamSynchronize` is a hot-path violation.
    std::regex first_launch_block(
        R"(if\s*\(\s*pair_capacity_host_\s*==\s*0\s*\)\s*\{[^}]*\})");
    std::string hot_path = std::regex_replace(body, first_launch_block, "");

    // Match actual call: `cudaStreamSynchronize(`
    std::regex call_pattern(R"(cudaStreamSynchronize\s*\()");
    std::smatch m;
    EXPECT_FALSE(std::regex_search(hot_path, m, call_pattern))
        << "PairLJCut::compute() contains a cudaStreamSynchronize call on "
           "the hot path (i.e. outside the gated first-launch capacity "
           "read). The per-step grid-sizing D2H was removed in "
           "cuda-md-review-cleanup-03-launch-purity; if this fires, the "
           "regression has been re-introduced.";
}

// ── ac-001: no D2H cudaMemcpy on the hot path ──

TEST(PairLJNoD2H, ComputeHasZeroD2HOnHotPath) {
    const std::string path =
        std::string(ATV_SOURCE_DIR) + "/src/cuda/potentials/pair_lj_cut.cu";
    const std::string body =
        strip_comments(extract_compute_body(read_file(path)));
    ASSERT_FALSE(body.empty()) << "compute() body extraction failed";

    std::regex first_launch_block(
        R"(if\s*\(\s*pair_capacity_host_\s*==\s*0\s*\)\s*\{[^}]*\})");
    std::string hot_path = std::regex_replace(body, first_launch_block, "");

    EXPECT_EQ(hot_path.find("cudaMemcpyDeviceToHost"), std::string::npos)
        << "PairLJCut::compute() contains a cudaMemcpyDeviceToHost on the "
           "hot path. The per-step n_pairs readback was eliminated in "
           "cuda-md-review-cleanup-03-launch-purity.";
}
