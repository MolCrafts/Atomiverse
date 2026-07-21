/// @file tests/cuda/test_driver_no_per_phase_sync.cu
/// @brief Static-grep gate for `cuda-md-review-cleanup-03-launch-purity`
///        ac-005: `Backend::step()` must contain zero per-phase
///        `cudaStreamSynchronize` calls inside the slot loop.
///
/// Reads `src/cuda/driver/driver.cu` and extracts `Backend::step()`. The
/// previous implementation issued `cudaStreamSynchronize(main_stream) +
/// cudaStreamSynchronize(defer_stream)` per phase (~12 host blocks per
/// step). The link-03 rewrite relies on intra-stream FIFO ordering +
/// AsyncBridge's explicit event-based cross-stream ordering, so neither
/// sync remains inside the slot loop.

#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        ADD_FAILURE() << "Cannot open " << path;
        return {};
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string extract_step_body(const std::string& src) {
    auto sig = src.find("void Backend::step(");
    if (sig == std::string::npos)
        return {};
    auto brace = src.find('{', sig);
    if (brace == std::string::npos)
        return {};
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

/// Count actual function CALLS, ignoring occurrences inside comments
/// and string literals. Requires `Foo(` (with optional whitespace
/// between the name and `(`) anchored outside `//` or `/* */`.
int count_calls(const std::string& src, const std::string& fn) {
    // Crude comment stripper: line comments and block comments.
    std::string stripped;
    stripped.reserve(src.size());
    bool in_line = false;
    bool in_block = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        char c = src[i];
        if (in_line) {
            if (c == '\n')
                in_line = false;
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
        stripped.push_back(c);
    }
    // Match `fn` followed by optional whitespace + `(`.
    std::regex call(fn + R"(\s*\()");
    return static_cast<int>(std::distance(
        std::sregex_iterator(stripped.begin(), stripped.end(), call),
        std::sregex_iterator()));
}

TEST(DriverNoPerPhaseSync, StepHasZeroStreamSync) {
    const std::string path =
        std::string(ATV_SOURCE_DIR) + "/src/cuda/driver/driver.cu";
    const std::string body = extract_step_body(read_file(path));
    ASSERT_FALSE(body.empty()) << "Backend::step() body extraction failed";

    EXPECT_EQ(count_calls(body, "cudaStreamSynchronize"), 0)
        << "Backend::step() contains a cudaStreamSynchronize call. The "
           "per-phase double-stream sync was removed in "
           "cuda-md-review-cleanup-03-launch-purity (ac-005); "
           "components rely on intra-stream FIFO ordering and "
           "AsyncBridge's event-based cross-stream ordering instead. "
           "Backend::sync() / retrieve() may still sync at end-of-run.";
}

TEST(DriverNoPerPhaseSync, AsyncBridgeUsesLaunchHostFunc) {
    const std::string path =
        std::string(ATV_SOURCE_DIR) + "/src/cuda/async/async_bridge.cu";
    const std::string src = read_file(path);
    ASSERT_FALSE(src.empty());

    EXPECT_GT(count_calls(src, "cudaLaunchHostFunc"), 0)
        << "async_bridge.cu should use cudaLaunchHostFunc to dispatch "
           "the on_data_ready hook from a worker thread (ac-006). The "
           "previous cudaStreamSynchronize(defer_stream) before "
           "on_data_ready was the only AsyncBridge sync remaining in "
           "the hot path.";

    // Allow exactly one cudaStreamSynchronize call — the destroy()-time
    // drain that ensures pending callbacks complete before pinned
    // buffers are freed.
    int count = count_calls(src, "cudaStreamSynchronize");
    EXPECT_LE(count, 1)
        << "async_bridge.cu has " << count
        << " cudaStreamSynchronize call(s); only the destroy()-time drain "
           "is permitted (one call max).";
}
