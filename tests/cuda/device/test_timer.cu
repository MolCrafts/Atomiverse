/// @file tests/cuda/device/test_timer.cu
/// @brief Event-slot reuse must preserve every recorded profiler operation.
#include "atomiverse/cuda/device/timer.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <sstream>

TEST(CudaTimer, BeginReusesBoundedEventsWithoutLosingOperations) {
    ATV::cuda::Timer::Impl timer;
    timer.init();
    for (int i = 0; i < 100; ++i) {
        timer.begin(nullptr);
        timer.end(nullptr, 0, "capacity_probe");
        timer.step_done();
    }
    timer.flush();
    const auto summary = timer.summary({"force"});
    timer.shutdown();
    std::istringstream lines(summary);
    std::string line;
    bool found = false;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        std::string component;
        fields >> component;
        if (component == "capacity_probe") {
            double ms;
            int calls;
            ASSERT_TRUE(bool(fields >> ms >> calls));
            EXPECT_EQ(calls, 100);
            EXPECT_TRUE(std::isfinite(ms));
            EXPECT_GE(ms, 0);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}
