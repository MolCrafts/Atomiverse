/// @file test_fft.cpp
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/planewave/internal/fft.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(FFT, RoundTrip) {
    int nx = 12, ny = 12, nz = 12;
    FFTPlan3D plan(nx, ny, nz);
    int nr = nx * ny * nz;
    int nc = (nx / 2 + 1) * ny * nz;

    std::vector<AccReal> orig(nr);
    std::mt19937 rng(42);
    std::uniform_real_distribution<AccReal> dist(-1, 1);
    for (int i = 0; i < nr; ++i)
        orig[i] = dist(rng);

    auto* cbuf = fftw_alloc_complex(static_cast<size_t>(nc));
    std::vector<AccReal> back(nr);

    plan.r2c(orig.data(), cbuf);
    plan.c2r(cbuf, back.data());

    // FFTW c2r is unnormalized: back[i] = nr * orig[i]. Normalize for round-trip check.
    const AccReal norm = AccReal(1) / static_cast<AccReal>(nr);
    AccReal max_err = 0;
    for (int i = 0; i < nr; ++i)
        max_err = std::max(max_err, std::abs(back[i] * norm - orig[i]));
    fftw_free(cbuf);
    EXPECT_LT(static_cast<double>(max_err), 1e-10);
}
