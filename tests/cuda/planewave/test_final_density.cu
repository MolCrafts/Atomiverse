#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include <array>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
using namespace ATV;
namespace host = cpu::pw::internal;
namespace dev = cuda::pw::internal;
struct Resources {
    cudaStream_t stream = nullptr;
    std::vector<void*> p;
    Resources() {
        ATV_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Resources() {
        cudaStreamSynchronize(stream);
        for (auto* q : p)
            cudaFree(q);
        cudaStreamDestroy(stream);
    }
    template <class T>
    T* allocate(std::size_t n) {
        T* q = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&q, n * sizeof(T)));
        p.push_back(q);
        return q;
    }
};
void require(bool ok, const char* s) {
    if (!ok)
        throw std::runtime_error(s);
}
} // namespace
TEST(CudaPWFinalDensity, RetainsIsolatedTinyTailsAndPreservesLegacyBehavior) {
    constexpr int n = 4, nr = 64;
    constexpr double volume = 120, tiny = 5e-15;
    std::vector<double> actual(nr), sentinel(nr, .125);
    host::PWProblem p;
    host::GVectorList g;
    g.nx = g.ny = g.nz = n;
    g.h = {0, 1, 0};
    g.k = {0, 0, 1};
    g.l = {0, 0, 0};
    g.g = {{0, 0, 0}, {2 * M_PI / 4, 0, 0}, {0, 2 * M_PI / 5, 0}};
    for (const auto& q : g.g)
        g.ekin.push_back(.5 * q.norm_squared());
    p.gv0 = g;
    p.gv_k = {g};
    p.vol_bohr = volume;
    p.n_bands = 2;
    p.ws.gvec = &p.gv0;
    p.ws.n_g = g.count();
    p.ws.omega = volume;
    p.ws.fft_nx = p.ws.fft_ny = p.ws.fft_nz = n;
    p.ws.nr = nr;
    Resources gpu;
    dev::DeviceBandSolver solver(n, n, n, 2, gpu.stream);
    auto* rho = gpu.allocate<double>(nr);
    auto* veff = gpu.allocate<double>(nr);
    ATV_CUDA_CHECK(cudaMemsetAsync(rho, 0, nr * sizeof(double), gpu.stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(veff, 0, nr * sizeof(double), gpu.stream));
    solver.bind_grids(rho, veff);
    solver.build_kpoints(p);
    solver.bind_kpoint(0);
    solver.solve(p.ws, 20);
    require(p.ws.eigenvalues.size() == 2, "free-electron two-band shape");
    auto capture = [&](bool exact, const std::array<double, 2>& f) {
        ATV_CUDA_CHECK(
            cudaMemsetAsync(rho, 0, nr * sizeof(double), gpu.stream));
        if (exact)
            solver.accumulate_final_density(1., volume, f);
        else
            solver.accumulate_density(1., volume, f.data(), 2);
        ATV_CUDA_CHECK(cudaMemcpyAsync(actual.data(), rho, nr * sizeof(double),
                                       cudaMemcpyDeviceToHost, gpu.stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        return actual;
    };
    double worst_scaled = 0;
    for (double tail : {tiny, -tiny, 1e-16}) {
        const auto legacy = capture(false, {0, tail}),
                   exact = capture(true, {0, tail});
        double integral = 0, old_integral = 0, grid_error = 0;
        for (int i = 0; i < nr; ++i) {
            require(legacy[i] == 0, "legacy tiny cutoff retained");
            grid_error =
                std::max(grid_error, std::abs(exact[i] * volume / tail - 1));
            integral += exact[i] * volume / nr;
            old_integral += legacy[i] * volume / nr;
        }
        const double scaled = std::abs(integral / tail - 1),
                     old_scaled = std::abs(old_integral / tail - 1);
        require(grid_error < 1e-12 && scaled < 1e-12 && old_scaled == 1,
                "isolated tail scaled density control");
        worst_scaled = std::max(worst_scaled, scaled);
        std::printf(
            "DENSITY_TAIL f=%.17g integrated=%.17g old_scaled_error=%.17g "
            "exact_scaled_error=%.17g grid_scaled_error=%.17g\n",
            tail, integral, old_scaled, scaled, grid_error);
    }
    const auto old_normal = capture(false, {.3, .7}),
               new_normal = capture(true, {.3, .7});
    for (int i = 0; i < nr; ++i) {
        require(old_normal[i] == new_normal[i],
                "normal legacy density unchanged bitwise");
        require(std::abs(new_normal[i] - 1 / volume) < 1e-15,
                "two-band normal analytic density");
    }
    const auto empty = capture(true, {0, 0});
    for (double r : empty)
        require(r == 0, "exact empty occupation zero");
    ATV_CUDA_CHECK(cudaMemcpyAsync(rho, sentinel.data(), nr * sizeof(double),
                                   cudaMemcpyHostToDevice, gpu.stream));
    bool rejected = false;
    try {
        solver.accumulate_final_density(
            1., volume,
            std::array<double, 2>{.3,
                                  std::numeric_limits<double>::quiet_NaN()});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "nonfinite final f rejected");
    ATV_CUDA_CHECK(cudaMemcpyAsync(actual.data(), rho, nr * sizeof(double),
                                   cudaMemcpyDeviceToHost, gpu.stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    require(actual == sentinel, "validation before density mutation");
    std::printf("PASS EXACT_FINAL_DENSITY bands=2 tiny_f=%.17g "
                "worst_relative_tail_error=%.17g legacy_tiny_zero=1 "
                "ordinary_bitwise_equal=1 "
                "nonfinite_fail_before_mutation=1 scf_solves=0\n",
                tiny, worst_scaled);
}
