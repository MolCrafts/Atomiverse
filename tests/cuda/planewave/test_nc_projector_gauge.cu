// Real DeviceBandSolver H application: no SCF, private Ritz access or new API.
#include "../../cpu/planewave/nc_gauge_fixture.h"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {
using namespace nc_gauge_test;
struct DeviceGrids {
    cudaStream_t stream;
    AccReal* rho = nullptr;
    AccReal* veff = nullptr;
    explicit DeviceGrids(cudaStream_t value) : stream(value) {}
    ~DeviceGrids() {
        cudaStreamSynchronize(stream);
        cudaFree(veff);
        cudaFree(rho);
    }
};
TEST(PWNCGaugeCUDA, FullHamiltonianPreservesProjectorNormalization) {
    for (bool nonzero_k : {false, true}) {
        const auto g = modes(nonzero_k);
        const auto c = wave(g);
        const auto original = potential(1, nonzero_k);
        PWSetup ref;
        PWWorkspace ref_ws;
        prepare(ref, ref_ws, original, g, c);
        const auto expected = full_d_h(original, ref, c);
        AccReal nonlocal_norm = 0;
        for (std::size_t i = 0; i < c.size(); ++i)
            nonlocal_norm +=
                std::norm(expected[i] - g.ekin[i % g.count()] * c[i]);
        ASSERT_GT(nonlocal_norm, 1e-4);
        for (AccReal scale : {AccReal(1), AccReal(1e9), AccReal(1e-35)}) {
            SCOPED_TRACE(::testing::Message()
                         << "k=" << nonzero_k << " scale=" << scale);
            const auto d = potential(scale, nonzero_k);
            PWSetup p;
            PWWorkspace p_ws;
            prepare(p, p_ws, d, g, c);
            cuda::pw::internal::DeviceBandSolver solver(g.nx, g.ny, g.nz, 3);
            DeviceGrids grids(solver.stream());
            const auto bytes = std::size_t(p_ws.nr) * sizeof(AccReal);
            ASSERT_EQ(cudaMalloc(&grids.rho, bytes), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&grids.veff, bytes), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(grids.rho, 0, bytes, grids.stream),
                      cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(grids.veff, 0, bytes, grids.stream),
                      cudaSuccess);
            solver.bind_grids(grids.rho, grids.veff);
            solver.build_kpoints(p.mesh, p.ions.positions, p.ions.vol_bohr,
                                 p_ws.wfc);
            solver.bind_kpoint(0);
            std::vector<C> actual(c.size());
            for (int b = 0; b < 3; ++b)
                solver.apply_h_host(c.data() + b * g.count(),
                                    actual.data() + b * g.count());
            for (std::size_t i = 0; i < c.size(); ++i) {
                const AccReal gate = 1e-8 + 1e-7 * std::abs(expected[i]);
                EXPECT_LE(std::abs(actual[i] - expected[i]), gate)
                    << "component " << i;
            }
        }
    }
}
} // namespace
