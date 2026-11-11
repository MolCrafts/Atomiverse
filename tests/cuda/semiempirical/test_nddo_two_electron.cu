/// @file tests/cuda/semiempirical/test_nddo_two_electron.cu
/// @brief Unit checks of CUDA pair ERI/Fock primitives, never an SCF solve.
#include "atomiverse/cpu/semiempirical/internal/nddo_derivative.h"
#include "atomiverse/cuda/semiempirical/internal/nddo_forces.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_hcore.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_two_electron.cuh"

#include "atomiverse/cpu/semiempirical/nddo_fock.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/rm1_parameters.h"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace gpu = ATV::cuda::semiempirical::internal;
namespace cpu = ATV::cpu::semiempirical;

class CudaNDDOTwoCenter : public ::testing::TestWithParam<bool> {
  protected:
    static constexpr int na = 8, n = 26, nn = n * n, np = na * (na - 1) / 2;
    cpu::NDDOSystem system;
    std::array<double, na> x{0.0, 1.1, 0.0, 0.0, 0.2, 2.7, -0.8, 1.6};
    std::array<double, na> y{0.0, 0.0, 1.5, 0.0, 0.8, -0.4, 0.4, 2.2};
    std::array<double, na> z{0.0, 0.0, 0.0, -1.2, 1.3, 0.6, 2.1, -0.7};
    std::array<ATV::StorageReal, na> hx{}, hy{}, hz{};
    std::array<gpu::NDDOAtom, na> descriptors{};
    std::vector<double> density, spin, hcore, packed_p, packed_spin, packed_h;
    cudaStream_t stream = nullptr;
    gpu::NDDOAtom* atoms = nullptr;
    double* data = nullptr;
    double *dp = nullptr, *ds = nullptr, *dh = nullptr, *df = nullptr,
           *dx = nullptr, *dy = nullptr, *dz = nullptr, *tiles = nullptr,
           *energy = nullptr;
    int* status = nullptr;
    void* scratch = nullptr;
    std::size_t scratch_bytes = 0;

    void upload_geometry() {
        for (int i = 0; i < na; ++i) {
            // Compare the same represented coordinates in storage-float builds.
            hx[i] = x[i];
            hy[i] = y[i];
            hz[i] = z[i];
            x[i] = hx[i];
            y[i] = hy[i];
            z[i] = hz[i];
        }
        ASSERT_EQ(cudaMemcpyAsync(dx, x.data(), sizeof(x),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(dy, y.data(), sizeof(y),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(dz, z.data(), sizeof(z),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }

    // Canonical CPU analytic pair derivatives contracted at the same fixed P;
    // no SCF and no displaced-coordinate production reference.
    std::array<double, 3 * na> reference_forces() const {
        std::array<double, 3 * na> force{};
        for (int i = 1; i < na; ++i)
            for (int j = 0; j < i; ++j) {
                cpu::NDDOSystem pair;
                pair.atoms = {system.atoms[j], system.atoms[i]};
                const int nj = pair.atoms[0].n_orbitals;
                const int ni = pair.atoms[1].n_orbitals;
                pair.n_orbitals = ni + nj;
                pair.atoms[0].first_ao = 0;
                pair.atoms[0].last_ao = nj - 1;
                pair.atoms[1].first_ao = nj;
                pair.atoms[1].last_ao = nj + ni - 1;
                const std::array<ATV::StorageReal, 2> xx{hx[j], hx[i]},
                    yy{hy[j], hy[i]}, zz{hz[j], hz[i]};
                const std::array<double, 3> xi{hx[i], hy[i], hz[i]},
                    xj{hx[j], hy[j], hz[j]};
                const auto dh =
                    cpu::internal::hcore_pair_derivative(pair, xi, xj);
                const auto dw = cpu::internal::two_center_integral_derivative(
                    pair, xx.data(), yy.data(), zz.data(), 1, 0);
                const auto element = [&](int a, int b) {
                    const int ga = a < nj ? system.atoms[j].first_ao + a
                                          : system.atoms[i].first_ao + a - nj;
                    const int gb = b < nj ? system.atoms[j].first_ao + b
                                          : system.atoms[i].first_ao + b - nj;
                    return density[gb * n + ga];
                };
                for (int d = 0; d < 3; ++d) {
                    double g = dh[d].nuclear_repulsion_ev;
                    for (int a = 0; a < ni + nj; ++a)
                        for (int b = 0; b <= a; ++b)
                            g += (a == b ? 1.0 : 2.0) * element(a, b)
                                 * dh[d].packed_hcore[cpu::packed_index(a, b)];
                    int ip = 0;
                    for (int a = 0; a < ni; ++a)
                        for (int b = 0; b <= a; ++b, ++ip) {
                            int jp = 0;
                            for (int c = 0; c < nj; ++c)
                                for (int e = 0; e <= c; ++e, ++jp) {
                                    const int aa = a + nj, bb = b + nj;
                                    const double exchange =
                                        0.125
                                        * (element(aa, c) * element(bb, e)
                                           + element(aa, e) * element(bb, c)
                                           + element(bb, c) * element(aa, e)
                                           + element(bb, e) * element(aa, c));
                                    g += (a == b ? 1.0 : 2.0)
                                         * (c == e ? 1.0 : 2.0)
                                         * (element(aa, bb) * element(c, e)
                                            - exchange)
                                         * dw[d][ip * (nj * (nj + 1) / 2) + jp];
                                }
                        }
                    force[3 * i + d] -= g;
                    force[3 * j + d] += g;
                }
            }
        return force;
    }

    void check_forces() {
        const auto expected = reference_forces();
        // Reuse fixture diagnostic tile storage for 3N output; the API owns none.
        gpu::pair_forces(na, n, atoms, dx, dy, dz, dp, tiles, status, stream);
        std::array<double, 3 * na> actual{};
        int error = -1;
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), tiles, sizeof(actual),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(error),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(error, 0);
        for (int i = 0; i < 3 * na; ++i)
            EXPECT_NEAR(actual[i], expected[i], 2e-9) << "component=" << i;
        for (int d = 0; d < 3; ++d) {
            double sum = 0;
            for (int i = 0; i < na; ++i)
                sum += actual[3 * i + d];
            EXPECT_NEAR(sum, 0.0, 2e-10);
        }
    }

    void SetUp() override {
        const std::array<int, na> zs{1, 6, 7, 8, 1, 6, 7, 8};
        int first = 0;
        for (int i = 0; i < na; ++i) {
            const auto* p = GetParam()
                                ? ATV::semiempirical::rm1_parameter(zs[i])
                                : ATV::semiempirical::am1_parameter(zs[i]);
            ASSERT_NE(p, nullptr);
            cpu::NDDOAtom a;
            a.atomic_number = zs[i];
            a.first_ao = first;
            a.n_orbitals = p->n_orbitals;
            a.last_ao = first + a.n_orbitals - 1;
            a.core_charge = p->core_charge;
            a.params = p;
            system.atoms.push_back(a);
            descriptors[i] = {first,  p->n_orbitals, p->core_charge, p->uss,
                              p->upp, p->gss,        p->gsp,         p->gpp,
                              p->gp2, p->hsp,        p->dd,          p->qq,
                              p->am,  p->ad,         p->aq};
            auto& d = descriptors[i];
            d.atomic_number = p->atomic_number;
            d.zs = p->zs;
            d.zp = p->zp;
            d.betas = p->betas;
            d.betap = p->betap;
            d.alp = p->alp;
            for (int g = 0; g < 4; ++g) {
                d.guess1[g] = p->guess1[g];
                d.guess2[g] = p->guess2[g];
                d.guess3[g] = p->guess3[g];
            }
            first += a.n_orbitals;
        }
        ASSERT_EQ(first, n);
        system.n_orbitals = n;
        density.resize(nn);
        spin.resize(nn);
        hcore.resize(nn);
        packed_p.resize(n * (n + 1) / 2);
        packed_spin.resize(packed_p.size());
        packed_h.resize(packed_p.size());
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                const double p =
                    i == j ? 0.7 + 0.01 * i : 0.06 * std::sin(i + 0.4 * j);
                const double s = p * (0.3 + 0.01 * (i + j));
                const double h =
                    i == j ? -7.0 - i : 0.2 * std::cos(0.3 * i + j);
                density[j * n + i] = density[i * n + j] = p;
                spin[j * n + i] = spin[i * n + j] = s;
                hcore[j * n + i] = hcore[i * n + j] = h;
                packed_p[cpu::packed_index(i, j)] = p;
                packed_spin[cpu::packed_index(i, j)] = s;
                packed_h[cpu::packed_index(i, j)] = h;
            }
        }
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&atoms, sizeof(descriptors)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&data,
                             (4 * nn + 3 * na + np * 100 + 1) * sizeof(double)),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&status, sizeof(int)), cudaSuccess);
        dp = data;
        ds = dp + nn;
        dh = ds + nn;
        df = dh + nn;
        dx = df + nn;
        dy = dx + na;
        dz = dy + na;
        tiles = dz + na;
        energy = tiles + np * 100;
        ASSERT_EQ(cudaMemcpyAsync(atoms, descriptors.data(),
                                  sizeof(descriptors), cudaMemcpyHostToDevice,
                                  stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(dp, density.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(ds, spin.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(dh, hcore.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        upload_geometry();
        scratch_bytes = gpu::electronic_energy_workspace_bytes(n);
        ASSERT_EQ(cudaMalloc(&scratch, scratch_bytes), cudaSuccess);
    }

    void TearDown() override {
        if (stream)
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(cudaFree(scratch), cudaSuccess);
        EXPECT_EQ(cudaFree(status), cudaSuccess);
        EXPECT_EQ(cudaFree(data), cudaSuccess);
        EXPECT_EQ(cudaFree(atoms), cudaSuccess);
        if (stream)
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
};

TEST_P(CudaNDDOTwoCenter,
       AnalyticPairForcesMatchCpuAcrossCHNOAxesAndDistances) {
    const auto x0 = x, y0 = y, z0 = z;
    for (double scale : {0.5, 1.0, 2.0, 8.0}) {
        for (int i = 0; i < na; ++i) {
            x[i] = x0[i] * scale + 0.25;
            y[i] = y0[i] * scale - 0.375;
            z[i] = z0[i] * scale + 0.125;
        }
        upload_geometry();
        check_forces();
    }
    // Both sides of the transverse-axis gauge switch, plus its exact value.
    for (double uz : {0.8 - 1e-7, 0.8, 0.8 + 1e-7}) {
        x = x0;
        y = y0;
        z = z0;
        x[3] = x[1] + 1.3 * std::sqrt(1.0 - uz * uz);
        y[3] = y[1];
        z[3] = z[1] + 1.3 * uz;
        upload_geometry();
        check_forces();
    }
}

TEST_P(CudaNDDOTwoCenter,
       AnalyticPairForcesIsolateCoreAndDensityContributions) {
    const auto saved = density;
    for (int mode = 0; mode < 3; ++mode) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                density[j * n + i] =
                    mode == 0
                        ? 0.0
                        : (mode == 1 && i != j ? 0.0
                                               : -0.37 * saved[j * n + i]);
        ASSERT_EQ(cudaMemcpyAsync(dp, density.data(), nn * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        check_forces();
    }
}

TEST_P(CudaNDDOTwoCenter, AnalyticPairForceFailuresCanRetryWithoutStaleOutput) {
    const auto x0 = x, y0 = y, z0 = z;
    const auto expect_error = [&](int wanted) {
        gpu::pair_forces(na, n, atoms, dx, dy, dz, dp, tiles, status, stream);
        int error = 0;
        EXPECT_EQ(cudaMemcpyAsync(&error, status, sizeof(error),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(error, wanted);
    };
    x[1] = x[0];
    y[1] = y[0];
    z[1] = z[0];
    upload_geometry();
    expect_error(1);
    x = x0;
    y = y0;
    z = z0;
    upload_geometry();
    check_forces();
    x[0] = std::numeric_limits<double>::quiet_NaN();
    upload_geometry();
    expect_error(1);
    x = x0;
    y = y0;
    z = z0;
    upload_geometry();
    check_forces();
    const double saved = density[0];
    density[0] = std::numeric_limits<double>::infinity();
    ASSERT_EQ(cudaMemcpyAsync(dp, density.data(), nn * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    expect_error(1);
    density[0] = saved;
    ASSERT_EQ(cudaMemcpyAsync(dp, density.data(), nn * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    check_forces();
    const int saved_z = descriptors[0].atomic_number;
    descriptors[0].atomic_number = 9;
    ASSERT_EQ(cudaMemcpyAsync(atoms, descriptors.data(), sizeof(descriptors),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    expect_error(2);
    descriptors[0].atomic_number = saved_z;
    ASSERT_EQ(cudaMemcpyAsync(atoms, descriptors.data(), sizeof(descriptors),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    check_forces();
    EXPECT_THROW(
        gpu::pair_forces(na, n, atoms, dx, dy, dz, dp, dp, status, stream),
        std::invalid_argument);
}

TEST_P(CudaNDDOTwoCenter, IntegralTilesMatchCpuAcrossElementsAxesAndDistances) {
    const auto x0 = x, y0 = y, z0 = z;
    for (const double scale : {0.5, 1.0, 2.0, 8.0}) {
        for (int i = 0; i < na; ++i) {
            x[i] = x0[i] * scale + 0.37;
            y[i] = y0[i] * scale - 0.21;
            z[i] = z0[i] * scale + 0.18;
        }
        upload_geometry();
        gpu::two_center_integrals(na, atoms, dx, dy, dz, tiles, status, stream);
        std::vector<double> actual(np * 100);
        int error = -1;
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), tiles,
                                  actual.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(error),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(error, 0);
        for (int i = 1; i < na; ++i) {
            for (int j = 0; j < i; ++j) {
                cpu::NDDOTwoCenterIntegralBlock expected;
                cpu::build_two_center_integral_block(
                    system, hx.data(), hy.data(), hz.data(), i, j, expected);
                const int pair = i * (i - 1) / 2 + j;
                for (int p = 0; p < 10; ++p)
                    for (int q = 0; q < 10; ++q) {
                        const double want =
                            p < expected.n_pairs_i && q < expected.n_pairs_j
                                ? expected.at(p, q)
                                : 0.0;
                        EXPECT_NEAR(actual[pair * 100 + p * 10 + q], want,
                                    3e-12)
                            << "atoms=" << i << ',' << j << " local=" << p
                            << ',' << q << " scale=" << scale;
                    }
            }
        }
    }
}

TEST_P(CudaNDDOTwoCenter, FockAndElectronicEnergyMatchCpuWithMixedSpinDensity) {
    const auto expected = cpu::NDDOFockBuilder{}.add_two_center(
        system, hx.data(), hy.data(), hz.data(), packed_h, packed_p,
        packed_spin);
    const double want_energy =
        cpu::electronic_energy_ev(n, packed_p, packed_h, expected);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(cudaMemcpyAsync(df, dh, nn * sizeof(double),
                                  cudaMemcpyDeviceToDevice, stream),
                  cudaSuccess);
        gpu::two_center_fock(na, n, atoms, dx, dy, dz, dp, ds, df, status,
                             stream);
        gpu::electronic_energy(n, dp, dh, df, energy, scratch, scratch_bytes,
                               stream);
        std::vector<double> actual(nn);
        double actual_energy = 0.0;
        int error = -1;
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), df, nn * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&actual_energy, energy, sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(error, 0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(actual[j * n + i],
                            expected[cpu::packed_index(i, j)], 1e-11)
                    << "i=" << i << " j=" << j;
        EXPECT_NEAR(actual_energy, want_energy, 1e-10);
    }
}

TEST_P(CudaNDDOTwoCenter, InvalidGeometryReportsDeviceErrorAndCanRetry) {
    const double saved_x = x[1];
    x[1] = x[0];
    upload_geometry(); // Pair (1,0) now coincides exactly.
    gpu::two_center_integrals(na, atoms, dx, dy, dz, tiles, status, stream);
    int error = 0;
    std::array<double, 100> invalid_tile{};
    ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(invalid_tile.data(), tiles, sizeof(invalid_tile),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(error, 1);
    for (double value : invalid_tile)
        EXPECT_TRUE(std::isnan(value));

    x[1] = saved_x;
    upload_geometry();
    gpu::two_center_integrals(na, atoms, dx, dy, dz, tiles, status, stream);
    ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(error, 0);
}

TEST_P(CudaNDDOTwoCenter, CompleteHcoreAndRepulsionMatchCpuAcrossAllCHNOPairs) {
    const auto x0 = x, y0 = y, z0 = z;
    for (double scale : {0.5, 1.0, 2.0, 8.0}) {
        for (int i = 0; i < na; ++i) {
            x[i] = x0[i] * scale + 0.37;
            y[i] = y0[i] * scale - 0.21;
            z[i] = z0[i] * scale + 0.18;
        }
        upload_geometry();
        const auto expected = cpu::NDDOHCoreBuilder{}.build(
            system, hx.data(), hy.data(), hz.data());
        gpu::hcore(na, n, atoms, dx, dy, dz, df, energy, status, stream);
        std::vector<double> actual(nn);
        double actual_core = 0;
        int error = -1;
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), df, nn * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&actual_core, energy, sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(error, 0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(actual[j * n + i],
                            expected.packed_hcore[cpu::packed_index(i, j)],
                            3e-10)
                    << "i=" << i << " j=" << j << " scale=" << scale;
        EXPECT_NEAR(actual_core, expected.nuclear_repulsion_ev, 1e-10);
    }
}

TEST_P(CudaNDDOTwoCenter, HcoreRejectsInvalidAndUnsupportedInputsThenRetries) {
    const double saved_x = x[1];
    for (const double invalid_x :
         {x[0], std::numeric_limits<double>::quiet_NaN()}) {
        x[1] = invalid_x;
        upload_geometry();
        gpu::hcore(na, n, atoms, dx, dy, dz, df, energy, status, stream);
        int error = 0;
        ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(error, 1);
    }
    x[1] = saved_x;
    upload_geometry();
    descriptors[0].atomic_number = 9;
    ASSERT_EQ(cudaMemcpyAsync(atoms, descriptors.data(), sizeof(gpu::NDDOAtom),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    // Even the no-pair case must reject an unsupported identity.
    gpu::hcore(1, 1, atoms, dx, dy, dz, df, energy, status, stream);
    int error = 0;
    ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(error, 2);
    descriptors[0].atomic_number = 1;
    ASSERT_EQ(cudaMemcpyAsync(atoms, descriptors.data(), sizeof(gpu::NDDOAtom),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    gpu::hcore(na, n, atoms, dx, dy, dz, df, energy, status, stream);
    double actual_core = 0;
    std::vector<double> actual(nn);
    ASSERT_EQ(cudaMemcpyAsync(&error, status, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&actual_core, energy, sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), df, nn * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(error, 0);
    const auto expected =
        cpu::NDDOHCoreBuilder{}.build(system, hx.data(), hy.data(), hz.data());
    EXPECT_NEAR(actual_core, expected.nuclear_repulsion_ev, 1e-10);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(actual[j * n + i],
                        expected.packed_hcore[cpu::packed_index(i, j)], 3e-10);
}

TEST_P(CudaNDDOTwoCenter, RestrictedSpinScalingAvoidsDensityCopy) {
    for (int i = 0; i < nn; ++i)
        spin[i] = 0.5 * density[i];
    ASSERT_EQ(cudaMemcpyAsync(ds, spin.data(), nn * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(df, dh, nn * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream),
              cudaSuccess);
    gpu::two_center_fock(na, n, atoms, dx, dy, dz, dp, ds, df, status, stream);
    std::vector<double> expected(nn), actual(nn);
    ASSERT_EQ(cudaMemcpyAsync(expected.data(), df, nn * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(df, dh, nn * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream),
              cudaSuccess);
    gpu::two_center_fock(na, n, atoms, dx, dy, dz, dp, dp, df, status, stream,
                         0.5);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), df, nn * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int i = 0; i < nn; ++i)
        EXPECT_NEAR(actual[i], expected[i], 3e-12);
}

INSTANTIATE_TEST_SUITE_P(AM1AndRM1, CudaNDDOTwoCenter, ::testing::Bool());
} // namespace
