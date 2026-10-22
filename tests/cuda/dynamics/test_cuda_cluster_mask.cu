/// @file tests/cuda/dynamics/test_cuda_cluster_mask.cu
/// @brief OpenMM CS=32 + real interaction masks (cuda-openmm-md-02-rebuild).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "../../../src/cuda/neighbor/nlist_impl.cuh"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"

using namespace ATV;

namespace {

cpu::Frame make_sparse_lattice(int n_side, double spacing) {
    int n = n_side * n_side * n_side;
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (ix + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (iy + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (iz + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(double(n_side) * spacing);
    return frame;
}

double min_image_r2(double dx, double dy, double dz, double L) {
    auto wrap = [L](double d) {
        d -= L * std::nearbyint(d / L);
        return d;
    };
    dx = wrap(dx);
    dy = wrap(dy);
    dz = wrap(dz);
    return dx * dx + dy * dy + dz * dz;
}

} // namespace

TEST(CudaClusterMask, ClusterSizeIs32) {
    EXPECT_EQ(ATV::cuda::kOpenmmClusterSize, 32);
    EXPECT_EQ(ATV::cuda::kOpenmmMaskWords, 16);
}

TEST(CudaClusterMask, MasksHaveClearedBitsAndCoverCutoff) {
    // Large spacing → sparse: many cluster-pair slots outside search radius.
    constexpr int n_side = 4;
    constexpr double spacing = 4.0;
    constexpr double cutoff = 5.0;
    constexpr double skin = 0.5;
    const double r_search = cutoff + skin;
    const double r_search_sq = r_search * r_search;

    auto frame = make_sparse_lattice(n_side, spacing);
    const int n = n_side * n_side * n_side;
    const double L = double(n_side) * spacing;

    ATV::cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    ATV::cuda::NeighborList nl{cutoff, skin};
    PlanBuilder<ATV::cuda::Component> builder;
    builder.add({0, 0, &nl});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    ATV::cuda::BuildContext bctx{backend.context(), backend.frame()};
    nl.build(bctx);
    backend.step(0);
    backend.sync();

    const ATV::cuda::PairList* pl = backend.context().md.pair_list;
    ASSERT_NE(pl, nullptr);
    ASSERT_NE(pl->cl_row_ptr, nullptr);
    ASSERT_NE(pl->cl_col_idx, nullptr);
    ASSERT_NE(pl->cl_orig_index, nullptr);
    ASSERT_NE(pl->cl_mask, nullptr);
    ASSERT_GT(pl->num_clusters, 0);

    const int nc = pl->num_clusters;
    const int CS = ATV::cuda::kOpenmmClusterSize;
    const int MW = ATV::cuda::kOpenmmMaskWords;

    std::vector<int> row_ptr(nc + 1);
    cudaMemcpy(row_ptr.data(), pl->cl_row_ptr, (nc + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    const int npairs = row_ptr[nc];
    ASSERT_GT(npairs, 0);

    std::vector<int> col(npairs);
    cudaMemcpy(col.data(), pl->cl_col_idx, npairs * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::vector<int> orig(static_cast<std::size_t>(nc) * CS);
    cudaMemcpy(orig.data(), pl->cl_orig_index, orig.size() * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::vector<std::uint64_t> masks(static_cast<std::size_t>(npairs) * MW);
    cudaMemcpy(masks.data(), pl->cl_mask, masks.size() * sizeof(std::uint64_t),
               cudaMemcpyDeviceToHost);

    // Host positions
    std::vector<double> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i) {
        x[i] = frame.span(kAtom_x)[i];
        y[i] = frame.span(kAtom_y)[i];
        z[i] = frame.span(kAtom_z)[i];
    }

    // Some mask bits must be clear on this sparse lattice (not all-ones).
    bool any_zero_bit = false;
    bool any_set_bit = false;
    for (int p = 0; p < npairs; ++p) {
        for (int w = 0; w < MW; ++w) {
            std::uint64_t word = masks[static_cast<std::size_t>(p) * MW + w];
            if (word != ~std::uint64_t{0})
                any_zero_bit = true;
            if (word != 0)
                any_set_bit = true;
        }
    }
    EXPECT_TRUE(any_zero_bit) << "masks look all-ones (0xFF memset path)";
    EXPECT_TRUE(any_set_bit) << "masks all zero";

    // Completeness: every atom pair with r < r_search appears as a set mask bit
    // in some cluster-pair expansion (FULL list → may appear twice).
    auto has_pair = [&](int ai, int aj) -> bool {
        for (int ci = 0; ci < nc; ++ci) {
            int si = -1;
            for (int s = 0; s < CS; ++s) {
                if (orig[ci * CS + s] == ai) {
                    si = s;
                    break;
                }
            }
            if (si < 0)
                continue;
            for (int p = row_ptr[ci]; p < row_ptr[ci + 1]; ++p) {
                int cj = col[p];
                for (int sj = 0; sj < CS; ++sj) {
                    if (orig[cj * CS + sj] != aj)
                        continue;
                    std::uint32_t bits =
                        ATV::cuda::openmm_mask_jbits(masks.data(), p, si);
                    if (bits & (1u << sj))
                        return true;
                }
            }
        }
        return false;
    };

    int missing = 0;
    int checked = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            double r2 = min_image_r2(x[j] - x[i], y[j] - y[i], z[j] - z[i], L);
            if (r2 >= r_search_sq)
                continue;
            ++checked;
            if (!has_pair(i, j))
                ++missing;
        }
    }
    EXPECT_GT(checked, 0);
    EXPECT_EQ(missing, 0) << "missing " << missing << " of " << checked
                          << " pairs under search radius";

    ATV::cuda::DestroyContext dctx{backend.context(), backend.frame()};
    nl.destroy(dctx);
    backend.shutdown();
}

namespace {

// Kernel inputs with a wide cluster whose center is far from its nearest
// neighbor, and a partial final hierarchy group (35 clusters / 32).
struct WideClusterBounds {
    static constexpr int count = 35;
    static constexpr int groups = 2;
    ATV::cuda::md::nlist::DeviceVector<ATV::cuda::force_t> atoms{6 * count};
    ATV::cuda::md::nlist::DeviceVector<ATV::cuda::force_t> large{6 * groups};
    ATV::cuda::Box box{};

    WideClusterBounds() {
        using ATV::cuda::force_t;
        std::vector<force_t> host(6 * count);
        for (int i = 0; i < count; ++i) {
            force_t x = i == 0   ? force_t(0)
                        : i == 1 ? force_t(100.5)
                                 : force_t(100 + 10 * i);
            host[i] = x;
            host[3 * count + i] = i == 0 ? force_t(100) : x;
        }
        std::vector<force_t> coarse(6 * groups);
        for (int group = 0; group < groups; ++group) {
            int begin = group * 32, end = std::min(begin + 32, count);
            for (int axis = 0; axis < 3; ++axis) {
                coarse[axis * groups + group] =
                    *std::min_element(host.begin() + axis * count + begin,
                                      host.begin() + axis * count + end);
                coarse[(axis + 3) * groups + group] =
                    *std::max_element(host.begin() + (axis + 3) * count + begin,
                                      host.begin() + (axis + 3) * count + end);
            }
        }
        cudaMemcpy(atoms.data(), host.data(), host.size() * sizeof(force_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(large.data(), coarse.data(), coarse.size() * sizeof(force_t),
                   cudaMemcpyHostToDevice);
        box.ortho = true;
    }

    ATV::cuda::md::nlist::ClusterBounds bounds() const {
        auto* p = atoms.data();
        return {p,
                p + count,
                p + 2 * count,
                p + 3 * count,
                p + 4 * count,
                p + 5 * count};
    }
};

} // namespace

TEST(CudaClusterMask, LargeBoundsIncludePartialFinalGroup) {
    using namespace ATV::cuda::md::nlist;
    WideClusterBounds input;
    DeviceVector<ATV::cuda::force_t> output(6 * WideClusterBounds::groups);
    launch_large_cluster_bounds(input.bounds(), WideClusterBounds::count,
                                output.data());
    std::vector<ATV::cuda::force_t> actual(output.size());
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), output.data(),
                         actual.size() * sizeof(ATV::cuda::force_t),
                         cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual[0], 0);
    EXPECT_EQ(actual[1], 420);
    EXPECT_EQ(actual[6], 410);
    EXPECT_EQ(actual[7], 440);
    for (int i : {2, 3, 4, 5, 8, 9, 10, 11})
        EXPECT_EQ(actual[i], 0);
}

TEST(CudaClusterMask, CountWideClustersBeyondCenterStencil) {
    using namespace ATV::cuda::md::nlist;
    WideClusterBounds input;
    DeviceVector<int> counts(WideClusterBounds::count);
    launch_count_cluster_pairs<ATV::cuda::force_t>(
        input.bounds(), input.large.data(), WideClusterBounds::count, false,
        ATV::cuda::force_t(1), input.box, counts.data());
    std::vector<int> actual(WideClusterBounds::count);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), counts.data(),
                         actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
    // Each cluster sees itself. The 0.5-Angstrom gap between the first wide
    // cluster and the next point adds both directed CSR entries.
    EXPECT_EQ(actual[0], 2);
    EXPECT_EQ(actual[1], 2);
    for (int i = 2; i < WideClusterBounds::count; ++i)
        EXPECT_EQ(actual[i], 1) << "cluster " << i;
}

TEST(CudaClusterMask, FillWideClustersMatchesExactOffsets) {
    using namespace ATV::cuda::md::nlist;
    WideClusterBounds input;
    std::vector<int> rows(WideClusterBounds::count + 1, 0);
    for (int i = 0; i < WideClusterBounds::count; ++i)
        rows[i + 1] = rows[i] + (i < 2 ? 2 : 1);
    DeviceVector<int> device_rows(rows.size()), columns(rows.back());
    cudaMemcpy(device_rows.data(), rows.data(), rows.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    launch_fill_cluster_pairs<ATV::cuda::force_t>(
        input.bounds(), input.large.data(), WideClusterBounds::count, false,
        ATV::cuda::force_t(1), input.box, device_rows.data(), columns.data());
    std::vector<int> actual(rows.back());
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), columns.data(),
                         actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual[0], 0);
    EXPECT_EQ(actual[1], 1);
    EXPECT_EQ(actual[2], 0);
    EXPECT_EQ(actual[3], 1);
    for (int i = 2; i < WideClusterBounds::count; ++i)
        EXPECT_EQ(actual[rows[i]], i) << "cluster " << i;
}

namespace {

struct ExclusionMaskInput {
    ATV::cuda::md::nlist::DeviceVector<int> rows{2}, columns{1}, index{32};
    ATV::cuda::md::nlist::DeviceVector<int> offsets{5}, partners{4};
    ATV::cuda::md::nlist::DeviceVector<std::uint64_t> masks{16};
    std::vector<std::uint32_t> initial = std::vector<std::uint32_t>(32, 15);

    ExclusionMaskInput() {
        const int host_rows[] = {0, 1}, host_columns[] = {0};
        const int host_offsets[] = {0, 1, 2, 3, 4};
        const int host_partners[] = {3, 2, 1, 0};
        std::vector<int> host_index(32, -1);
        host_index[0] = 2;
        host_index[1] = 0;
        host_index[2] = 3;
        host_index[3] = 1;
        for (int i = 0; i < 4; ++i)
            initial[i] &= ~(1u << i);
        cudaMemcpy(rows.data(), host_rows, sizeof(host_rows),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(columns.data(), host_columns, sizeof(host_columns),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(index.data(), host_index.data(),
                   host_index.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(offsets.data(), host_offsets, sizeof(host_offsets),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(partners.data(), host_partners, sizeof(host_partners),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(masks.data(), initial.data(),
                   initial.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice);
    }
};

} // namespace

TEST(CudaClusterMask, BakeFullExclusionsUsesOriginalIndicesAndClearsPadding) {
    ExclusionMaskInput input;
    EXPECT_TRUE(ATV::cuda::md::nlist::launch_bake_csr_exclusions(
        input.rows.data(), input.columns.data(), input.index.data(), 1,
        input.offsets.data(), input.partners.data(), 0.0, input.masks.data()));
    std::vector<std::uint32_t> actual(32);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), input.masks.data(),
                                      actual.size() * sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual[0], 6u);
    EXPECT_EQ(actual[1], 9u);
    EXPECT_EQ(actual[2], 9u);
    EXPECT_EQ(actual[3], 6u);
    for (int i = 4; i < 32; ++i)
        EXPECT_EQ(actual[i], 0u) << "padded slot " << i;
}

TEST(CudaClusterMask, PartialSpecialScalePreservesGeometricMask) {
    ExclusionMaskInput input;
    EXPECT_FALSE(ATV::cuda::md::nlist::launch_bake_csr_exclusions(
        input.rows.data(), input.columns.data(), input.index.data(), 1,
        input.offsets.data(), input.partners.data(), 0.5, input.masks.data()));
    std::vector<std::uint32_t> actual(32);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), input.masks.data(),
                                      actual.size() * sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, input.initial);
}

TEST(CudaClusterMask, CurrentPeriodicMotionOverwritesMappedPartialBlockFlags) {
    using namespace ATV::cuda::md::nlist;
    using ATV::cuda::force_t;
    constexpr int n = 257;
    DeviceVector<force_t> x(n), ref(n), zero(n);
    std::vector<force_t> host(n, 0), reference(n, 0);
    reference[0] = force_t(9.9);
    host[0] = force_t(0.1);     // Only 0.2 A across the periodic face.
    host[n - 1] = force_t(0.6); // The partial final block must request rebuild.
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(ref.data(), reference.data(), n * sizeof(force_t),
                         cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(zero.data(), 0, n * sizeof(force_t)));
    int *flags = nullptr, *device_flags = nullptr;
    ASSERT_EQ(cudaSuccess,
              cudaHostAlloc(&flags, 2 * sizeof(int), cudaHostAllocMapped));
    ASSERT_EQ(cudaSuccess, cudaHostGetDevicePointer(&device_flags, flags, 0));
    ATV::cuda::Box box{};
    box.ortho = true;
    for (int d = 0; d < 3; ++d) {
        box.cell[4 * d] = 10;
        box.inv_l[d] = 0.1;
        box.periodic[d] = true;
    }
    for (int pass = 0; pass < 2; ++pass) {
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(x.data(), host.data(), n * sizeof(force_t),
                             cudaMemcpyHostToDevice));
        launch_atom_displacement_max<ATV::cuda::force_t>(
            x.data(), zero.data(), zero.data(), ref.data(), zero.data(),
            zero.data(), n, ATV::cuda::force_t(0.25), box, device_flags);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        EXPECT_EQ(flags[0], 0) << "periodic crossing is inside skin/2";
        EXPECT_EQ(flags[1], pass == 0 ? 1 : 0)
            << "current tail block overwrites prior flag";
        host[n - 1] = 0;
    }
    ASSERT_EQ(cudaSuccess, cudaFreeHost(flags));
}

namespace {

/// @brief Cluster-pair CSR, masks and original indices copied to the host.
struct HostPairList {
    int nc = 0;
    std::vector<int> row_ptr, col, orig;
    std::vector<std::uint64_t> masks;

    explicit HostPairList(const ATV::cuda::PairList& pl) : nc(pl.num_clusters) {
        const int CS = ATV::cuda::kOpenmmClusterSize;
        const int MW = ATV::cuda::kOpenmmMaskWords;
        row_ptr.resize(nc + 1);
        cudaMemcpy(row_ptr.data(), pl.cl_row_ptr, (nc + 1) * sizeof(int),
                   cudaMemcpyDeviceToHost);
        col.resize(row_ptr[nc]);
        cudaMemcpy(col.data(), pl.cl_col_idx, col.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        orig.resize(static_cast<std::size_t>(nc) * CS);
        cudaMemcpy(orig.data(), pl.cl_orig_index, orig.size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        masks.resize(col.size() * MW);
        cudaMemcpy(masks.data(), pl.cl_mask,
                   masks.size() * sizeof(std::uint64_t),
                   cudaMemcpyDeviceToHost);
    }

    /// @return Every (i, j) carried by a set mask bit, i < j.
    std::vector<std::pair<int, int>> pairs() const {
        const int CS = ATV::cuda::kOpenmmClusterSize;
        std::vector<std::pair<int, int>> out;
        for (int ci = 0; ci < nc; ++ci)
            for (int p = row_ptr[ci]; p < row_ptr[ci + 1]; ++p)
                for (int si = 0; si < CS; ++si) {
                    const int ai = orig[ci * CS + si];
                    const std::uint32_t bits =
                        ATV::cuda::openmm_mask_jbits(masks.data(), p, si);
                    for (int sj = 0; sj < CS; ++sj) {
                        const int aj = orig[col[p] * CS + sj];
                        if (ai >= 0 && aj >= 0 && ai != aj
                            && (bits & (1u << sj)))
                            out.emplace_back(std::min(ai, aj),
                                             std::max(ai, aj));
                    }
                }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }
};

} // namespace

TEST(CudaClusterMask, SkewCellCandidatesKeepEveryPairAndRejectFarClusters) {
    // Strongly tilted LAMMPS-form cell (upper-triangular H: lattice vectors
    // are the columns a = (L,0,0), b = (xy,0.9L,0), c = (xz,yz,1.1L)),
    // atoms hugging every face, a final
    // partial cluster (509 atoms) and sparse random placement that yields
    // wide clusters. Pairs at the search radius ± 1e-4 A are present.
    // Liquid number density (~0.1 / A^3) so clusters are compact, as in MD.
    constexpr int n = 6337; // not a multiple of 32: partial final cluster
    constexpr double L = 40.0, cutoff = 6.0, skin = 1.0;
    const double rs = cutoff + skin;
    Mat3 cell = Mat3::diagonal(L, 0.9 * L, 1.1 * L);
    cell(0, 1) = 0.45 * L;  // xy
    cell(0, 2) = 0.30 * L;  // xz
    cell(1, 2) = -0.40 * L; // yz
    cpu::Frame frame(n);
    frame.box.set_cell(cell);
    std::uint64_t state = 12345;
    auto uniform = [&state] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return double(state >> 11) * (1.0 / 9007199254740992.0);
    };
    std::vector<std::array<double, 3>> frac(n);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d)
            frac[i][d] = uniform();
        if (i % 7 == 0) { // hug one face (alternating sides and axes)
            const int d = (i / 7) % 3;
            frac[i][d] = (i / 21) % 2 ? 1e-6 : 1.0 - 1e-6;
        }
    }
    // Critical pairs: partner placed at exactly rs ∓ 1e-4 A along x.
    for (int k = 0; k < 8; ++k) {
        const int a = 3 + 2 * k, b = a + 1;
        const double offset = rs + (k % 2 ? 1e-4 : -1e-4);
        frac[b] = frac[a];
        frac[b][0] += offset / L;
        frac[b][0] -= std::floor(frac[b][0]);
    }
    std::vector<std::array<double, 3>> pos(n);
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) {
            double r = 0;
            for (int k = 0; k < 3; ++k)
                r += cell(a, k) * frac[i][k];
            pos[i][a] = r;
            const auto& key = a == 0 ? kAtom_x : (a == 1 ? kAtom_y : kAtom_z);
            frame.span(key)[i] = r;
        }
    for (int i = 0; i < n; ++i) {
        frame.span(kAtom_mass)[i] = 1.0;
        frame.span(kAtom_type)[i] = 0;
        frame.span(kAtom_tag)[i] = i;
    }
    frame.rebuild_tag_index(0, n);

    ATV::cuda::Backend backend;
    backend.init();
    backend.materialize(frame);
    ATV::cuda::NeighborList nl{cutoff, skin};
    PlanBuilder<ATV::cuda::Component> builder;
    builder.add({0, 0, &nl});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);
    ATV::cuda::BuildContext bctx{backend.context(), backend.frame()};
    nl.build(bctx);
    backend.step(0);
    backend.sync();
    const ATV::cuda::PairList* pl = backend.context().md.pair_list;
    ASSERT_NE(pl, nullptr);
    const HostPairList host(*pl);
    const auto found = host.pairs();

    const Mat3 inv = frame.box.inv_cell();
    int missing = 0, expected = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double s[3];
            for (int k = 0; k < 3; ++k) {
                s[k] = 0;
                for (int a = 0; a < 3; ++a)
                    s[k] += inv(k, a) * (pos[j][a] - pos[i][a]);
            }
            // rs is below half of every plane spacing, so the rounded
            // fractional image is the exact minimum image of any pair that
            // lies inside the search radius.
            double best = 0;
            for (int a = 0; a < 3; ++a) {
                double r = 0;
                for (int k = 0; k < 3; ++k)
                    r += cell(a, k) * (s[k] - std::nearbyint(s[k]));
                best += r * r;
            }
            if (best >= rs * rs * (1 - 1e-9))
                continue;
            ++expected;
            if (!std::binary_search(found.begin(), found.end(),
                                    std::make_pair(i, j)))
                ++missing;
        }
    EXPECT_GT(expected, 0);
    EXPECT_EQ(missing, 0) << missing << " of " << expected
                          << " pairs inside the search radius are absent";
    // The bound must actually reject far cluster pairs.
    const long long all = static_cast<long long>(host.nc) * host.nc;
    EXPECT_LT(static_cast<long long>(host.col.size()), all / 2)
        << "candidate tiles " << host.col.size() << " of " << all;
    std::printf(
        "skew cell: %d clusters, %zu candidate tiles of %lld, %d pairs\n",
        host.nc, host.col.size(), all, expected);

    ATV::cuda::DestroyContext dctx{backend.context(), backend.frame()};
    nl.destroy(dctx);
    backend.shutdown();
}
