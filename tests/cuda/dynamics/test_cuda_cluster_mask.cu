/// @file tests/cuda/dynamics/test_cuda_cluster_mask.cu
/// @brief OpenMM CS=32 + real interaction masks (cuda-openmm-md-02-rebuild).

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

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
    EXPECT_EQ(cuda::kOpenmmClusterSize, 32);
    EXPECT_EQ(cuda::kOpenmmMaskWords, 16);
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

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    cuda::NeighborList nl{cutoff, skin};
    PlanBuilder<cuda::Component> builder;
    builder.add({0, 0, &nl});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    nl.build(bctx);
    backend.step(0);
    backend.sync();

    const cuda::PairList* pl = backend.context().md.pair_list;
    ASSERT_NE(pl, nullptr);
    ASSERT_NE(pl->cl_row_ptr, nullptr);
    ASSERT_NE(pl->cl_col_idx, nullptr);
    ASSERT_NE(pl->cl_orig_index, nullptr);
    ASSERT_NE(pl->cl_mask, nullptr);
    ASSERT_GT(pl->num_clusters, 0);

    const int nc = pl->num_clusters;
    const int CS = cuda::kOpenmmClusterSize;
    const int MW = cuda::kOpenmmMaskWords;

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
                        cuda::openmm_mask_jbits(masks.data(), p, si);
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

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    nl.destroy(dctx);
    backend.shutdown();
}
