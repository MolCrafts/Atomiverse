/// @file cuda/potentials/dihedral_multi_harmonic.cu
/// @brief CUDA DihedralMultiHarmonic — V = Σₙ Cₙ cosⁿ(φ).

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/dihedral_multi_harmonic.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

struct DihedralMultiHarmonic::State {
    force_t* d_C = nullptr; // n_types * 5 coefficients, flat
    int n_types = 0;
};

DihedralMultiHarmonic::DihedralMultiHarmonic(
    std::vector<std::array<double, 5>> C)
    : C_(std::move(C)), state_(std::make_unique<State>()) {}

DihedralMultiHarmonic::~DihedralMultiHarmonic() = default;
DihedralMultiHarmonic::DihedralMultiHarmonic(DihedralMultiHarmonic&&) noexcept =
    default;
DihedralMultiHarmonic&
DihedralMultiHarmonic::operator=(DihedralMultiHarmonic&&) noexcept = default;

namespace {

constexpr int kDihedralBlock = 128;

// Dihedral kernel has ~30+ double locals (b1/b2/b3 cross products,
// fi/fk/fl decomposition); register pressure approaches Hopper's
// 255-reg limit. Cap at 128 threads/block to give the compiler more
// register budget per thread before spilling
// (cuda-md-review-cleanup-06-launch-config-streams).
__global__ __launch_bounds__(kDihedralBlock, 4) void multi_harmonic_dihedral_k(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, fixed_t* __restrict__ fx,
    fixed_t* __restrict__ fy, fixed_t* __restrict__ fz, ForceFault* fault,
    acc_t* __restrict__ energy, acc_t* __restrict__ stress,
    const int* __restrict__ di, const int* __restrict__ dj,
    const int* __restrict__ dk, const int* __restrict__ dl,
    const int* __restrict__ dt,
    const force_t* __restrict__ C, // [n_types * 5]
    cuda::Box box_arg, int n_dihedrals) {
    const auto box = box_arg.current().as<force_t>();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tail/degenerate lanes contribute zero and participate in CUB.
    acc_t v = 0;
    acc_t my_v[6] = {};
    DeferredForceAdds<12> adds;
    bool active = (idx < n_dihedrals);

    // Predeclare locals used by the active branch — referenced from the
    // virial accumulation block at the bottom of the active section.
    force_t b1x = 0, b1y = 0, b1z = 0;
    force_t b2x = 0, b2y = 0, b2z = 0;
    force_t b3x = 0, b3y = 0, b3z = 0;

    if (active) {
        int i = di[idx], j = dj[idx], k = dk[idx], l = dl[idx];
        int type = dt[idx];
        const force_t* Cn = &C[type * 5];

        // Vectors along the backbone
        b1x = x[j] - x[i];
        b1y = y[j] - y[i];
        b1z = z[j] - z[i];
        b2x = x[k] - x[j];
        b2y = y[k] - y[j];
        b2z = z[k] - z[j];
        b3x = x[l] - x[k];
        b3y = y[l] - y[k];
        b3z = z[l] - z[k];

        // Minimum image via cached inv_l on the by-value Box
        // (cuda-md-review-cleanup-04-box-by-value-pbc).
        box.min_image(b1x, b1y, b1z);
        box.min_image(b2x, b2y, b2z);
        box.min_image(b3x, b3y, b3z);

        // Normal vectors: n1 = b1 × b2, n2 = b2 × b3
        force_t n1x = b1y * b2z - b1z * b2y;
        force_t n1y = b1z * b2x - b1x * b2z;
        force_t n1z = b1x * b2y - b1y * b2x;

        force_t n2x = b2y * b3z - b2z * b3y;
        force_t n2y = b2z * b3x - b2x * b3z;
        force_t n2z = b2x * b3y - b2y * b3x;

        force_t n1_len = sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
        force_t n2_len = sqrt(n2x * n2x + n2y * n2y + n2z * n2z);
        constexpr force_t kVecEpsilon = force_t(1e-10);
        if (n1_len < kVecEpsilon || n2_len < kVecEpsilon) {
            // Degenerate geometry — skip this dihedral's force/energy
            // contribution but stay in the kernel so we participate in
            // the block reductions below with v=my_v=0.
            active = false;
        }

        if (active) {

            force_t n1_inv = force_t(1) / n1_len;
            force_t n2_inv = force_t(1) / n2_len;
            n1x *= n1_inv;
            n1y *= n1_inv;
            n1z *= n1_inv;
            n2x *= n2_inv;
            n2y *= n2_inv;
            n2z *= n2_inv;

            force_t cos_phi = n1x * n2x + n1y * n2y + n1z * n2z;
            cos_phi = fmin(fmax(cos_phi, force_t(-1)), force_t(1));

            // V = Σ C_n cos^n(φ) — straight Horner-form polynomial in cos_phi.
            // dV/d(cos φ) = Σ n·C_n cos^(n-1).
            force_t cp = cos_phi;
            v = acc_t(
                Cn[0]
                + cp * (Cn[1] + cp * (Cn[2] + cp * (Cn[3] + cp * Cn[4]))));
            force_t dv_dcos =
                Cn[1]
                + cp
                      * (force_t(2) * Cn[2]
                         + cp * (force_t(3) * Cn[3] + cp * force_t(4) * Cn[4]));

            // ─── Force decomposition (verbatim port of cpu::DihedralMultiHarmonic;
            //     FD-verified by tests/cpu/md/test_pe_fd_force.cpp) ───────────
            //
            // Define unit normals n1_hat, n2_hat, then
            //   ∂cos_phi/∂n1 = (n2_hat - cos_phi·n1_hat) / |n1|   ≡ fh1
            //   ∂cos_phi/∂n2 = (n1_hat - cos_phi·n2_hat) / |n2|   ≡ fh2
            // and propagate to bond-vector gradients via cross products:
            //   ∂cos/∂b1 = b2 × fh1
            //   ∂cos/∂b3 = fh2 × b2
            //   ∂cos/∂b2 = (fh1 × b1) + (b3 × fh2)
            // Forces on the 4 atoms (i, j, k, l) follow:
            //   F_i = dV/dcos · ∂cos/∂b1
            //   F_l = -dV/dcos · ∂cos/∂b3
            //   F_j = -F_i + dV/dcos · ∂cos/∂b2
            //   F_k = -dV/dcos · ∂cos/∂b2 - F_l
            //
            // Note: this matches LAMMPS dihedral_multi_harmonic.cpp algebraically
            // (LAMMPS pre-multiplies dV/dcos into the working `c`/`s12` and
            // builds aₘₙ scalars; this form keeps dv_dcos as a single outer
            // factor for clarity). The previous CUDA kernel used a wrong
            // textbook decomposition `F_i = a11·n1_hat` that was missing the
            // b2-direction component of ∂cos/∂b1 entirely — the force vector
            // had no projection along the dihedral's central bond, so each
            // dihedral pumped net momentum into the system and stage-1 PE
            // simulations failed to thermalise.
            const force_t n1h_x = n1x, n1h_y = n1y,
                          n1h_z = n1z; // already normalized
            const force_t n2h_x = n2x, n2h_y = n2y, n2h_z = n2z;

            const force_t fh1_x = (n2h_x - cos_phi * n1h_x) * n1_inv;
            const force_t fh1_y = (n2h_y - cos_phi * n1h_y) * n1_inv;
            const force_t fh1_z = (n2h_z - cos_phi * n1h_z) * n1_inv;

            const force_t fh2_x = (n1h_x - cos_phi * n2h_x) * n2_inv;
            const force_t fh2_y = (n1h_y - cos_phi * n2h_y) * n2_inv;
            const force_t fh2_z = (n1h_z - cos_phi * n2h_z) * n2_inv;

            // ∂cos/∂b1 = b2 × fh1
            const force_t db1_x = b2y * fh1_z - b2z * fh1_y;
            const force_t db1_y = b2z * fh1_x - b2x * fh1_z;
            const force_t db1_z = b2x * fh1_y - b2y * fh1_x;

            // ∂cos/∂b3 = fh2 × b2
            const force_t db3_x = fh2_y * b2z - fh2_z * b2y;
            const force_t db3_y = fh2_z * b2x - fh2_x * b2z;
            const force_t db3_z = fh2_x * b2y - fh2_y * b2x;

            // ∂cos/∂b2 = (fh1 × b1) + (b3 × fh2)
            const force_t db2_x =
                (fh1_y * b1z - fh1_z * b1y) + (b3y * fh2_z - b3z * fh2_y);
            const force_t db2_y =
                (fh1_z * b1x - fh1_x * b1z) + (b3z * fh2_x - b3x * fh2_z);
            const force_t db2_z =
                (fh1_x * b1y - fh1_y * b1x) + (b3x * fh2_y - b3y * fh2_x);

            const force_t fix = dv_dcos * db1_x;
            const force_t fiy = dv_dcos * db1_y;
            const force_t fiz = dv_dcos * db1_z;

            const force_t flx = -dv_dcos * db3_x;
            const force_t fly = -dv_dcos * db3_y;
            const force_t flz = -dv_dcos * db3_z;

            const force_t gx = dv_dcos * db2_x;
            const force_t gy = dv_dcos * db2_y;
            const force_t gz = dv_dcos * db2_z;

            const force_t fjx = -fix + gx;
            const force_t fjy = -fiy + gy;
            const force_t fjz = -fiz + gz;

            const force_t fkx = -gx - flx;
            const force_t fky = -gy - fly;
            const force_t fkz = -gz - flz;

            adds.add(0, fx, i, fix, fault);
            adds.add(1, fy, i, fiy, fault);
            adds.add(2, fz, i, fiz, fault);
            adds.add(3, fx, j, fjx, fault);
            adds.add(4, fy, j, fjy, fault);
            adds.add(5, fz, j, fjz, fault);
            adds.add(6, fx, k, fkx, fault);
            adds.add(7, fy, k, fky, fault);
            adds.add(8, fz, k, fkz, fault);
            adds.add(9, fx, l, flx, fault);
            adds.add(10, fy, l, fly, fault);
            adds.add(11, fz, l, flz, fault);

            // Dihedral virial (LAMMPS ev_tally convention) — use translation-
            // invariant form r_i · F_i + r_l · F_l + r_k · F_k with j as origin.
            // r_i = -b1, r_k = b2, r_l = b2 + b3 (relative to j).
            force_t r_lx = b2x + b3x, r_ly = b2y + b3y, r_lz = b2z + b3z;
            my_v[0] = acc_t(-b1x * fix + b2x * fkx + r_lx * flx); // xx
            my_v[1] = acc_t(-b1y * fiy + b2y * fky + r_ly * fly); // yy
            my_v[2] = acc_t(-b1z * fiz + b2z * fkz + r_lz * flz); // zz
            my_v[3] = acc_t(-b1x * fiy + b2x * fky + r_lx * fly); // xy
            my_v[4] = acc_t(-b1x * fiz + b2x * fkz + r_lx * flz); // xz
            my_v[5] = acc_t(-b1y * fiz + b2y * fkz + r_ly * flz); // yz
        } // end inner `if (active)` (post-degeneracy check)
    } // end outer `if (active)` (idx < n_dihedrals)

    // Reduce energy and six virial components in one CUB collective.
    ::cuda::std::array<acc_t, 7> my_vals = {v,       my_v[0], my_v[1], my_v[2],
                                            my_v[3], my_v[4], my_v[5]};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDihedralBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out =
        Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
    if (threadIdx.x == 0) {
        atomicAdd(energy, out[0]);
        if (stress != nullptr) {
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress[c], out[c + 1]);
        }
    }
    adds.check(fault);
}

} // anonymous namespace

void DihedralMultiHarmonic::build(const BuildContext& /*ctx*/) {
    int nt = static_cast<int>(C_.size());
    state_->n_types = nt;
    // Flatten coefficients: [C0_0, C1_0, ..., C4_0, C0_1, ...]
    std::vector<force_t> flat(nt * 5);
    for (int t = 0; t < nt; ++t)
        for (int c = 0; c < 5; ++c)
            flat[t * 5 + c] = C_[t][c];
    cudaMalloc(&state_->d_C, nt * 5 * sizeof(force_t));
    cudaMemcpy(state_->d_C, flat.data(), nt * 5 * sizeof(force_t),
               cudaMemcpyHostToDevice);
}

void DihedralMultiHarmonic::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int nd = f.n_dihedrals();
    if (nd == 0)
        return;

    // multi_harmonic_dihedral_k is __launch_bounds__(kDihedralBlock, 4) (capped at 128
    // threads/block for the 255-register limit). Launching kDefaultBlock (256)
    // exceeds that cap -> cuLaunchKernel returns CUDA_ERROR_INVALID_VALUE, the
    // kernel never runs, and the sticky error later surfaces at an unrelated
    // CUDA check (e.g. the neighbor prune). Launch at 128 to match the bound;
    // grid and the per-warp reduction scratch must use the same block size.
    constexpr int block = kDihedralBlock;
    int grid = grid_size(static_cast<int>(nd), block);
    constexpr int shmem = 0;
    multi_harmonic_dihedral_k<<<grid, block, shmem, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.force_fault, &f.scalars->energy, &f.scalars->stress[0],
        f.dihedrals.atomi, f.dihedrals.atomj, f.dihedrals.atomk,
        f.dihedrals.atoml, f.dihedrals.type, state_->d_C, ctx.runtime.host_box,
        nd);
}

void DihedralMultiHarmonic::destroy(const DestroyContext& /*ctx*/) {
    cudaFree(state_->d_C);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV
