/// @file cuda/semiempirical/nddo_forces.cu
/// @brief Warp-cooperative analytic pair differentiation and direct contraction.
/// The fixed-density contraction follows CPU NDDOSolver::compute_forces.
/// Device scalar physics is shared with the existing value kernels, not CPU.
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_forces.cuh"
#include "nddo_one_electron.cuh"
#include <cub/warp/warp_reduce.cuh>
#include <limits>
#include <stdexcept>

namespace ATV::cuda::semiempirical::internal {
namespace {
using namespace detail;

__global__ void validate_force_inputs(int na, int n, const NDDOAtom* atoms,
                                      const double* x, const double* y,
                                      const double* z, const double* p,
                                      int* status) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n * n && !isfinite(p[k]))
        atomicMax(status, 1);
    if (k >= na)
        return;
    const auto& a = atoms[k];
    if ((a.atomic_number != 1 && a.atomic_number != 6 && a.atomic_number != 7
         && a.atomic_number != 8)
        || a.n_orbitals != (a.atomic_number == 1 ? 1 : 4) || a.first_ao < 0
        || a.first_ao > n - a.n_orbitals)
        atomicMax(status, 2);
    if (!isfinite(x[k]) || !isfinite(y[k]) || !isfinite(z[k]))
        atomicMax(status, 1);
}

// Independent dimension: atom pairs. A block is exactly one warp. Lane 0
// computes radial/STO scalars, lanes cooperatively rotate and contract up to
// 100 ERIs. Bounded shared derivatives are reused; no derivative tile is stored.
// Three CUB reductions produce one pair gradient, then six FP64 atomics enforce
// equal/opposite force contributions. Atom type divergence is warp uniform.
__global__ void force_pairs(int n, const NDDOAtom* atoms, const double* x,
                            const double* y, const double* z, const double* p,
                            double* force, int* status) {
    __shared__ PairDerivative ri[22], axes[9], transform[100], overlap[5], core;
    __shared__ int ai_index, aj_index, invalid;
    using Reduce = cub::WarpReduce<double>;
    __shared__ typename Reduce::TempStorage reduction;
    const int lane = threadIdx.x;
    if (lane == 0) {
        invalid = atomicAdd(status, 0) != 0;
        const long long pair = blockIdx.x;
        int i = static_cast<int>((1.0 + ::sqrt(1.0 + 8.0 * pair)) * 0.5);
        while (static_cast<long long>(i) * (i - 1) / 2 > pair)
            --i;
        while (static_cast<long long>(i + 1) * i / 2 <= pair)
            ++i;
        ai_index = i;
        aj_index =
            static_cast<int>(pair - static_cast<long long>(i) * (i - 1) / 2);
        if (!invalid) {
            // Differentiate with respect to atom i; displacement is Rj-Ri.
            PairDerivative dx(x[aj_index] - x[i]), dy(y[aj_index] - y[i]),
                dz(z[aj_index] - z[i]);
            dx.gradient[0] = dy.gradient[1] = dz.gradient[2] = -1.0;
            const auto r2 = dx * dx + dy * dy + dz * dz;
            const auto r = sqrt(r2);
            invalid = !finite(r) || r.value < 1e-12;
            if (!invalid) {
                for (int k = 0; k < 22; ++k)
                    ri[k] = 0.0;
                pair_axes(dx, dy, dz, r, axes);
                radial_integrals<true>(atoms[i], atoms[aj_index], r, ri);
                sto_overlap(atoms[i], atoms[aj_index], r, r2, overlap);
                core = core_repulsion(atoms[i], atoms[aj_index], r, ri[0]);
                invalid = !finite(core);
                for (int k = 0; k < 22; ++k)
                    invalid |= !finite(ri[k]);
                for (int k = 0; k < 9; ++k)
                    invalid |= !finite(axes[k]);
                for (int k = 0; k < 5; ++k)
                    invalid |= !finite(overlap[k]);
            }
        }
        if (invalid)
            atomicMax(status, 1);
    }
    __syncthreads();
    if (invalid)
        return;
    for (int k = lane; k < 100; k += 32)
        transform[k] = rotation_coefficient(axes, k / 10, k % 10);
    __syncthreads();
    const auto& ai = atoms[ai_index];
    const auto& aj = atoms[aj_index];
    PairDerivative gradient = lane == 0 ? core : PairDerivative(0.0);
    // Full on-center blocks count off-diagonal terms twice, as Tr[P dH].
    for (int side = 0; side < 2; ++side) {
        const auto& a = side == 0 ? ai : aj;
        if (lane < a.n_orbitals * a.n_orbitals) {
            const int mu = lane / a.n_orbitals, nu = lane % a.n_orbitals;
            const double charge = side == 0 ? aj.core_charge : ai.core_charge;
            gradient -= charge * p[(a.first_ao + nu) * n + a.first_ao + mu]
                        * attraction(side, mu, nu, ri, axes);
        }
    }
    if (lane < ai.n_orbitals * aj.n_orbitals) {
        const int mu = lane / aj.n_orbitals, nu = lane % aj.n_orbitals;
        gradient += 2.0 * p[(aj.first_ao + nu) * n + ai.first_ao + mu]
                    * resonance(ai, aj, mu, nu, axes, overlap);
    }
    const int ni = ai.n_orbitals * (ai.n_orbitals + 1) / 2;
    const int nj = aj.n_orbitals * (aj.n_orbitals + 1) / 2;
    for (int k = lane; k < 100; k += 32) {
        const int ip = k / 10, jp = k % 10;
        if (ip >= ni || jp >= nj)
            continue;
        PairDerivative eri(0.0);
        for (int term = 0; term < 34; ++term)
            eri += transform[ip * 10 + pair_row[term]]
                   * transform[jp * 10 + pair_col[term]]
                   * ri[radial_index[term]];
        const int a = ai.first_ao + local_row[ip],
                  b = ai.first_ao + local_col[ip];
        const int c = aj.first_ao + local_row[jp],
                  d = aj.first_ao + local_col[jp];
        const double multiplicity = (a == b ? 1.0 : 2.0) * (c == d ? 1.0 : 2.0);
        // CPU's four-term symmetrized contraction with Pspin=P/2.
        const double exchange =
            0.25 * (p[c * n + a] * p[d * n + b] + p[d * n + a] * p[c * n + b]);
        gradient +=
            multiplicity * (p[b * n + a] * p[d * n + c] - exchange) * eri;
    }
    if (!finite(gradient)) {
        atomicMax(status, 1);
        gradient = 0.0;
    }
    double sum[3];
    for (int d = 0; d < 3; ++d) {
        sum[d] = Reduce(reduction).Sum(gradient.gradient[d]);
        __syncwarp(); // CUB TempStorage is reused only after all lanes finish.
    }
    if (lane == 0) {
        if (!isfinite(sum[0]) || !isfinite(sum[1]) || !isfinite(sum[2])) {
            atomicMax(status, 1);
            return;
        }
        for (int d = 0; d < 3; ++d) {
            atomicAdd(force + 3 * ai_index + d, -sum[d]);
            atomicAdd(force + 3 * aj_index + d, sum[d]);
        }
    }
}

__global__ void validate_force_output(int count, const double* force,
                                      int* status) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count && !isfinite(force[k]))
        atomicMax(status, 1);
}
} // namespace

void pair_forces(int na, int n, const NDDOAtom* atoms, const double* x,
                 const double* y, const double* z, const double* density,
                 double* forces, int* status, cudaStream_t stream) {
    const long long pairs = static_cast<long long>(na) * (na - 1) / 2;
    if (na < 1 || n < na || n > std::numeric_limits<int>::max() / n
        || pairs > std::numeric_limits<int>::max() || !atoms || !x || !y || !z
        || !density || !forces || !status || forces == density || forces == x
        || forces == y || forces == z)
        throw std::invalid_argument(
            "NDDO pair forces require valid separate device storage");
    ATV_CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(int), stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(
        forces, 0, 3 * static_cast<std::size_t>(na) * sizeof(double), stream));
    validate_force_inputs<<<grid_size(n * n), kDefaultBlock, 0, stream>>>(
        na, n, atoms, x, y, z, density, status);
    ATV_CUDA_CHECK(cudaGetLastError());
    if (pairs > 0) {
        force_pairs<<<static_cast<int>(pairs), kWarpSize, 0, stream>>>(
            n, atoms, x, y, z, density, forces, status);
        ATV_CUDA_CHECK(cudaGetLastError());
        validate_force_output<<<grid_size(3 * na), kDefaultBlock, 0, stream>>>(
            3 * na, forces, status);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ATV::cuda::semiempirical::internal
