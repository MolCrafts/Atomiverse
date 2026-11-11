/// @file cuda/semiempirical/nddo_hcore.cu
/// @brief Native device CHNO analytic-STO Hcore and AM1/RM1 core repulsion.
/// Uses the validated n=1/2 diat2 formulas in cpu/semiempirical/nddo_overlap.cpp
/// and the correction model in cpu/semiempirical/nddo_hcore.cpp. Coordinates,
/// matrices and reductions stay on the caller's device stream.
#include "atomiverse/cuda/semiempirical/internal/nddo_hcore.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "nddo_one_electron.cuh"
#include <cmath>
#include <stdexcept>

namespace ATV::cuda::semiempirical::internal {
namespace {

__global__ void validate_atoms(int n_atoms, const NDDOAtom* atoms,
                               const double* x, const double* y,
                               const double* z, int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms)
        return;
    const auto& a = atoms[i];
    if ((a.atomic_number != 1 && a.atomic_number != 6 && a.atomic_number != 7
         && a.atomic_number != 8)
        || a.n_orbitals != (a.atomic_number == 1 ? 1 : 4))
        atomicMax(status, 2);
    if (!isfinite(x[i]) || !isfinite(y[i]) || !isfinite(z[i]))
        atomicMax(status, 1);
}

__global__ void hcore_pairs(int n, const NDDOAtom* atoms, const double* x,
                            const double* y, const double* z, double* matrix,
                            double* core_energy, int* status) {
    __shared__ double radial[22], axes[9], overlap[5];
    __shared__ int ia, ja, invalid;
    // Read the shared error word atomically: other blocks may publish a
    // geometry failure concurrently. Unsupported identities stop every block.
    if (threadIdx.x == 0)
        invalid = atomicAdd(status, 0) == 2;
    __syncthreads();
    if (invalid)
        return;
    // Every lane must finish reading the identity flag before lane 0 reuses
    // the shared word for geometry validation (independent warp scheduling).
    __syncthreads();
    if (threadIdx.x == 0) {
        const long long pair = blockIdx.x;
        int i = static_cast<int>((1.0 + sqrt(1.0 + 8.0 * pair)) * 0.5);
        while (static_cast<long long>(i) * (i - 1) / 2 > pair)
            --i;
        while (static_cast<long long>(i + 1) * i / 2 <= pair)
            ++i;
        ia = i;
        ja = static_cast<int>(pair - static_cast<long long>(i) * (i - 1) / 2);
        const double dx = x[ja] - x[ia], dy = y[ja] - y[ia], dz = z[ja] - z[ia];
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double r = sqrt(r2);
        invalid = !isfinite(r) || r < 1e-12;
        for (int k = 0; k < 22; ++k)
            radial[k] = 0;
        if (!invalid) {
            detail::pair_axes(dx, dy, dz, r, axes);
            detail::radial_integrals<false>(atoms[ia], atoms[ja], r, radial);
            detail::sto_overlap(atoms[ia], atoms[ja], r, r2, overlap);
            const double core =
                detail::core_repulsion(atoms[ia], atoms[ja], r, radial[0]);
            invalid = !isfinite(core);
            for (int k = 0; k < 22; ++k)
                invalid |= !isfinite(radial[k]);
            for (int k = 0; k < 5; ++k)
                invalid |= !isfinite(overlap[k]);
            if (!invalid)
                atomicAdd(core_energy, core);
        }
        if (invalid)
            atomicMax(status, 1);
    }
    __syncthreads();
    if (invalid)
        return;
    const auto& ai = atoms[ia];
    const auto& aj = atoms[ja];
    const int lane = threadIdx.x;
    // Cross-atom resonance block: only this pair owns these matrix entries.
    if (lane < ai.n_orbitals * aj.n_orbitals) {
        const int a = lane / aj.n_orbitals, b = lane % aj.n_orbitals;
        const double h = detail::resonance(ai, aj, a, b, axes, overlap);
        matrix[static_cast<std::size_t>(aj.first_ao + b) * n + ai.first_ao
               + a] = h;
        matrix[static_cast<std::size_t>(ai.first_ao + a) * n + aj.first_ao
               + b] = h;
    }
    // Every pair contributes electron-core attraction to its two atom blocks.
    for (int side = 0; side < 2; ++side) {
        const auto& a = side == 0 ? ai : aj;
        const double charge = side == 0 ? aj.core_charge : ai.core_charge;
        if (lane >= a.n_orbitals * a.n_orbitals)
            continue;
        const int mu = lane / a.n_orbitals, nu = lane % a.n_orbitals;
        const double value = detail::attraction(side, mu, nu, radial, axes);
        atomicAdd(matrix + static_cast<std::size_t>(a.first_ao + nu) * n
                      + a.first_ao + mu,
                  -charge * value);
    }
}
} // namespace

void hcore(int n_atoms, int n, const NDDOAtom* atoms, const double* x,
           const double* y, const double* z, double* matrix,
           double* core_energy, int* status, cudaStream_t stream) {
    if (!x || !y || !z || !core_energy || !status || matrix == core_energy)
        throw std::invalid_argument(
            "NDDO Hcore requires separate device arrays and scalars");
    one_center_hcore(n_atoms, n, atoms, matrix, stream);
    ATV_CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(int), stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(core_energy, 0, sizeof(double), stream));
    validate_atoms<<<grid_size(n_atoms), kDefaultBlock, 0, stream>>>(
        n_atoms, atoms, x, y, z, status);
    ATV_CUDA_CHECK(cudaGetLastError());
    const int pairs =
        static_cast<int>(static_cast<long long>(n_atoms) * (n_atoms - 1) / 2);
    if (pairs > 0) {
        hcore_pairs<<<pairs, kWarpSize, 0, stream>>>(n, atoms, x, y, z, matrix,
                                                     core_energy, status);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ATV::cuda::semiempirical::internal
