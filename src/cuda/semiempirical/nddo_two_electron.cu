/// @file cuda/semiempirical/nddo_two_electron.cu
/// @brief Independent CUDA implementation of the native NDDO s/p pair model.
/// Scalar multipole formulas follow the validated CPU reppd_sp implementation
/// in src/cpu/semiempirical/nddo_two_electron.cpp. The CUDA scheduling, sparse
/// rotation and direct contraction are backend-local; no CPU science fallback.
#include "atomiverse/cuda/semiempirical/internal/nddo_two_electron.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "nddo_pair.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ATV::cuda::semiempirical::internal {
namespace {
using namespace detail;

__device__ void add_symmetric(double* fock, int n, int i, int j, double v) {
    atomicAdd(fock + static_cast<std::size_t>(j) * n + i, v);
    if (i != j)
        atomicAdd(fock + static_cast<std::size_t>(i) * n + j, v);
}

template <bool Contract>
__global__ void pair_kernel(int n, const NDDOAtom* atoms, const double* x,
                            const double* y, const double* z,
                            const double* density, const double* spin_density,
                            double* output, int* status, double spin_scale) {
    __shared__ double ri[22], axes[9], transform[100];
    __shared__ int atom_i, atom_j, invalid;
    if (threadIdx.x == 0) {
        const long long pair = blockIdx.x;
        int i = static_cast<int>((1.0 + sqrt(1.0 + 8.0 * pair)) * 0.5);
        // Correct rounding at a triangular boundary without storing a pair list.
        while (static_cast<long long>(i) * (i - 1) / 2 > pair)
            --i;
        while (static_cast<long long>(i + 1) * i / 2 <= pair)
            ++i;
        atom_i = i;
        atom_j =
            static_cast<int>(pair - static_cast<long long>(i) * (i - 1) / 2);
        const int j = atom_j;
        const double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        invalid = !isfinite(r) || r < 1e-12;
        for (int k = 0; k < 22; ++k)
            ri[k] = 0.0;
        if (!invalid) {
            detail::pair_axes(dx, dy, dz, r, axes);
            detail::radial_integrals<true>(atoms[i], atoms[j], r, ri);
            for (int k = 0; k < 22; ++k)
                invalid |= !isfinite(ri[k]);
        }
        if (invalid)
            atomicExch(status, 1);
    }
    __syncthreads();
    if (invalid) {
        if constexpr (!Contract)
            for (int entry = threadIdx.x; entry < 100; entry += blockDim.x)
                output[static_cast<std::size_t>(blockIdx.x) * 100 + entry] =
                    nan("");
        return;
    }
    for (int entry = threadIdx.x; entry < 100; entry += blockDim.x)
        transform[entry] = rotation_coefficient(axes, entry / 10, entry % 10);
    __syncthreads();
    const auto& ai = atoms[atom_i];
    const auto& aj = atoms[atom_j];
    const int ni = ai.n_orbitals * (ai.n_orbitals + 1) / 2;
    const int nj = aj.n_orbitals * (aj.n_orbitals + 1) / 2;
    for (int entry = threadIdx.x; entry < 100; entry += blockDim.x) {
        const int p = entry / 10, q = entry % 10;
        double value = 0.0;
        if (p < ni && q < nj)
            for (int k = 0; k < 34; ++k)
                value += transform[p * 10 + pair_row[k]]
                         * transform[q * 10 + pair_col[k]]
                         * ri[radial_index[k]];
        if (!isfinite(value)) {
            atomicExch(status, 1);
            if constexpr (Contract)
                continue;
        }
        if constexpr (!Contract) {
            output[static_cast<std::size_t>(blockIdx.x) * 100 + entry] = value;
        } else {
            // Contract immediately; no global or shared ERI tile is stored.
            if (p >= ni || q >= nj)
                continue;
            const int a = ai.first_ao + local_row[p];
            const int b = ai.first_ao + local_col[p];
            const int c = aj.first_ao + local_row[q];
            const int d = aj.first_ao + local_col[q];
            const double mi = a == b ? 1.0 : 2.0;
            const double mj = c == d ? 1.0 : 2.0;
            const double w = value;
            add_symmetric(output, n, a, b,
                          mj * density[static_cast<std::size_t>(d) * n + c]
                              * w);
            add_symmetric(output, n, c, d,
                          mi * density[static_cast<std::size_t>(b) * n + a]
                              * w);
            const double exchange = -0.25 * mi * mj * w * spin_scale;
            add_symmetric(
                output, n, a, c,
                exchange * spin_density[static_cast<std::size_t>(d) * n + b]);
            add_symmetric(
                output, n, a, d,
                exchange * spin_density[static_cast<std::size_t>(c) * n + b]);
            add_symmetric(
                output, n, b, c,
                exchange * spin_density[static_cast<std::size_t>(d) * n + a]);
            add_symmetric(
                output, n, b, d,
                exchange * spin_density[static_cast<std::size_t>(c) * n + a]);
        }
    }
}

int pair_count(int n_atoms) {
    if (n_atoms < 1)
        throw std::invalid_argument(
            "NDDO pair kernels require at least one atom");
    const long long count = static_cast<long long>(n_atoms) * (n_atoms - 1) / 2;
    if (count > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "NDDO atom-pair extent exceeds the device grid");
    return static_cast<int>(count);
}
} // namespace

void two_center_integrals(int n_atoms, const NDDOAtom* atoms, const double* x,
                          const double* y, const double* z, double* tiles,
                          int* status, cudaStream_t stream) {
    const int count = pair_count(n_atoms);
    if (!atoms || !x || !y || !z || !tiles || !status)
        throw std::invalid_argument(
            "NDDO two-center integrals require device storage");
    ATV_CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(int), stream));
    if (count == 0)
        return;
    pair_kernel<false><<<count, kWarpSize, 0, stream>>>(
        0, atoms, x, y, z, nullptr, nullptr, tiles, status, 1.0);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void two_center_fock(int n_atoms, int n, const NDDOAtom* atoms, const double* x,
                     const double* y, const double* z, const double* density,
                     const double* spin_density, double* fock, int* status,
                     cudaStream_t stream, double spin_scale) {
    const int count = pair_count(n_atoms);
    if (n <= 0 || n > std::numeric_limits<int>::max() / n || n_atoms > n
        || !atoms || !x || !y || !z || !density || !spin_density || !fock
        || !status || density == fock || spin_density == fock)
        throw std::invalid_argument(
            "NDDO pair Fock requires valid non-aliased device storage");
    ATV_CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(int), stream));
    if (count == 0)
        return;
    pair_kernel<true><<<count, kWarpSize, 0, stream>>>(
        n, atoms, x, y, z, density, spin_density, fock, status, spin_scale);
    ATV_CUDA_CHECK(cudaGetLastError());
}
} // namespace ATV::cuda::semiempirical::internal
