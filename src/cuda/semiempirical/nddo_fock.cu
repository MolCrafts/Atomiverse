/// @file cuda/semiempirical/nddo_fock.cu
/// @brief Native CUDA one-center NDDO contractions and observables.
/// Same s/p integral convention as cpu/semiempirical/nddo_fock.cpp; no
/// scientific arrays are computed on or round-tripped through the host.

#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"

#include <cub/device/device_reduce.cuh>
#include <limits>
#include <stdexcept>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"

namespace ATV::cuda::semiempirical::internal {
namespace {

__device__ double integral(const NDDOAtom& a, int i, int j, int k, int l) {
    if (i == j && k == l) {
        if (i == 0 && k == 0)
            return a.gss;
        if (i == 0 || k == 0)
            return a.gsp;
        return i == k ? a.gpp : a.gp2;
    }
    if ((i == k && j == l) || (i == l && j == k)) {
        if (i == 0 || j == 0)
            return a.hsp;
        return 0.5 * (a.gpp - a.gp2);
    }
    return 0.0;
}

__global__ void hcore_kernel(int n_atoms, int n, const NDDOAtom* atoms,
                             double* hcore) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms)
        return;
    const auto& a = atoms[i];
    for (int mu = 0; mu < a.n_orbitals; ++mu) {
        const auto ao = static_cast<std::size_t>(a.first_ao + mu);
        hcore[ao * n + ao] = mu == 0 ? a.uss : a.upp;
    }
}

__global__ void fock_kernel(int n, const NDDOAtom* atoms, const double* density,
                            const double* spin_density, double* fock,
                            double spin_scale) {
    const auto& a = atoms[blockIdx.x];
    // One warp per atom; 1 or 16 active lanes, each owning one Fock entry.
    const int entry = threadIdx.x;
    if (entry >= a.n_orbitals * a.n_orbitals)
        return;
    const int i = entry % a.n_orbitals;
    const int j = entry / a.n_orbitals;
    double sum = 0.0;
    for (int k = 0; k < a.n_orbitals; ++k) {
        for (int l = 0; l < a.n_orbitals; ++l) {
            const std::size_t kl =
                static_cast<std::size_t>(a.first_ao + l) * n + a.first_ao + k;
            sum += density[kl] * integral(a, i, j, k, l)
                   - spin_scale * spin_density[kl] * integral(a, k, j, l, i);
        }
    }
    fock[static_cast<std::size_t>(a.first_ao + j) * n + a.first_ao + i] += sum;
}

__global__ void charges_kernel(int n_atoms, int n, const NDDOAtom* atoms,
                               const double* density, double* charges) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms)
        return;
    const auto& a = atoms[i];
    double q = a.core_charge;
    for (int mu = 0; mu < a.n_orbitals; ++mu) {
        const auto ao = static_cast<std::size_t>(a.first_ao + mu);
        q -= density[ao * n + ao];
    }
    charges[i] = q;
}

struct EnergyTerm {
    const double* density;
    const double* hcore;
    const double* fock;
    __host__ __device__ double operator()(int i) const {
        return 0.5 * density[i] * (hcore[i] + fock[i]);
    }
};

using EnergyIterator =
    thrust::transform_iterator<EnergyTerm, thrust::counting_iterator<int>>;

// CUB's integer item count bounds the matrix extent, before any multiplication.
int matrix_elements(int n) {
    if (n <= 0 || n > std::numeric_limits<int>::max() / n)
        throw std::invalid_argument("NDDO device matrix extent is invalid");
    return n * n;
}

} // namespace

void one_center_hcore(int n_atoms, int n, const NDDOAtom* atoms, double* hcore,
                      cudaStream_t stream) {
    const int nn = matrix_elements(n);
    if (n_atoms <= 0 || n_atoms > n || !atoms || !hcore)
        throw std::invalid_argument(
            "NDDO one-center Hcore requires valid device storage");
    ATV_CUDA_CHECK(cudaMemsetAsync(
        hcore, 0, static_cast<std::size_t>(nn) * sizeof(double), stream));
    hcore_kernel<<<grid_size(n_atoms), kDefaultBlock, 0, stream>>>(
        n_atoms, n, atoms, hcore);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void one_center_fock(int n_atoms, int n, const NDDOAtom* atoms,
                     const double* hcore, const double* density,
                     const double* spin_density, double* fock,
                     cudaStream_t stream, double spin_scale) {
    const int nn = matrix_elements(n);
    if (n_atoms <= 0 || n_atoms > n || !atoms || !hcore || !density
        || !spin_density || !fock || fock == density || fock == spin_density)
        throw std::invalid_argument(
            "NDDO one-center Fock requires valid non-aliased device storage");
    if (fock != hcore)
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            fock, hcore, static_cast<std::size_t>(nn) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
    fock_kernel<<<n_atoms, kWarpSize, 0, stream>>>(
        n, atoms, density, spin_density, fock, spin_scale);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void atomic_charges(int n_atoms, int n, const NDDOAtom* atoms,
                    const double* density, double* charges,
                    cudaStream_t stream) {
    (void)matrix_elements(n);
    if (n_atoms <= 0 || n_atoms > n || !atoms || !density || !charges)
        throw std::invalid_argument(
            "NDDO populations require valid device storage");
    charges_kernel<<<grid_size(n_atoms), kDefaultBlock, 0, stream>>>(
        n_atoms, n, atoms, density, charges);
    ATV_CUDA_CHECK(cudaGetLastError());
}

std::size_t electronic_energy_workspace_bytes(int n) {
    const int nn = matrix_elements(n);
    EnergyIterator input(thrust::counting_iterator<int>(0),
                         EnergyTerm{nullptr, nullptr, nullptr});
    std::size_t bytes = 0;
    ATV_CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, input,
                                          static_cast<double*>(nullptr), nn));
    return bytes;
}

void electronic_energy(int n, const double* density, const double* hcore,
                       const double* fock, double* energy, void* scratch,
                       std::size_t scratch_bytes, cudaStream_t stream) {
    const int nn = matrix_elements(n);
    if (!density || !hcore || !fock || !energy || !scratch)
        throw std::invalid_argument(
            "NDDO energy requires device storage and reusable CUB scratch");
    EnergyIterator input(thrust::counting_iterator<int>(0),
                         EnergyTerm{density, hcore, fock});
    ATV_CUDA_CHECK(cub::DeviceReduce::Sum(scratch, scratch_bytes, input, energy,
                                          nn, stream));
}

} // namespace ATV::cuda::semiempirical::internal
