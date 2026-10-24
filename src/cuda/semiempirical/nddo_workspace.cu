/// @file cuda/semiempirical/nddo_workspace.cu
/// @brief Native NDDO physical steps over persistent FP64 device storage.
#include "atomiverse/cuda/semiempirical/internal/nddo_workspace.cuh"

#include <algorithm>
#include <cmath>
#include <cub/device/device_reduce.cuh>
#include <limits>
#include <stdexcept>
#include <string>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_forces.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_hcore.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_two_electron.cuh"
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/rm1_parameters.h"

namespace ATV::cuda::semiempirical::internal {
namespace {
void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("NDDO cuBLAS failure: "
                                 + std::to_string(status));
}
void check_solver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("NDDO cuSOLVER failure: "
                                 + std::to_string(status));
}
int matrix_size(int n) {
    if (n <= 0 || n > std::numeric_limits<int>::max() / n)
        throw std::invalid_argument("NDDO invalid dense matrix extent");
    return n * n;
}
struct DensityDifference {
    const double *current, *next;
    __host__ __device__ double operator()(int i) const {
        const double delta = fabs(next[i] - current[i]);
        return isfinite(delta) ? delta : HUGE_VAL;
    }
};
using DifferenceIterator =
    thrust::transform_iterator<DensityDifference,
                               thrust::counting_iterator<int>>;

__global__ void inspect_identity(int count, const int* types,
                                 const NDDOAtom* atoms, const double* box,
                                 NddoMetrics* metrics) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && types[i] != atoms[i].atomic_number)
        atomicExch(&metrics->identity_changed, 1);
    // A nonzero cell is periodic in the current Frame contract. Molecules only.
    if (i < 9 && (!isfinite(box[i]) || box[i] != 0))
        atomicExch(&metrics->pair_error, 3);
}
__global__ void initial_density(int count, int n, const NDDOAtom* atoms,
                                double* p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto& a = atoms[i];
    // CPU initial_diagonal_density_guess, neutral closed shell (charge term 0).
    const double value = a.core_charge / a.n_orbitals;
    for (int k = 0; k < a.n_orbitals; ++k) {
        const int mu = a.first_ao + k;
        p[static_cast<std::size_t>(mu) * n + mu] = value;
    }
}
__global__ void prepare_metrics(NddoMetrics* m) {
    m->energy = 0;
    m->previous_energy = HUGE_VAL;
    m->energy_delta = HUGE_VAL;
    m->density_delta = HUGE_VAL;
    m->eigen_info = 0;
}
__global__ void energy_metrics(NddoMetrics* m) {
    m->energy += m->nuclear_energy;
    m->energy_delta = fabs(m->energy - m->previous_energy);
    // This update changes no judgement: the first residual is infinity, as CPU.
    m->previous_energy = m->energy;
}
__global__ void publish_result(int count, const double* charges,
                               const NddoMetrics* metrics, const double* forces,
                               cuda::Frame frame, bool accumulate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        frame.atoms.charge[i] = charges[i];
        if (forces) {
            if (accumulate) {
                add_force_owned(frame.atoms.fx, i, forces[3 * i],
                                frame.force_fault);
                add_force_owned(frame.atoms.fy, i, forces[3 * i + 1],
                                frame.force_fault);
                add_force_owned(frame.atoms.fz, i, forces[3 * i + 2],
                                frame.force_fault);
            } else {
                store_force(frame.atoms.fx, i, forces[3 * i],
                            frame.force_fault);
                store_force(frame.atoms.fy, i, forces[3 * i + 1],
                            frame.force_fault);
                store_force(frame.atoms.fz, i, forces[3 * i + 2],
                            frame.force_fault);
            }
        }
    }
    if (i == 0) {
        if (accumulate)
            frame.scalars->energy += metrics->energy;
        else
            frame.scalars->energy = metrics->energy;
    }
}
} // namespace

std::size_t density_residual_workspace_bytes(int n) {
    const int nn = matrix_size(n);
    DifferenceIterator input(thrust::counting_iterator<int>(0),
                             {nullptr, nullptr});
    std::size_t bytes = 0;
    ATV_CUDA_CHECK(cub::DeviceReduce::Max(nullptr, bytes, input,
                                          static_cast<double*>(nullptr), nn));
    return bytes;
}
void density_residual(int n, const double* current, const double* next,
                      double* result, void* scratch, std::size_t bytes,
                      cudaStream_t stream) {
    const int nn = matrix_size(n);
    if (!current || !next || !result || !scratch)
        throw std::invalid_argument(
            "NDDO density residual requires device storage");
    DifferenceIterator input(thrust::counting_iterator<int>(0),
                             {current, next});
    ATV_CUDA_CHECK(
        cub::DeviceReduce::Max(scratch, bytes, input, result, nn, stream));
}

NddoWorkspace::NddoWorkspace(ATV::semiempirical::SemiEmpiricalMethod identity)
    : stages{{{*this, ATV::nddo::Phase::diagonalize},
              {*this, ATV::nddo::Phase::density},
              {*this, ATV::nddo::Phase::fock},
              {*this, ATV::nddo::Phase::check}}},
      criterion(*this), method(identity) {
    result.method = method;
    result.reference = "Native CUDA AM1/RM1 NDDO, shared published parameters";
}
NddoWorkspace::~NddoWorkspace() { release(); }
void NddoWorkspace::release() noexcept {
    if (solver)
        cusolverDnDestroy(solver);
    if (blas)
        cublasDestroy(blas);
    cudaFree(eigen_work);
    cudaFree(matrices);
    cudaFree(atoms);
    cudaFree(eigenvalues);
    cudaFree(charges);
    cudaFree(forces);
    cudaFree(metrics);
    cudaFree(scratch);
    if (host_metrics)
        cudaFreeHost(host_metrics);
    solver = nullptr;
    blas = nullptr;
    eigen_work = nullptr;
    matrices = nullptr;
    atoms = nullptr;
    eigenvalues = nullptr;
    charges = nullptr;
    forces = nullptr;
    metrics = nullptr;
    scratch = nullptr;
    host_metrics = nullptr;
    h = f = c = p = next = accepted_p = nullptr;
    n = n_atoms = occupied = 0;
    accepted = false;
    results_on_host = forces_valid = false;
    result.converged = false;
}

void NddoWorkspace::ensure(const cuda::Frame& frame, cudaStream_t stream,
                           bool trial_evaluation) {
    if (cuda::kPrecisionMode != cuda::PrecisionMode::Double)
        throw std::invalid_argument("CUDA AM1/RM1 requires FP64 Frame storage");
    const int count = frame.n_atoms();
    if (count <= 0 || !frame.atoms.type || !frame.box || !frame.scalars
        || !frame.atoms.charge || !frame.atoms.x || !frame.atoms.y
        || !frame.atoms.z)
        throw std::invalid_argument(
            "CUDA AM1/RM1 requires a nonempty molecular Frame");
    bool changed = count != n_atoms || !matrices;
    if (!changed) {
        ATV_CUDA_CHECK(cudaMemsetAsync(&metrics->identity_changed, 0,
                                       sizeof(int), stream));
        ATV_CUDA_CHECK(
            cudaMemsetAsync(&metrics->pair_error, 0, sizeof(int), stream));
        inspect_identity<<<grid_size(std::max(count, 9)), kDefaultBlock, 0,
                           stream>>>(count, frame.atoms.type, atoms, frame.box,
                                     metrics);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaMemcpyAsync(host_metrics, metrics,
                                       sizeof(NddoMetrics),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_metrics->pair_error)
            throw std::invalid_argument(
                "CUDA AM1/RM1 supports isolated molecules without a cell");
        changed = host_metrics->identity_changed != 0;
    }
    if (changed) {
        if (trial_evaluation && matrices)
            throw std::invalid_argument("CUDA AM1/RM1 trial evaluations cannot "
                                        "change molecular identity");
        // Atom identities are build metadata, never electronic scientific arrays.
        std::vector<int> types(count);
        double box[9];
        ATV_CUDA_CHECK(cudaMemcpyAsync(types.data(), frame.atoms.type,
                                       count * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(box, frame.box, sizeof(box),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        for (double b : box)
            if (!std::isfinite(b) || b != 0)
                throw std::invalid_argument(
                    "CUDA AM1/RM1 supports isolated molecules without a cell");
        int nao = 0, electrons = 0;
        double eisol = 0, heat = 0;
        std::vector<NDDOAtom> descriptors;
        descriptors.reserve(count);
        for (const int z : types) {
            if (z != 1 && z != 6 && z != 7 && z != 8)
                throw std::invalid_argument(
                    "CUDA AM1/RM1 currently supports CHNO only");
            const auto* a =
                method == ATV::semiempirical::SemiEmpiricalMethod::AM1
                    ? ATV::semiempirical::am1_parameter(z)
                    : ATV::semiempirical::rm1_parameter(z);
            if (!a)
                throw std::invalid_argument("CUDA NDDO parameter is missing");
            NDDOAtom d{nao,    a->n_orbitals, a->core_charge, a->uss, a->upp,
                       a->gss, a->gsp,        a->gpp,         a->gp2, a->hsp,
                       a->dd,  a->qq,         a->am,          a->ad,  a->aq};
            d.atomic_number = z;
            d.zs = a->zs;
            d.zp = a->zp;
            d.betas = a->betas;
            d.betap = a->betap;
            d.alp = a->alp;
            for (int g = 0; g < 4; ++g) {
                d.guess1[g] = a->guess1[g];
                d.guess2[g] = a->guess2[g];
                d.guess3[g] = a->guess3[g];
            }
            descriptors.push_back(d);
            if (nao > 46340 - a->n_orbitals)
                throw std::invalid_argument("CUDA NDDO matrix is too large");
            nao += a->n_orbitals;
            electrons += static_cast<int>(a->core_charge);
            eisol += a->eisol;
            heat += method == ATV::semiempirical::SemiEmpiricalMethod::AM1
                        ? ATV::semiempirical::am1_atomic_heat_kcal_mol(z)
                        : ATV::semiempirical::rm1_atomic_heat_kcal_mol(z);
        }
        if (electrons % 2 != 0 || electrons <= 0 || electrons > 2 * nao)
            throw std::invalid_argument(
                "CUDA AM1/RM1 requires a neutral closed shell");
        release();
        try {
            n = nao;
            n_atoms = count;
            occupied = electrons / 2;
            sum_eisol = eisol;
            sum_atomic_heat = heat;
            const std::size_t nn = matrix_size(n);
            ATV_CUDA_CHECK(cudaMalloc(&matrices, 6 * nn * sizeof(double)));
            h = matrices;
            f = h + nn;
            c = f + nn;
            p = c + nn;
            next = p + nn;
            accepted_p = next + nn;
            ATV_CUDA_CHECK(cudaMalloc(&atoms, n_atoms * sizeof(NDDOAtom)));
            ATV_CUDA_CHECK(cudaMalloc(&eigenvalues, n * sizeof(double)));
            ATV_CUDA_CHECK(cudaMalloc(&charges, n_atoms * sizeof(double)));
            ATV_CUDA_CHECK(
                cudaMalloc(&forces, 3 * static_cast<std::size_t>(n_atoms)
                                        * sizeof(double)));
            ATV_CUDA_CHECK(cudaMalloc(&metrics, sizeof(NddoMetrics)));
            ATV_CUDA_CHECK(cudaMallocHost(&host_metrics, sizeof(NddoMetrics)));
            ATV_CUDA_CHECK(
                cudaMemsetAsync(metrics, 0, sizeof(NddoMetrics), stream));
            ATV_CUDA_CHECK(cudaMemcpyAsync(atoms, descriptors.data(),
                                           n_atoms * sizeof(NDDOAtom),
                                           cudaMemcpyHostToDevice, stream));
            check_blas(cublasCreate(&blas));
            check_blas(cublasSetStream(blas, stream));
            check_solver(cusolverDnCreate(&solver));
            check_solver(cusolverDnSetStream(solver, stream));
            check_solver(cusolverDnDsyevd_bufferSize(
                solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, c,
                n, eigenvalues, &eigen_work_size));
            ATV_CUDA_CHECK(cudaMalloc(&eigen_work, std::max(1, eigen_work_size)
                                                       * sizeof(double)));
            scratch_bytes = std::max(electronic_energy_workspace_bytes(n),
                                     density_residual_workspace_bytes(n));
            ATV_CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
            result.mulliken_charges.resize(n_atoms);
            result.forces_ev_a.reserve(3 * static_cast<std::size_t>(n_atoms));
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (...) {
            release();
            throw;
        }
    }
    check_blas(cublasSetStream(blas, stream));
    check_solver(cusolverDnSetStream(solver, stream));
}

void NddoWorkspace::prepare(const cuda::Frame& frame, cudaStream_t stream) {
    if (!matrices)
        throw std::logic_error("NDDO workspace was not built");
    result.converged = false;
    result.iterations = 0;
    result.forces_ev_a.clear();
    results_on_host = forces_valid = false;
    hcore(n_atoms, n, atoms, reinterpret_cast<const double*>(frame.atoms.x),
          reinterpret_cast<const double*>(frame.atoms.y),
          reinterpret_cast<const double*>(frame.atoms.z), h,
          &metrics->nuclear_energy, &metrics->pair_error, stream);
    // Hcore validation precedes Fock, whose primitive resets its own status.
    ATV_CUDA_CHECK(cudaMemcpyAsync(host_metrics, metrics, sizeof(NddoMetrics),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_metrics->pair_error
        || !std::isfinite(host_metrics->nuclear_energy))
        throw std::invalid_argument(
            "CUDA NDDO invalid geometry or core Hamiltonian");
    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(double);
    if (accepted)
        ATV_CUDA_CHECK(cudaMemcpyAsync(p, accepted_p, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    else {
        ATV_CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, stream));
        initial_density<<<grid_size(n_atoms), kDefaultBlock, 0, stream>>>(
            n_atoms, n, atoms, p);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    prepare_metrics<<<1, 1, 0, stream>>>(metrics);
    ATV_CUDA_CHECK(cudaGetLastError());
    fock(frame, stream);
    // Do not enter a library eigen solve with failed pair data.
    ATV_CUDA_CHECK(cudaMemcpyAsync(host_metrics, metrics, sizeof(NddoMetrics),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_metrics->pair_error)
        throw std::runtime_error("CUDA NDDO initial Fock assembly failed");
}
void NddoWorkspace::diagonalize(cudaStream_t stream) {
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(c, f, static_cast<std::size_t>(n) * n * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream));
    check_solver(cusolverDnDsyevd(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, c, n,
        eigenvalues, eigen_work, eigen_work_size, &metrics->eigen_info));
}
void NddoWorkspace::density(cudaStream_t stream) {
    const double two = 2, zero = 0;
    check_blas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, occupied, &two,
                           c, n, c, n, &zero, next, n));
    density_residual(n, p, next, &metrics->density_delta, scratch,
                     scratch_bytes, stream);
    std::swap(p, next);
}
void NddoWorkspace::fock(const cuda::Frame& frame, cudaStream_t stream) {
    one_center_fock(n_atoms, n, atoms, h, p, p, f, stream, 0.5);
    two_center_fock(n_atoms, n, atoms,
                    reinterpret_cast<const double*>(frame.atoms.x),
                    reinterpret_cast<const double*>(frame.atoms.y),
                    reinterpret_cast<const double*>(frame.atoms.z), p, p, f,
                    &metrics->pair_error, stream, 0.5);
}
void NddoWorkspace::check(cudaStream_t stream) {
    electronic_energy(n, p, h, f, &metrics->energy, scratch, scratch_bytes,
                      stream);
    energy_metrics<<<1, 1, 0, stream>>>(metrics);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void NddoWorkspace::commit(cudaStream_t stream) {
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        accepted_p, p, static_cast<std::size_t>(n) * n * sizeof(double),
        cudaMemcpyDeviceToDevice, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    accepted = true;
}
void NddoWorkspace::compute_forces(const cuda::Frame& frame,
                                   cudaStream_t stream) {
    forces_valid = false;
    if (!frame.atoms.fx || !frame.atoms.fy || !frame.atoms.fz
        || !frame.force_fault)
        throw std::invalid_argument(
            "CUDA NDDO forces require Frame force storage");
    pair_forces(n_atoms, n, atoms,
                reinterpret_cast<const double*>(frame.atoms.x),
                reinterpret_cast<const double*>(frame.atoms.y),
                reinterpret_cast<const double*>(frame.atoms.z), p, forces,
                &metrics->pair_error, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(&host_metrics->pair_error,
                                   &metrics->pair_error, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_metrics->pair_error)
        throw std::runtime_error("CUDA NDDO analytic force assembly failed");
    forces_valid = true;
}
void NddoWorkspace::publish(cuda::Frame& frame, cudaStream_t stream,
                            bool accumulate) {
    atomic_charges(n_atoms, n, atoms, p, charges, stream);
    publish_result<<<grid_size(n_atoms), kDefaultBlock, 0, stream>>>(
        n_atoms, charges, metrics, forces_valid ? forces : nullptr, frame,
        accumulate);
    ATV_CUDA_CHECK(cudaGetLastError());
    results_on_host = false;
}
void NddoWorkspace::download(cudaStream_t stream) {
    if (!result.converged || results_on_host)
        return;
    if (forces_valid)
        result.forces_ev_a.resize(3 * static_cast<std::size_t>(n_atoms));
    if (!stream) {
        ATV_CUDA_CHECK(cudaMemcpy(result.mulliken_charges.data(), charges,
                                  n_atoms * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        if (forces_valid)
            ATV_CUDA_CHECK(
                cudaMemcpy(result.forces_ev_a.data(), forces,
                           result.forces_ev_a.size() * sizeof(double),
                           cudaMemcpyDeviceToHost));
        results_on_host = true;
        return;
    }
    ATV_CUDA_CHECK(cudaMemcpyAsync(result.mulliken_charges.data(), charges,
                                   n_atoms * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    if (forces_valid)
        ATV_CUDA_CHECK(
            cudaMemcpyAsync(result.forces_ev_a.data(), forces,
                            result.forces_ev_a.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    results_on_host = true;
}
std::string NddoStep::name() const {
    static const char* names[] = {"nddo_diagonalize", "nddo_density",
                                  "nddo_fock", "nddo_check"};
    return names[static_cast<int>(phase_)];
}
void NddoStep::launch(const cuda::LaunchContext& ctx) {
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    switch (phase_) {
    case ATV::nddo::Phase::diagonalize:
        work_.diagonalize(stream);
        break;
    case ATV::nddo::Phase::density:
        work_.density(stream);
        break;
    case ATV::nddo::Phase::fock:
        work_.fock(ctx.frame, stream);
        break;
    case ATV::nddo::Phase::check:
        work_.check(stream);
        break;
    default:
        throw std::logic_error("Invalid NDDO physical stage");
    }
}
Verdict NddoConverged::verdict(const cuda::LoopState& state) {
    const auto stream = static_cast<cudaStream_t>(state.runtime.stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(work_.host_metrics, work_.metrics,
                                   sizeof(NddoMetrics), cudaMemcpyDeviceToHost,
                                   stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto& m = *work_.host_metrics;
    work_.result.iterations = static_cast<int>(state.iteration) + 1;
    if (state.runtime.loop.metrics) {
        state.runtime.loop.metrics->energy = m.energy;
        state.runtime.loop.metrics->energy_change = m.energy_delta;
        state.runtime.loop.metrics->residual = m.density_delta;
    }
    if (m.pair_error || m.eigen_info || !std::isfinite(m.energy)
        || !std::isfinite(m.density_delta) || std::isnan(m.energy_delta))
        return Verdict::Stop;
    work_.result.total_energy_ev = m.energy;
    work_.result.electronic_energy_hartree = m.energy * units::kEvInHartree;
    work_.result.heat_of_formation_kcal_mol =
        (m.energy - work_.sum_eisol) * ATV::semiempirical::kAm1EvToKcalMol
        + work_.sum_atomic_heat;
    return m.density_delta <= density_tol && m.energy_delta <= energy_tol
               ? Verdict::Converged
               : Verdict::Continue;
}
} // namespace ATV::cuda::semiempirical::internal
