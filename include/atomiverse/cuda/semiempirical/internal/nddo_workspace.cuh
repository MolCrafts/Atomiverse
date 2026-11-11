/// @file cuda/semiempirical/internal/nddo_workspace.cuh
/// @brief Method-owned NDDO storage and physical steps. No iteration runner.
#pragma once

#include <array>
#include <cstddef>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>

#include "atomiverse/cuda/component/criterion.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_fock.cuh"
#include "atomiverse/semiempirical/nddo_result.h"

namespace ATV::cuda::semiempirical::internal {

/// Small convergence/error bundle; no electronic arrays cross the SCF boundary.
struct NddoMetrics {
    double energy = 0, previous_energy = 0, energy_delta = 0;
    double density_delta = 0, nuclear_energy = 0;
    int pair_error = 0, eigen_info = 0, identity_changed = 0;
};

std::size_t density_residual_workspace_bytes(int n);
/// Maximum absolute full-matrix density change. Nonfinite input yields infinity.
void density_residual(int n, const double* current, const double* next,
                      double* result, void* scratch, std::size_t bytes,
                      cudaStream_t stream);

struct NddoWorkspace;

class NddoStep final : public cuda::Component {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    NddoStep(NddoWorkspace& work, ATV::nddo::Phase phase)
        : work_(work), phase_(phase) {}
    std::string name() const override;
    void launch(const cuda::LaunchContext& ctx) override;

  private:
    NddoWorkspace& work_;
    ATV::nddo::Phase phase_;
};

class NddoConverged final : public cuda::Criterion {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    explicit NddoConverged(NddoWorkspace& work) : work_(work) {}
    std::string name() const override { return "nddo_converged"; }
    bool contributes_to_convergence() const override { return true; }
    Verdict verdict(const cuda::LoopState& state) override;
    double density_tol = 1e-8, energy_tol = 1e-10;

  private:
    NddoWorkspace& work_;
};

/// Only persistent data, parameter preparation and physical single-step bodies.
/// AM1 / RM1 own their Driver plans, loops, budgets and publication decisions.
struct NddoWorkspace {
    explicit NddoWorkspace(ATV::semiempirical::SemiEmpiricalMethod method);
    ~NddoWorkspace();
    NddoWorkspace(const NddoWorkspace&) = delete;
    NddoWorkspace& operator=(const NddoWorkspace&) = delete;

    void ensure(const cuda::Frame& frame, cudaStream_t stream,
                bool trial_evaluation = false);
    void prepare(const cuda::Frame& frame, cudaStream_t stream);
    void diagonalize(cudaStream_t stream);
    void density(cudaStream_t stream);
    void fock(const cuda::Frame& frame, cudaStream_t stream);
    void check(cudaStream_t stream);
    void compute_forces(const cuda::Frame& frame, cudaStream_t stream);
    void stage() noexcept;
    void commit() noexcept;
    void discard() noexcept { pending = false; }
    void publish(cuda::Frame& frame, cudaStream_t stream, bool accumulate);
    void download(cudaStream_t stream);
    void release() noexcept;

    ATV::semiempirical::NDDOResult result;
    std::array<NddoStep, 4> stages;
    NddoConverged criterion;
    const ATV::semiempirical::SemiEmpiricalMethod method;
    bool accepted = false, results_on_host = false, forces_valid = false;
    bool pending = false;
    int n_atoms = 0, n = 0, occupied = 0;
    double sum_eisol = 0, sum_atomic_heat = 0;
    // Seven dense matrices: H, F, C, P, Pnext, Paccepted, Ppending.
    double* matrices = nullptr;
    double *h = nullptr, *f = nullptr, *c = nullptr, *p = nullptr;
    double *next = nullptr, *accepted_p = nullptr, *eigenvalues = nullptr;
    double* pending_p = nullptr;
    double* charges = nullptr;
    double* forces =
        nullptr; // Interleaved 3*N FP64 force scratch, eV/Angstrom.
    NDDOAtom* atoms = nullptr;
    NddoMetrics *metrics = nullptr, *host_metrics = nullptr;
    void* scratch = nullptr;
    std::size_t scratch_bytes = 0;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    double* eigen_work = nullptr;
    int eigen_work_size = 0;
};
} // namespace ATV::cuda::semiempirical::internal
