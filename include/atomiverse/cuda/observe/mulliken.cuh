/// @file cuda/observe/mulliken.cuh
/// @brief MullikenObs -- header-only Mulliken population observer (async D2H bridge).
///
/// Pure observer: computes and stores per-atom charges.  Does NOT write to
/// any file format — that is the recorder's job.
///
/// Registers on a `cuda::Evaluate` driver alongside a
/// `cuda::scf::SCF`; it runs in the `evaluate::Phase::finalize`
/// pass, after the inner `the concrete method (cuda::scf::RHF) private cycle` has converged the density.
///
/// The converged density matrix is no longer carried on `cuda::Frame`; the
/// observer holds a reference to the `cuda::scf::SCF` and reads the
/// density from its SCF workspace. Construct the evaluator as a stable
/// lvalue and register it borrowed (lvalue `add`), so the reference the
/// observer holds stays valid for the whole run.
///
/// @code{.cpp}
/// cuda::scf::SCF scf{integrals, tol, max_iter};
/// MullikenObs mulliken(integrals, scf);
/// cuda::Evaluate eval;
/// eval.on(evaluate::Phase::evaluate).add(scf);   // lvalue -> borrowed
/// eval.on(evaluate::Phase::finalize).add(mulliken);
/// eval.build();
/// eval.run(frame);
/// auto q = mulliken.charges();            // observer: pure data
/// rec.add_field("charge", q);             // recorder: I/O
/// @endcode
#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/observe/mulliken.h"
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"

#ifdef ATV_HAS_HF
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cuda/scf/scf_evaluator.cuh"
#endif

namespace ATV {
namespace cuda {

#ifdef ATV_HAS_HF

/// @brief Mulliken population observer for the CUDA RHF SCF pipeline.
///
/// Header-only — no .cu needed.  Pure observer (stores charges, no I/O).
/// Shares the pure C++ Mulliken implementation with the CPU path; the only
/// CUDA-specific responsibility is freezing the device density matrix into a
/// device-ring slot on the compute stream (before a subsequent SCF solve in an
/// AIMD loop could overwrite it) and bringing it back to host via the async D2H
/// bridge.
class MullikenObs : public AsyncObserver {
  public:
    /// @brief Construct a Mulliken observer.
    /// @param ints      Non-owning view (S, shells, atoms); field storage must
    ///                  outlive this observer.
    /// @param evaluator SCF evaluator whose converged workspace supplies the
    ///                  density matrix (borrowed -- must outlive this observer).
    /// @param every     Sampling stride in steps (1 = every step).
    MullikenObs(cpu::scf::internal::IntegralView ints,
                const scf::SCF& evaluator, int every = 1)
        : integrals_(ints), evaluator_(&evaluator), every_(every) {}

    ~MullikenObs() override = default;

    std::string name() const override { return "mulliken"; }
    /* role via AsyncObserver / Observer */
    int frequency() const override { return every_; }

    const std::vector<double>& charges() const { return charges_; }
    double energy() const { return energy_val_; }
    int iterations() const { return iterations_; }
    int n_occ() const { return n_occ_; }

  protected:
    void on_build(const BuildContext&) override {
        n_basis_ = evaluator_->workspace().n;
        const std::size_t n2 = static_cast<std::size_t>(n_basis_)
                               * static_cast<std::size_t>(n_basis_);
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_P_[i], n2 * sizeof(double)); // device-ring slot
            cudaMalloc(&d_energy_[i], sizeof(double));
            cudaMallocHost(&h_P_[i], n2 * sizeof(double)); // pinned host slot
            cudaMallocHost(&h_energy_[i], sizeof(double));
            *h_energy_[i] = 0.0;
        }
    }

    bool can_run(const LaunchContext&) const override {
        return evaluator_->workspace().P != nullptr;
    }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of the converged density + energy on the compute stream,
        // before a subsequent (AIMD) SCF solve overwrites the workspace.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& f = ctx.frame;
        const std::size_t n2 = static_cast<std::size_t>(n_basis_)
                               * static_cast<std::size_t>(n_basis_);
        cudaMemcpyAsync(d_P_[slot], evaluator_->workspace().P,
                        n2 * sizeof(double), cudaMemcpyDeviceToDevice, ms);
        if (f.scalars)
            cudaMemcpyAsync(d_energy_[slot], &f.scalars->energy, sizeof(double),
                            cudaMemcpyDeviceToDevice, ms);
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        const std::size_t n2 = static_cast<std::size_t>(n_basis_)
                               * static_cast<std::size_t>(n_basis_);
        cudaMemcpyAsync(h_P_[slot], d_P_[slot], n2 * sizeof(double),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_energy_[slot], d_energy_[slot], sizeof(double),
                        cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const LaunchContext&, int slot) override {
        // Adapt the pinned host slot as a non-owning xtensor view.
        const std::size_t nb = static_cast<std::size_t>(n_basis_);
        auto P_view = xt::adapt(h_P_[slot], nb * nb, xt::no_ownership(),
                                std::vector<std::size_t>{nb, nb});
        const xt::xtensor<double, 2> P = P_view;

        charges_ = cpu::mulliken_charge(integrals_, P);
        energy_val_ = *h_energy_[slot];
        iterations_ = evaluator_->workspace().iteration;
        n_occ_ = static_cast<int>(evaluator_->workspace().n_occ);
    }

    void on_destroy(const DestroyContext&) override {
        for (int i = 0; i < kRingDepth; ++i) {
            if (d_P_[i]) {
                cudaFree(d_P_[i]);
                d_P_[i] = nullptr;
            }
            if (d_energy_[i]) {
                cudaFree(d_energy_[i]);
                d_energy_[i] = nullptr;
            }
            if (h_P_[i]) {
                cudaFreeHost(h_P_[i]);
                h_P_[i] = nullptr;
            }
            if (h_energy_[i]) {
                cudaFreeHost(h_energy_[i]);
                h_energy_[i] = nullptr;
            }
        }
    }

  private:
    cpu::scf::internal::IntegralView
        integrals_; ///< Refs into method-owned fields.
    const scf::SCF* evaluator_ =
        nullptr; ///< Borrowed SCF evaluator (density source).
    int every_;

    std::vector<double> charges_;
    double energy_val_ = 0;
    int iterations_ = 0;
    int n_occ_ = 0;

    int n_basis_ = 0;
    double* d_P_[kRingDepth] = {};      ///< device-ring frozen density matrix
    double* d_energy_[kRingDepth] = {}; ///< device-ring frozen energy [eV]
    double* h_P_[kRingDepth] = {};      ///< pinned host density matrix
    double* h_energy_[kRingDepth] = {}; ///< pinned host energy [eV]
};

#endif // ATV_HAS_HF

} // namespace cuda
} // namespace ATV
