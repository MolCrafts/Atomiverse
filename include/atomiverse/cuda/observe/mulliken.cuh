/// @file cuda/observe/mulliken.cuh
/// @brief MullikenObs -- header-only Mulliken population observer (an
///        asynchronous consumer of the shared capture buffer).
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

#include <cstdint>
#include <string>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/observe/mulliken.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"

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
/// CUDA-specific part is that the converged device density matrix travels
/// as this consumer's payload block of the driver's shared capture buffer,
/// frozen on the compute stream (before a subsequent SCF solve in an AIMD
/// loop could overwrite it), together with the step's energy (slot header).
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
    int frequency() const override { return every_; }
    /// @brief The density pointer is resolved per launch: never graphed.
    bool graph_capturable() const override { return false; }
    /// @brief The energy rides in the slot header; P is the payload.
    CaptureColumns capture_columns() const override { return {}; }
    std::uint64_t payload_bytes(const BuildContext&) const override {
        const auto nb = static_cast<std::uint64_t>(evaluator_->workspace().n);
        return nb * nb * sizeof(double);
    }

    const std::vector<double>& charges() const { return charges_; }
    double energy() const { return energy_val_; }
    int iterations() const { return iterations_; }
    int n_occ() const { return n_occ_; }

  protected:
    void on_build(const BuildContext&) override {
        n_basis_ = evaluator_->workspace().n;
    }

    bool can_run(const LaunchContext&) const override {
        return evaluator_->workspace().P != nullptr;
    }

    const void* payload_source(const LaunchContext&) const override {
        return evaluator_->workspace().P;
    }

    void consume(const CaptureView& v) override {
        const std::size_t nb = static_cast<std::size_t>(n_basis_);
        const auto* P_data = reinterpret_cast<const double*>(v.payload);
        auto P_view = xt::adapt(P_data, nb * nb, xt::no_ownership(),
                                std::vector<std::size_t>{nb, nb});
        const xt::xtensor<double, 2> P = P_view;

        charges_ = cpu::mulliken_charge(integrals_, P);
        energy_val_ = v.scalars->energy;
        iterations_ = evaluator_->workspace().iteration;
        n_occ_ = static_cast<int>(evaluator_->workspace().n_occ);
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
};

#endif // ATV_HAS_HF

} // namespace cuda
} // namespace ATV
