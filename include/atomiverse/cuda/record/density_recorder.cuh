/// @file cuda/recorder/density_recorder.cuh
/// @brief DensityRecorder -- header-only electron density grid recorder (async D2H bridge).
///
/// @see cpu::eval_density_grid (cpu/observe/density.h)
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/cpu/observe/density.h"
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/scf_evaluator.cuh"

namespace ATV {
namespace cuda {

/// @brief Evaluate and store electron density rho(r) on a uniform grid.
///
/// Registers on a `cuda::Evaluate` driver alongside a
/// `cuda::scf::SCF`; it runs in the `evaluate::Phase::finalize`
/// pass, after the inner `the concrete method (cuda::scf::RHF) private cycle` has converged the density.
///
/// The converged density matrix is no longer carried on `cuda::Frame`; the
/// recorder holds a reference to the `cuda::scf::SCF` and reads the
/// density from its SCF workspace. Register the evaluator borrowed (lvalue
/// `add`) so the reference stays valid for the whole run.
///
/// @code{.cpp}
/// cuda::scf::SCF scf{integrals, tol, max_iter};
/// cuda::Evaluate eval;
/// eval.set_observer(scf);                          // lvalue -> borrowed
/// eval.on(evaluate::Phase::finalize).add(
///     DensityRecorder{"density.zarr", scf, 20, 20, 20});
/// @endcode
class DensityRecorder : public AsyncRecorder {
  public:
    /// @brief Construct an electron-density grid recorder.
    /// @param store_path Output store path.
    /// @param evaluator  SCF evaluator whose converged workspace supplies
    ///                    the density matrix (borrowed -- must outlive this).
    /// @param nx,ny,nz   Grid dimensions.
    /// @param padding    Bounding-box padding [Angstrom].
    /// @param every      Sampling stride in steps.
    DensityRecorder(std::string store_path, const scf::SCF& evaluator, int nx,
                    int ny, int nz, double padding = double(2.0), int every = 1)
        : store_path_(std::move(store_path)), evaluator_(&evaluator), nx_(nx),
          ny_(ny), nz_(nz), every_(every), padding_(padding) {}

    ~DensityRecorder() override = default;
    DensityRecorder(DensityRecorder&&) noexcept = default;
    DensityRecorder& operator=(DensityRecorder&&) noexcept = default;

    std::string name() const override { return "density_recorder"; }
    /* role via AsyncRecorder / Recorder */
    int frequency() const override { return every_; }

  protected:
    void on_build(const BuildContext& ctx) override {
        auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& frame = ctx.frame;
        n_atoms_ = frame.n_atoms();
        n_basis_ = evaluator_->workspace().n;

        const auto na = static_cast<size_t>(n_atoms_);

        htype_.resize(na);
        hx_.resize(na);
        hy_.resize(na);
        hz_.resize(na);
        cudaMemcpyAsync(htype_.data(), frame.atoms.type, na * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        // frame.atoms.x/y/z are real_t device buffers -- stage as real_t,
        // then widen to double for cpu::eval_density_grid.
        std::vector<real_t> tx(na), ty(na), tz(na);
        cudaMemcpyAsync(tx.data(), frame.atoms.x, na * sizeof(real_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(ty.data(), frame.atoms.y, na * sizeof(real_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tz.data(), frame.atoms.z, na * sizeof(real_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (size_t i = 0; i < na; ++i) {
            hx_[i] = static_cast<double>(tx[i]);
            hy_[i] = static_cast<double>(ty[i]);
            hz_[i] = static_cast<double>(tz[i]);
        }

        const auto nb = static_cast<size_t>(n_basis_);
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_P_[i], nb * nb * sizeof(double)); // device-ring slot
            cudaMallocHost(&h_P_[i],
                           nb * nb * sizeof(double)); // pinned host slot
        }
    }

    bool can_run(const LaunchContext&) const override {
        return evaluator_->workspace().P != nullptr && n_basis_ > 0;
    }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of the converged density on the compute stream, before a
        // subsequent (AIMD) SCF solve overwrites the workspace.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        const size_t nb = static_cast<size_t>(n_basis_);
        cudaMemcpyAsync(d_P_[slot], evaluator_->workspace().P,
                        nb * nb * sizeof(double), cudaMemcpyDeviceToDevice, ms);
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        const size_t nb = static_cast<size_t>(n_basis_);
        cudaMemcpyAsync(h_P_[slot], d_P_[slot], nb * nb * sizeof(double),
                        cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const LaunchContext& ctx, int slot) override {
        auto grid = cpu::eval_density_grid(htype_, hx_, hy_, hz_, h_P_[slot],
                                           n_basis_, nx_, ny_, nz_, padding_);
        if (grid.empty())
            return;
        // NOTE (cuda-resurrect): the zarr density store was removed CPU-side
        // (cpu/record/zarr.h + cpu::zarr_append_density deleted). The async
        // density pipeline (grid eval + ring buffering) is preserved; the sink
        // is disconnected until a cube-writer equivalent is wired in. Reviving
        // the file sink is a follow-up, not part of the API-mirror scope.
        (void)ctx;
        (void)grid;
    }

    void on_destroy(const DestroyContext&) override {
        for (int i = 0; i < kRingDepth; ++i) {
            if (d_P_[i]) {
                cudaFree(d_P_[i]);
                d_P_[i] = nullptr;
            }
            if (h_P_[i]) {
                cudaFreeHost(h_P_[i]);
                h_P_[i] = nullptr;
            }
        }
    }

  private:
    std::string store_path_;
    const scf::SCF* evaluator_ =
        nullptr; ///< Borrowed SCF evaluator (density source).
    int nx_, ny_, nz_, every_;
    double padding_;

    int n_atoms_ = 0;
    int n_basis_ = 0;

    double* d_P_[kRingDepth] = {}; ///< device-ring frozen density matrix
    double* h_P_[kRingDepth] = {}; ///< pinned host density matrix
    std::vector<int> htype_;
    std::vector<double> hx_, hy_, hz_;
};

} // namespace cuda
} // namespace ATV

#endif // ATV_HAS_HF
