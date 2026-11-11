/// @file cuda/record/density_recorder.cuh
/// @brief DensityRecorder -- header-only electron density grid recorder (an
///        asynchronous consumer of the shared capture buffer).
///
/// @see cpu::eval_density_grid (cpu/observe/density.h)
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/observe/density.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"

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
    int frequency() const override { return every_; }
    /// @brief The density pointer is resolved per launch: never graphed.
    bool graph_capturable() const override { return false; }
    /// @brief Positions of the captured step; P is the payload block.
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = true;
        return c;
    }
    std::uint64_t payload_bytes(const BuildContext&) const override {
        const auto nb = static_cast<std::uint64_t>(evaluator_->workspace().n);
        return nb * nb * sizeof(double);
    }

  protected:
    void on_build(const BuildContext& ctx) override {
        n_basis_ = evaluator_->workspace().n;
        if (!ctx.frame.atoms.atomic_number)
            throw std::invalid_argument(
                std::string("cuda::DensityRecorder: device frame has no "
                            "species column; ")
                + atom_field_how_to("atomic_number"));
    }

    bool can_run(const LaunchContext&) const override {
        return evaluator_->workspace().P != nullptr && n_basis_ > 0;
    }

    const void* payload_source(const LaunchContext&) const override {
        return evaluator_->workspace().P;
    }

    void consume(const CaptureView& v) override {
        const auto na = static_cast<std::size_t>(v.n_atoms);
        species_.assign(v.atomic_number, v.atomic_number + na);
        hx_.assign(v.x, v.x + na);
        hy_.assign(v.y, v.y + na);
        hz_.assign(v.z, v.z + na);
        auto grid = cpu::eval_density_grid(
            species_, hx_, hy_, hz_, reinterpret_cast<const double*>(v.payload),
            n_basis_, nx_, ny_, nz_, padding_);
        // NOTE (cuda-resurrect): the zarr density store was removed CPU-side
        // (cpu/record/zarr.h + cpu::zarr_append_density deleted). The grid is
        // evaluated per captured step; the file sink stays disconnected until
        // a cube-writer equivalent is wired in (out of this file's scope).
        (void)grid;
    }

  private:
    std::string store_path_;
    const scf::SCF* evaluator_ =
        nullptr; ///< Borrowed SCF evaluator (density source).
    int nx_, ny_, nz_, every_;
    double padding_;

    int n_basis_ = 0;
    std::vector<int> species_;
    std::vector<double> hx_, hy_, hz_;
};

} // namespace cuda
} // namespace ATV

#endif // ATV_HAS_HF
