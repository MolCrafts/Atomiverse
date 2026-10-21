/// @file cuda/record/density_recorder.cuh
/// @brief DensityRecorder -- frozen density/geometry snapshots written as Cube files.
#pragma once

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/cuda/asynchronous/async_bridge.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#ifdef ATV_HAS_HF

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/density.h"
#include "atomiverse/cpu/record/cube.h"
#include "atomiverse/cuda/frame/transfer.cuh"

#include "atomiverse/cuda/scf/scf_evaluator.cuh"

namespace ATV {
namespace cuda {

/// @brief Store rho(r) snapshots on a uniform grid using the canonical STO-3G AOs.
///
/// The borrowed evaluator supplies its density matrix; its AO count must match
/// the STO-3G basis of the atomic numbers in Frame. The existing scf::SCF source
/// is synthetic, not molecular RHF. This recorder interprets its supplied matrix
/// in that documented AO basis; it does not perform an electronic solve.
///
/// Bind after the density producer. Each snapshot freezes the density matrix,
/// complete physical coordinates, and atomic numbers on the compute stream,
/// before any later geometry/density update. CPU evaluation and Cube conversion
/// happen after that snapshot's D2H finishes, on the asynchronous bridge.
///
/// Files are named `<prefix>.step-<step>.cube`. A trailing `.cube` on the given
/// prefix is removed. The evaluator must outlive this recorder.
class DensityRecorder : public AsyncRecorder {
  public:
    /// @param store_path Output filename prefix (one Cube file per sampled step).
    /// @param evaluator Borrowed density-matrix source, built before the recorder.
    /// @param nx,ny,nz Positive grid sample counts.
    /// @param padding Nonnegative bounding-box padding [Angstrom].
    /// @param every Positive sampling stride [steps].
    DensityRecorder(std::string store_path, const scf::SCF& evaluator, int nx,
                    int ny, int nz, double padding = 2.0, int every = 1)
        : store_path_(std::move(store_path)), evaluator_(&evaluator), nx_(nx),
          ny_(ny), nz_(nz), every_(every), padding_(padding) {
        if (store_path_.empty() || nx <= 0 || ny <= 0 || nz <= 0 || every <= 0
            || !std::isfinite(padding) || padding < 0.0)
            throw std::invalid_argument(
                "DensityRecorder: invalid output/grid parameters");
        std::size_t points = 1;
        for (int n : {nx, ny, nz}) {
            if (points > std::numeric_limits<std::size_t>::max()
                             / static_cast<std::size_t>(n))
                throw std::overflow_error(
                    "DensityRecorder: grid size overflow");
            points *= static_cast<std::size_t>(n);
        }
        if (points > std::numeric_limits<std::size_t>::max() / sizeof(double))
            throw std::overflow_error(
                "DensityRecorder: grid byte size overflow");
    }

    ~DensityRecorder() override = default;
    DensityRecorder(DensityRecorder&&) noexcept = default;
    DensityRecorder& operator=(DensityRecorder&&) noexcept = default;

    std::string name() const override { return "density_recorder"; }
    int frequency() const override { return every_; }

  protected:
    void on_build(const BuildContext& ctx) override {
        const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        n_atoms_ = ctx.frame.n_atoms();
        n_basis_ = evaluator_->workspace().n;
        if (n_atoms_ <= 0)
            return;
        if (n_basis_ <= 0 || evaluator_->workspace().P == nullptr)
            throw std::invalid_argument(
                "DensityRecorder: build the density source first");
        const auto na = static_cast<std::size_t>(n_atoms_);
        const auto nb = static_cast<std::size_t>(n_basis_);
        const auto max_size = std::numeric_limits<std::size_t>::max();
        if (nb > max_size / nb / sizeof(prec_t)
            || nb > max_size / nb / sizeof(double)
            || na > max_size / (3 * sizeof(motion_t)))
            throw std::overflow_error(
                "DensityRecorder: snapshot byte size overflow");
        matrix_size_ = nb * nb;
        geometry_ = std::make_unique<cpu::Frame>(n_atoms_);
        htype_.resize(na);
        hx_.resize(na);
        hy_.resize(na);
        hz_.resize(na);
        density_.assign(matrix_size_, 0.0);
        // Check the STO-3G AO contract before allocating the snapshot ring.
        // Build-time transfer is allowed and uses the canonical Frame bridge.
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        cuda::to_cpu(ctx.frame, *geometry_);
        for (int i = 0; i < n_atoms_; ++i) {
            htype_[i] = geometry_->span(kAtom_type)[i];
            hx_[i] = geometry_->span(kAtom_x)[i];
            hy_[i] = geometry_->span(kAtom_y)[i];
            hz_[i] = geometry_->span(kAtom_z)[i];
        }
        if (cpu::eval_density_grid(htype_, hx_, hy_, hz_, density_.data(),
                                   n_basis_, 1, 1, 1, padding_)
                .empty())
            throw std::invalid_argument(
                "DensityRecorder: atomic numbers/AO count must match the "
                "supported STO-3G basis");
        for (int slot = 0; slot < kRingDepth; ++slot) {
            ATV_CUDA_CHECK(
                cudaMalloc(&d_P_[slot], matrix_size_ * sizeof(prec_t)));
            ATV_CUDA_CHECK(
                cudaMallocHost(&h_P_[slot], matrix_size_ * sizeof(prec_t)));
            ATV_CUDA_CHECK(
                cudaMalloc(&d_pos_[slot], 3 * na * sizeof(motion_t)));
            ATV_CUDA_CHECK(
                cudaMallocHost(&h_pos_[slot], 3 * na * sizeof(motion_t)));
            ATV_CUDA_CHECK(cudaMalloc(&d_type_[slot], na * sizeof(int)));
            ATV_CUDA_CHECK(cudaMallocHost(&h_type_[slot], na * sizeof(int)));
        }
        auto prefix = std::filesystem::path(store_path_);
        if (prefix.extension() == ".cube")
            prefix.replace_extension();
        if (prefix.has_parent_path())
            std::filesystem::create_directories(prefix.parent_path());
        prefix_ = prefix.string();
    }

    bool can_run(const LaunchContext& ctx) const override {
        if (ctx.frame.n_atoms() != n_atoms_
            || evaluator_->workspace().n != n_basis_)
            throw std::runtime_error(
                "DensityRecorder: atom/AO count changed; rebuild the recorder");
        if (n_atoms_ > 0 && evaluator_->workspace().P == nullptr)
            throw std::runtime_error("DensityRecorder: density source was "
                                     "destroyed before its recorder");
        return n_atoms_ > 0 && n_basis_ > 0;
    }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        capture_geometry_(ctx, slot);
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_P_[slot], evaluator_->workspace().P,
                                       matrix_size_ * sizeof(prec_t),
                                       cudaMemcpyDeviceToDevice, stream));
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        const auto stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        const auto na = static_cast<std::size_t>(n_atoms_);
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_P_[slot], d_P_[slot],
                                       matrix_size_ * sizeof(prec_t),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_pos_[slot], d_pos_[slot],
                                       3 * na * sizeof(motion_t),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_type_[slot], d_type_[slot],
                                       na * sizeof(int), cudaMemcpyDeviceToHost,
                                       stream));
    }

    void on_data_ready(const LaunchContext& ctx, int slot) override {
        // Deferred callbacks are serialized; these build-reserved CPU buffers
        // are reused only after the preceding snapshot's writer finishes.
        unpack_geometry_(slot);
        for (std::size_t i = 0; i < matrix_size_; ++i)
            density_[i] = static_cast<double>(h_P_[slot][i]);
        const auto grid =
            cpu::eval_density_grid(htype_, hx_, hy_, hz_, density_.data(),
                                   n_basis_, nx_, ny_, nz_, padding_);
        if (grid.empty())
            throw std::runtime_error(
                "DensityRecorder: frozen geometry/AO count is not supported");
        cpu::recorder::write_cube(
            prefix_ + ".step-" + std::to_string(ctx.step) + ".cube",
            "Electron density from Atomiverse",
            "rho(r) in e / Bohr^3; step " + std::to_string(ctx.step),
            *geometry_, grid);
    }

    void on_destroy(const DestroyContext&) override {
        for (int slot = 0; slot < kRingDepth; ++slot) {
            ATV_CUDA_CHECK(cudaFree(d_P_[slot]));
            ATV_CUDA_CHECK(cudaFreeHost(h_P_[slot]));
            ATV_CUDA_CHECK(cudaFree(d_pos_[slot]));
            ATV_CUDA_CHECK(cudaFreeHost(h_pos_[slot]));
            ATV_CUDA_CHECK(cudaFree(d_type_[slot]));
            ATV_CUDA_CHECK(cudaFreeHost(h_type_[slot]));
            d_P_[slot] = h_P_[slot] = nullptr;
            d_pos_[slot] = h_pos_[slot] = nullptr;
            d_type_[slot] = h_type_[slot] = nullptr;
        }
        geometry_.reset();
        n_atoms_ = n_basis_ = 0;
        matrix_size_ = 0;
    }

  private:
    void capture_geometry_(const LaunchContext& ctx, int slot) {
        const auto n = static_cast<std::size_t>(n_atoms_);
        capture_positions(ctx.frame, d_pos_[slot], d_pos_[slot] + n,
                          d_pos_[slot] + 2 * n, ctx.runtime.stream);
        ATV_CUDA_CHECK(
            cudaMemcpyAsync(d_type_[slot], ctx.frame.atoms.type,
                            n * sizeof(int), cudaMemcpyDeviceToDevice,
                            static_cast<cudaStream_t>(ctx.runtime.stream)));
    }

    void unpack_geometry_(int slot) {
        auto gx = geometry_->span(kAtom_x);
        auto gy = geometry_->span(kAtom_y);
        auto gz = geometry_->span(kAtom_z);
        auto gt = geometry_->span(kAtom_type);
        for (int i = 0; i < n_atoms_; ++i) {
            hx_[i] = static_cast<double>(h_pos_[slot][i]);
            hy_[i] = static_cast<double>(h_pos_[slot][n_atoms_ + i]);
            hz_[i] = static_cast<double>(h_pos_[slot][2 * n_atoms_ + i]);
            htype_[i] = h_type_[slot][i];
            gx[i] = static_cast<StorageReal>(hx_[i]);
            gy[i] = static_cast<StorageReal>(hy_[i]);
            gz[i] = static_cast<StorageReal>(hz_[i]);
            gt[i] = htype_[i];
        }
    }

    std::string store_path_, prefix_;
    const scf::SCF* evaluator_ = nullptr;
    int nx_, ny_, nz_, every_;
    double padding_;
    int n_atoms_ = 0, n_basis_ = 0;
    std::size_t matrix_size_ = 0;
    prec_t* d_P_[kRingDepth] = {};
    prec_t* h_P_[kRingDepth] = {};
    motion_t* d_pos_[kRingDepth] = {};
    motion_t* h_pos_[kRingDepth] = {};
    int* d_type_[kRingDepth] = {};
    int* h_type_[kRingDepth] = {};
    std::unique_ptr<cpu::Frame> geometry_;
    std::vector<int> htype_;
    std::vector<double> hx_, hy_, hz_, density_;
};

} // namespace cuda
} // namespace ATV

#endif // ATV_HAS_HF
