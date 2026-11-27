/// @file cpu/longrange/pencil_fft.cpp
/// @brief PencilFft: LAMMPS `fft_3d` on pencils, real-to-complex.
///
/// Ported from LAMMPS `src/KSPACE/fft3d.cpp`, checkout e8c4c278a8.
/// LAMMPS: Copyright (2003) Sandia Corporation. GPL-2.0-only.

#include "atomiverse/cpu/longrange/pencil_fft.h"

#ifdef ATV_HAS_MPI

#include <cmath>
#include <stdexcept>

#include <fftw3.h>

#include "atomiverse/cpu/device/fftw_threads.h"

namespace ATV::longrange {
namespace {

// MPI tags of the six remap plans.
constexpr int kTagBrickToZ = 351;
constexpr int kTagZToY = 352;
constexpr int kTagYToX = 353;
constexpr int kTagXToY = 354;
constexpr int kTagYToZ = 355;
constexpr int kTagZToBrick = 356;

std::size_t volume(const std::array<int, 6>& b) {
    if (b[1] < b[0] || b[3] < b[2] || b[5] < b[4])
        return 0;
    return static_cast<std::size_t>(b[1] - b[0] + 1)
           * static_cast<std::size_t>(b[3] - b[2] + 1)
           * static_cast<std::size_t>(b[5] - b[4] + 1);
}

fftw_plan as_plan(void* p) { return static_cast<fftw_plan>(p); }

} // namespace

std::array<int, 2> pencil_bifactor(int n) {
    if (n < 1)
        throw std::invalid_argument("pencil_bifactor: n must be >= 1");
    const int facmax = static_cast<int>(std::sqrt(static_cast<double>(n)));
    for (int n1 = facmax; n1 > 0; --n1) {
        const int n2 = n / n1;
        if (n1 * n2 == n)
            return {n1, n2};
    }
    return {1, n};
}

PencilFft::PencilFft(MPI_Comm comm, const std::array<int, 3>& mesh,
                     const std::array<int, 6>& brick, int active)
    : K_{mesh[0], mesh[1], mesh[2]} {
    int me = 0, nprocs = 0;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &nprocs);
    if (mesh[0] < 1 || mesh[1] < 1 || mesh[2] < 1)
        throw std::invalid_argument("PencilFft: mesh sizes must be >= 1");
    if (active < 1 || active > nprocs)
        throw std::invalid_argument(
            "PencilFft: active must be in [1, size(comm)]");
    const int K0 = K_[0], K1 = K_[1], K2 = K_[2];
    const int n2c = K2 / 2 + 1;

    // LAMMPS index names: i = z (fast), j = y (mid), k = x (slow). Boxes are
    // {ilo, ihi, jlo, jhi, klo, khi}.
    const std::array<int, 6> empty{0, -1, 0, -1, 0, -1};
    const std::array<int, 6> brick_ijk{brick[4], brick[5], brick[2],
                                       brick[3], brick[0], brick[1]};
    std::array<int, 6> first_real = empty; // z pencils, real space
    std::array<int, 6> first = empty;      // z pencils, half spectrum in z
    std::array<int, 6> second = empty;     // y pencils
    std::array<int, 6> third = empty;      // x pencils
    if (me < active) {
        const auto np = pencil_bifactor(active);
        const int np1 = np[0], np2 = np[1];
        const int ip1 = me % np1;
        const int ip2 = me / np1;
        first_real = {0,
                      K2 - 1,
                      ip1 * K1 / np1,
                      (ip1 + 1) * K1 / np1 - 1,
                      ip2 * K0 / np2,
                      (ip2 + 1) * K0 / np2 - 1};
        first = first_real;
        first[1] = n2c - 1;
        second = {ip1 * n2c / np1, (ip1 + 1) * n2c / np1 - 1, 0, K1 - 1,
                  ip2 * K0 / np2,  (ip2 + 1) * K0 / np2 - 1};
        third = {ip1 * n2c / np1,
                 (ip1 + 1) * n2c / np1 - 1,
                 ip2 * K1 / np2,
                 (ip2 + 1) * K1 / np2 - 1,
                 0,
                 K0 - 1};
    }
    real_size_ = volume(first_real);
    half_size_ = volume(first);
    mid_size_ = volume(second);
    spectrum_size_ = volume(third);
    if (spectrum_size_ > 0)
        spectrum_box_ = {third[2], third[3], third[0], third[1]};

    // Rotate a box to the storage order (b, c, a) / (c, a, b) of a permuted
    // layout, as LAMMPS does when it calls remap_3d_create_plan for mid2.
    auto rot1 = [](const std::array<int, 6>& b) { // (i, j, k) -> (j, k, i)
        return std::array<int, 6>{b[2], b[3], b[4], b[5], b[0], b[1]};
    };
    auto rot2 = [](const std::array<int, 6>& b) { // (i, j, k) -> (k, i, j)
        return std::array<int, 6>{b[4], b[5], b[0], b[1], b[2], b[3]};
    };
    brick_to_z_ = std::make_unique<GridRemap>(comm, brick_ijk, first_real, 1, 0,
                                              kTagBrickToZ);
    // mid1: (i, j, k) storage -> y pencils stored (j, k, i).
    z_to_y_ = std::make_unique<GridRemap>(comm, first, second, 2, 1, kTagZToY);
    // mid2: (j, k, i) storage -> x pencils stored (k, i, j).
    y_to_x_ = std::make_unique<GridRemap>(comm, rot1(second), rot1(third), 2, 1,
                                          kTagYToX);
    // Backward: (k, i, j) storage -> (j, k, i), two permutations.
    x_to_y_ = std::make_unique<GridRemap>(comm, rot2(third), rot2(second), 2, 2,
                                          kTagXToY);
    // (j, k, i) storage -> (i, j, k).
    y_to_z_ = std::make_unique<GridRemap>(comm, rot1(second), rot1(first), 2, 2,
                                          kTagYToZ);
    z_to_brick_ = std::make_unique<GridRemap>(comm, first_real, brick_ijk, 1, 0,
                                              kTagZToBrick);

    if (real_size_ > 0) {
        real_ = fftw_alloc_real(real_size_);
        half_ = reinterpret_cast<std::complex<double>*>(
            fftw_alloc_complex(half_size_));
    }
    if (mid_size_ > 0)
        mid_ = reinterpret_cast<std::complex<double>*>(
            fftw_alloc_complex(mid_size_));
    if (spectrum_size_ > 0)
        spectrum_ = reinterpret_cast<std::complex<double>*>(
            fftw_alloc_complex(spectrum_size_));

    // FFTW_ESTIMATE, as LAMMPS: a measured plan could differ between ranks
    // and runs and round differently.
    cpu::device::ensure_fftw_threads();
    if (real_size_ > 0) {
        const int lines = static_cast<int>(real_size_ / K2);
        auto* half = reinterpret_cast<fftw_complex*>(half_);
        plan_z_r2c_ =
            fftw_plan_many_dft_r2c(1, &K2, lines, real_, nullptr, 1, K2, half,
                                   nullptr, 1, n2c, FFTW_ESTIMATE);
        plan_z_c2r_ =
            fftw_plan_many_dft_c2r(1, &K2, lines, half, nullptr, 1, n2c, real_,
                                   nullptr, 1, K2, FFTW_ESTIMATE);
        if (!plan_z_r2c_ || !plan_z_c2r_)
            throw std::runtime_error("PencilFft: FFTW z plan failed");
    }
    if (mid_size_ > 0) {
        const int lines = static_cast<int>(mid_size_ / K1);
        auto* mid = reinterpret_cast<fftw_complex*>(mid_);
        plan_y_fwd_ =
            fftw_plan_many_dft(1, &K1, lines, mid, nullptr, 1, K1, mid, nullptr,
                               1, K1, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_y_bwd_ =
            fftw_plan_many_dft(1, &K1, lines, mid, nullptr, 1, K1, mid, nullptr,
                               1, K1, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!plan_y_fwd_ || !plan_y_bwd_)
            throw std::runtime_error("PencilFft: FFTW y plan failed");
    }
    if (spectrum_size_ > 0) {
        const int lines = static_cast<int>(spectrum_size_ / K0);
        auto* spec = reinterpret_cast<fftw_complex*>(spectrum_);
        plan_x_fwd_ =
            fftw_plan_many_dft(1, &K0, lines, spec, nullptr, 1, K0, spec,
                               nullptr, 1, K0, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_x_bwd_ =
            fftw_plan_many_dft(1, &K0, lines, spec, nullptr, 1, K0, spec,
                               nullptr, 1, K0, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!plan_x_fwd_ || !plan_x_bwd_)
            throw std::runtime_error("PencilFft: FFTW x plan failed");
    }
}

PencilFft::~PencilFft() {
    for (void* p : {plan_z_r2c_, plan_z_c2r_, plan_y_fwd_, plan_y_bwd_,
                    plan_x_fwd_, plan_x_bwd_})
        if (p)
            fftw_destroy_plan(as_plan(p));
    fftw_free(real_);
    fftw_free(half_);
    fftw_free(mid_);
    fftw_free(spectrum_);
}

void PencilFft::forward(const double* brick) {
    brick_to_z_->perform(brick, real_);
    if (plan_z_r2c_)
        fftw_execute(as_plan(plan_z_r2c_));
    z_to_y_->perform(reinterpret_cast<const double*>(half_),
                     reinterpret_cast<double*>(mid_));
    if (plan_y_fwd_)
        fftw_execute(as_plan(plan_y_fwd_));
    y_to_x_->perform(reinterpret_cast<const double*>(mid_),
                     reinterpret_cast<double*>(spectrum_));
    if (plan_x_fwd_)
        fftw_execute(as_plan(plan_x_fwd_));
}

void PencilFft::backward(double* brick) {
    if (plan_x_bwd_)
        fftw_execute(as_plan(plan_x_bwd_));
    x_to_y_->perform(reinterpret_cast<const double*>(spectrum_),
                     reinterpret_cast<double*>(mid_));
    if (plan_y_bwd_)
        fftw_execute(as_plan(plan_y_bwd_));
    y_to_z_->perform(reinterpret_cast<const double*>(mid_),
                     reinterpret_cast<double*>(half_));
    if (plan_z_c2r_)
        fftw_execute(as_plan(plan_z_c2r_));
    z_to_brick_->perform(real_, brick);
}

std::size_t PencilFft::memory_bytes() const {
    std::size_t b = real_size_ * sizeof(double)
                    + (half_size_ + mid_size_ + spectrum_size_)
                          * sizeof(std::complex<double>);
    for (const auto* r : {brick_to_z_.get(), z_to_y_.get(), y_to_x_.get(),
                          x_to_y_.get(), y_to_z_.get(), z_to_brick_.get()})
        b += r->memory_bytes();
    return b;
}

} // namespace ATV::longrange

#endif // ATV_HAS_MPI
