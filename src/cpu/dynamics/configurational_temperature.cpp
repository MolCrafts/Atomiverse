/// @file src/cpu/dynamics/configurational_temperature.cpp
/// @brief First-order configurational temperature (see the header).
#include "atomiverse/cpu/dynamics/configurational_temperature.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {

void CurvatureAssembly::set_terms(std::vector<CurvatureSource*> sources,
                                  std::vector<const ExternalDrive*> drives) {
    sources_ = std::move(sources);
    drives_ = std::move(drives);
}

ConfigurationalSample CurvatureAssembly::evaluate(const LaunchContext& ctx) {
    if (multi_rank_dd(ctx.runtime))
        throw std::runtime_error(
            "configurational temperature: spatial decomposition is not "
            "supported");
    if (sources_.empty())
        throw std::runtime_error(
            "configurational temperature: the run has no CurvatureSource "
            "(no interaction potential)");
    const Frame& f = ctx.frame;
    const int n = owned_count(ctx);
    const int nt = force_count(ctx);
    n_owned_ = n;
    n_total_ = nt;
    wu_.assign(static_cast<std::size_t>(3) * nt, 0.0);
    wh_.assign(static_cast<std::size_t>(3) * nt, 0.0);
    wg_.assign(static_cast<std::size_t>(3) * nt, 0.0);
    wlap_.assign(static_cast<std::size_t>(nt), 0.0);

    // Interaction force on the owned rows: the frame force minus the drives.
    const AccReal* fcol[3] = {f.span(kAtom_fx).data(), f.span(kAtom_fy).data(),
                              f.span(kAtom_fz).data()};
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n; ++i)
            wu_[d * nt + i] = fcol[d][i];
    for (const ExternalDrive* drive : drives_)
        drive->subtract_drive_force(ctx, n, wu_.data(), wu_.data() + nt,
                                    wu_.data() + 2 * nt);

    const bool ghosts = nt > n;
    if (ghosts) {
        if (ctx.runtime.md.comm == nullptr)
            throw std::runtime_error(
                "configurational temperature: ghost rows without a "
                "communicator");
        ctx.runtime.md.comm->forward_comm(static_cast<ForwardPacker&>(*this));
    }

    CurvatureRows rows;
    rows.n_total = nt;
    rows.ux = wu_.data();
    rows.uy = wu_.data() + nt;
    rows.uz = wu_.data() + 2 * nt;
    rows.hx = wh_.data();
    rows.hy = wh_.data() + nt;
    rows.hz = wh_.data() + 2 * nt;
    rows.lap = wlap_.data();
    rows.gx = wg_.data();
    rows.gy = wg_.data() + nt;
    rows.gz = wg_.data() + 2 * nt;
    for (CurvatureSource* source : sources_)
        source->add_curvature(ctx, rows);
    if (ghosts)
        ctx.runtime.md.comm->reverse_comm(static_cast<ReversePacker&>(*this));

    u_.assign(static_cast<std::size_t>(3) * n, 0.0);
    h_.assign(static_cast<std::size_t>(3) * n, 0.0);
    g_.assign(static_cast<std::size_t>(3) * n, 0.0);
    lap_.assign(wlap_.begin(), wlap_.begin() + n);
    double force_sq = 0.0, laplacian = 0.0;
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n; ++i) {
            u_[d * n + i] = wu_[d * nt + i];
            h_[d * n + i] = wh_[d * nt + i];
            g_[d * n + i] = wg_[d * nt + i];
            force_sq += u_[d * n + i] * u_[d * n + i];
        }
    for (int i = 0; i < n; ++i)
        laplacian += lap_[i];
    last_.force_sq = force_sq;
    last_.laplacian = laplacian;
    last_.temperature =
        force_sq / (ctx.runtime.units.boltzmann_constant() * laplacian);
    return last_;
}

void CurvatureAssembly::temperature_gradient(double kB,
                                             std::vector<double>& out) const {
    const int n = n_owned_;
    out.resize(static_cast<std::size_t>(3) * n);
    const double N = last_.force_sq, D = last_.laplacian;
    const double scale = -1.0 / (kB * D * D);
    for (int k = 0; k < 3 * n; ++k)
        out[k] = scale * (2.0 * h_[k] * D + N * g_[k]);
}

int CurvatureAssembly::pack_forward_comm(int n, const int* list, double* buf,
                                         int /*pbc_flag*/, const int* /*pbc*/) {
    const int nt = n_total_;
    int m = 0;
    for (int k = 0; k < n; ++k) {
        const int i = list[k];
        buf[m++] = wu_[i];
        buf[m++] = wu_[nt + i];
        buf[m++] = wu_[2 * nt + i];
    }
    return m;
}

void CurvatureAssembly::unpack_forward_comm(int n, int first,
                                            const double* buf) {
    const int nt = n_total_;
    int m = 0;
    for (int k = 0; k < n; ++k) {
        const int i = first + k;
        wu_[i] = buf[m++];
        wu_[nt + i] = buf[m++];
        wu_[2 * nt + i] = buf[m++];
    }
}

int CurvatureAssembly::pack_reverse_comm(int n, int first, double* buf) {
    const int nt = n_total_;
    int m = 0;
    for (int k = 0; k < n; ++k) {
        const int i = first + k;
        buf[m++] = wh_[i];
        buf[m++] = wh_[nt + i];
        buf[m++] = wh_[2 * nt + i];
        buf[m++] = wlap_[i];
        buf[m++] = wg_[i];
        buf[m++] = wg_[nt + i];
        buf[m++] = wg_[2 * nt + i];
    }
    return m;
}

void CurvatureAssembly::unpack_reverse_comm(int n, const int* list,
                                            const double* buf) {
    const int nt = n_total_;
    int m = 0;
    for (int k = 0; k < n; ++k) {
        const int i = list[k];
        wh_[i] += buf[m++];
        wh_[nt + i] += buf[m++];
        wh_[2 * nt + i] += buf[m++];
        wlap_[i] += buf[m++];
        wg_[i] += buf[m++];
        wg_[nt + i] += buf[m++];
        wg_[2 * nt + i] += buf[m++];
    }
}

ConfigurationalTemperature::ConfigurationalTemperature(int every)
    : every_(every) {
    if (every < 1)
        throw std::invalid_argument(
            "ConfigurationalTemperature: every must be >= 1");
}

void ConfigurationalTemperature::set_curvature_terms(
    std::vector<CurvatureSource*> sources,
    std::vector<const ExternalDrive*> drives) {
    assembly_.set_terms(std::move(sources), std::move(drives));
}

void ConfigurationalTemperature::setup_output(const LaunchContext& /*ctx*/) {
    last_ = ConfigurationalSample{};
    mean_ = ConfigurationalSample{};
    sum_force_sq_ = 0.0;
    sum_laplacian_ = 0.0;
    samples_ = 0;
}

void ConfigurationalTemperature::launch(const LaunchContext& ctx) {
    last_ = assembly_.evaluate(ctx);
    sum_force_sq_ += last_.force_sq;
    sum_laplacian_ += last_.laplacian;
    ++samples_;
    const double n = static_cast<double>(samples_);
    mean_.force_sq = sum_force_sq_ / n;
    mean_.laplacian = sum_laplacian_ / n;
    mean_.temperature =
        mean_.force_sq
        / (ctx.runtime.units.boltzmann_constant() * mean_.laplacian);
}

} // namespace cpu
} // namespace ATV
