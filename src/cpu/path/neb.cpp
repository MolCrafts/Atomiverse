/// @file cpu/path/neb.cpp
/// @brief CI-NEB coupling + band L-BFGS.

#include "atomiverse/cpu/path/neb.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/replica/replica_set.h"

namespace ATV {
namespace cpu {
namespace path {

namespace {

/// The band this component is coupling. Throws rather than returning null: a coupling
/// component registered on anything but a PathDriver has no replicas to couple, and the
/// alternative to throwing is silently computing a tangent from one image and itself.
cpu::ReplicaSet& band_of(const cpu::LaunchContext& ctx, const char* who) {
    cpu::ReplicaSet* b = ctx.runtime.path.replicas;
    if (!b) {
        throw std::runtime_error(
            std::string(who)
            + ": no replica band on the context — this component only works "
              "inside a "
              "PathDriver, which is what publishes ctx.path.replicas");
    }
    return *b;
}

} // namespace

void NebCoupling::build(const cpu::BuildContext& ctx) {
    tangent_.assign(ctx.frame.nlocal() * 3, 0.0);
}

void NebCoupling::launch(const cpu::LaunchContext& ctx) {
    cpu::ReplicaSet& band = band_of(ctx, "NebCoupling");
    const int n_img = band.size();
    const int n = band.frame(0).nlocal();

    // The climbing image is the highest-energy one — but only once the band has had a
    // few iterations to relax. See the ctor doc: climbing from a raw interpolation sends
    // the wrong image uphill in the wrong direction.
    climb_idx_ = -1;
    if (climbing_ && ctx.step >= climb_after_) {
        int hi = 1;
        for (int i = 2; i < n_img - 1; ++i) {
            if (band.frame(i).meta.get(kEnergy)
                > band.frame(hi).meta.get(kEnergy))
                hi = i;
        }
        if (n_img > 2)
            climb_idx_ = hi;
    }

    // Endpoints are the given reactant and product: frozen.
    for (const int e : {0, n_img - 1}) {
        auto& f = band.frame(e);
        AccReal* __restrict sp_fx = f.span(kAtom_fx);
        AccReal* __restrict sp_fy = f.span(kAtom_fy);
        AccReal* __restrict sp_fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            sp_fx[i] = 0.0;
            sp_fy[i] = 0.0;
            sp_fz[i] = 0.0;
        }
    }

    for (int im = 1; im < n_img - 1; ++im) {
        // Three DISTINCT frames this iteration — each gets its own uniquely-named
        // span pointers below. Never share a pointer name across prev/cur/next: doing
        // so would silently turn e.g. `next.x[i] - cur.x[i]` into `p[i] - p[i]` == 0.
        auto& prev = band.frame(im - 1);
        auto& cur = band.frame(im);
        auto& next = band.frame(im + 1);

        // Hoisted once per image iteration (this loop's launch boundary — the frame
        // identity changes every iteration).
        const StorageReal* __restrict prev_x = prev.span(kAtom_x);
        const StorageReal* __restrict prev_y = prev.span(kAtom_y);
        const StorageReal* __restrict prev_z = prev.span(kAtom_z);
        const StorageReal* __restrict cur_x = cur.span(kAtom_x);
        const StorageReal* __restrict cur_y = cur.span(kAtom_y);
        const StorageReal* __restrict cur_z = cur.span(kAtom_z);
        AccReal* __restrict cur_fx = cur.span(kAtom_fx);
        AccReal* __restrict cur_fy = cur.span(kAtom_fy);
        AccReal* __restrict cur_fz = cur.span(kAtom_fz);
        const StorageReal* __restrict next_x = next.span(kAtom_x);
        const StorageReal* __restrict next_y = next.span(kAtom_y);
        const StorageReal* __restrict next_z = next.span(kAtom_z);

        const double v_prev = prev.meta.get(kEnergy);
        const double v_cur = cur.meta.get(kEnergy);
        const double v_next = next.meta.get(kEnergy);

        // ── Improved tangent (Henkelman & Jónsson 2000, JCP 113, 9978) ──────────
        //
        // The obvious tangent — the normalized (R_{i+1} - R_{i-1}) bisector — develops
        // kinks wherever the band is stiff, and the kinks then get amplified by the
        // spring force. The improved estimate simply picks the neighbour that is UPHILL
        // in energy, so the tangent always leans toward the barrier.
        double norm = 0.0;
        if (v_next > v_cur && v_cur > v_prev) {
            for (int i = 0; i < n; ++i) {
                tangent_[i * 3 + 0] = next_x[i] - cur_x[i];
                tangent_[i * 3 + 1] = next_y[i] - cur_y[i];
                tangent_[i * 3 + 2] = next_z[i] - cur_z[i];
            }
        } else if (v_next < v_cur && v_cur < v_prev) {
            for (int i = 0; i < n; ++i) {
                tangent_[i * 3 + 0] = cur_x[i] - prev_x[i];
                tangent_[i * 3 + 1] = cur_y[i] - prev_y[i];
                tangent_[i * 3 + 2] = cur_z[i] - prev_z[i];
            }
        } else {
            // At an extremum of the energy along the band, neither neighbour is
            // unambiguously uphill: blend the two, weighting by the energy differences.
            const double dv_max =
                std::max(std::abs(v_next - v_cur), std::abs(v_prev - v_cur));
            const double dv_min =
                std::min(std::abs(v_next - v_cur), std::abs(v_prev - v_cur));
            const double w_next = (v_next > v_prev) ? dv_max : dv_min;
            const double w_prev = (v_next > v_prev) ? dv_min : dv_max;
            for (int i = 0; i < n; ++i) {
                tangent_[i * 3 + 0] = w_next * (next_x[i] - cur_x[i])
                                      + w_prev * (cur_x[i] - prev_x[i]);
                tangent_[i * 3 + 1] = w_next * (next_y[i] - cur_y[i])
                                      + w_prev * (cur_y[i] - prev_y[i]);
                tangent_[i * 3 + 2] = w_next * (next_z[i] - cur_z[i])
                                      + w_prev * (cur_z[i] - prev_z[i]);
            }
        }
        for (const double t : tangent_)
            norm += t * t;
        norm = std::sqrt(norm);
        if (norm < 1e-12)
            continue; // coincident images — nothing to couple along
        for (double& t : tangent_)
            t /= norm;

        // ── Project the true force ──────────────────────────────────────────────
        double f_dot_tau = 0.0;
        for (int i = 0; i < n; ++i) {
            f_dot_tau += cur_fx[i] * tangent_[i * 3 + 0]
                         + cur_fy[i] * tangent_[i * 3 + 1]
                         + cur_fz[i] * tangent_[i * 3 + 2];
        }

        if (im == climb_idx_) {
            // Climbing image: invert the parallel component, no spring. It climbs the
            // path uphill while relaxing perpendicular to it — converging ONTO the
            // saddle, which is the one point a plain band never reaches.
            for (int i = 0; i < n; ++i) {
                cur_fx[i] -= 2.0 * f_dot_tau * tangent_[i * 3 + 0];
                cur_fy[i] -= 2.0 * f_dot_tau * tangent_[i * 3 + 1];
                cur_fz[i] -= 2.0 * f_dot_tau * tangent_[i * 3 + 2];
            }
            continue;
        }

        // Spring force acts ONLY along the tangent, and the true force ONLY
        // perpendicular to it. This split is the "nudging" the method is named for.
        double d_next = 0.0, d_prev = 0.0;
        for (int i = 0; i < n; ++i) {
            const double ax = next_x[i] - cur_x[i];
            const double ay = next_y[i] - cur_y[i];
            const double az = next_z[i] - cur_z[i];
            d_next += ax * ax + ay * ay + az * az;
            const double bx = cur_x[i] - prev_x[i];
            const double by = cur_y[i] - prev_y[i];
            const double bz = cur_z[i] - prev_z[i];
            d_prev += bx * bx + by * by + bz * bz;
        }
        const double f_spring = k_ * (std::sqrt(d_next) - std::sqrt(d_prev));

        for (int i = 0; i < n; ++i) {
            cur_fx[i] += (f_spring - f_dot_tau) * tangent_[i * 3 + 0];
            cur_fy[i] += (f_spring - f_dot_tau) * tangent_[i * 3 + 1];
            cur_fz[i] += (f_spring - f_dot_tau) * tangent_[i * 3 + 2];
        }
    }
}

void BandLBFGS::build(const cpu::BuildContext& ctx) {
    // The band's dimension is not knowable from one Frame, so the buffers are sized on
    // the first launch, when ctx.path.replicas exists. build() only clears state.
    (void)ctx;
    hist_.reset();
    x_.clear();
    g_.clear();
    dir_.clear();
    image_max_force_.clear();
    max_force_ = 0.0;
}

void BandLBFGS::launch(const cpu::LaunchContext& ctx) {
    cpu::ReplicaSet& band = band_of(ctx, "BandLBFGS");
    const int n_img = band.size();
    const int n = band.frame(0).nlocal();
    const int dim = n_img * n * 3;

    if (!hist_) {
        // The band vector is complete on every rank in this v1 (the band is replicated,
        // and the spatial decomposition lives INSIDE each image), so the reduction is
        // the identity. A future one-partition-per-replica layout would pass a
        // replica-communicator all-reduce here — and would need one, which does not
        // exist anywhere in the tree today. @see PathDriver
        hist_ = std::make_unique<cpu::minimizer::LbfgsHistory>(
            memory_, dim, [](double v) { return v; });
        x_.assign(dim, 0.0);
        g_.assign(dim, 0.0);
        dir_.assign(dim, 0.0);
        image_max_force_.assign(n_img, 0.0);
    }

    if (ctx.at(ATV::path::Phase::convergence_check)) {
        // Judged on the forces evaluated this iteration — i.e. on the geometry as it was
        // BEFORE search_direction moved it. Endpoints are frozen and so are converged by
        // construction; if they were counted by force magnitude they would be converged
        // anyway (their force is zeroed), but saying it explicitly is cheaper than
        // making a reader prove it.
        band.set_converged(0, true);
        band.set_converged(n_img - 1, true);
        for (int im = 1; im < n_img - 1; ++im)
            band.set_converged(im, image_max_force_[im] < ftol_);
        return;
    }

    // ── search_direction ───────────────────────────────────────────────────────
    max_force_ = 0.0;
    for (int im = 0; im < n_img; ++im) {
        auto& f = band.frame(im);
        // Hoisted once per image iteration (the frame identity changes every im).
        const StorageReal* __restrict f_x = f.span(kAtom_x);
        const StorageReal* __restrict f_y = f.span(kAtom_y);
        const StorageReal* __restrict f_z = f.span(kAtom_z);
        const AccReal* __restrict f_fx = f.span(kAtom_fx);
        const AccReal* __restrict f_fy = f.span(kAtom_fy);
        const AccReal* __restrict f_fz = f.span(kAtom_fz);
        double im_max = 0.0;
        for (int i = 0; i < n; ++i) {
            const int b = (im * n + i) * 3;
            x_[b + 0] = f_x[i];
            x_[b + 1] = f_y[i];
            x_[b + 2] = f_z[i];
            // Gradient = -force. The "force" here is the NEB force the coupling
            // component wrote, not the gradient of any energy — which is why there is no
            // line search below: there is no phi(alpha) to search on.
            g_[b + 0] = -f_fx[i];
            g_[b + 1] = -f_fy[i];
            g_[b + 2] = -f_fz[i];
            const double fm = std::sqrt(f_fx[i] * f_fx[i] + f_fy[i] * f_fy[i]
                                        + f_fz[i] * f_fz[i]);
            im_max = std::max(im_max, fm);
        }
        image_max_force_[im] = im_max;
        if (im > 0 && im < n_img - 1)
            max_force_ = std::max(max_force_, im_max);
    }

    hist_->push(x_.data(), g_.data(), dim);
    hist_->direction(g_.data(), dir_.data(), dim);
    hist_->save(x_.data(), g_.data(), dim);

    // Trust radius: scale the whole step so no coordinate moves more than max_step_.
    // Uniform scaling, not per-coordinate clipping — clipping changes the direction,
    // and a quasi-Newton direction that has been bent is no longer a descent direction.
    double dmax = 0.0;
    for (int j = 0; j < dim; ++j)
        dmax = std::max(dmax, std::abs(dir_[j]));
    const double scale = (dmax > max_step_) ? (max_step_ / dmax) : 1.0;

    for (int im = 1; im < n_img - 1; ++im) { // endpoints stay put
        auto& f = band.frame(im);
        StorageReal* __restrict f_x = f.span(kAtom_x);
        StorageReal* __restrict f_y = f.span(kAtom_y);
        StorageReal* __restrict f_z = f.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            const int b = (im * n + i) * 3;
            f_x[i] = x_[b + 0] + scale * dir_[b + 0];
            f_y[i] = x_[b + 1] + scale * dir_[b + 1];
            f_z[i] = x_[b + 2] + scale * dir_[b + 2];
        }
    }
}

} // namespace path
} // namespace cpu
} // namespace ATV
