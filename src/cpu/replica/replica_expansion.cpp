/// @file cpu/replica/replica_expansion.cpp
/// @brief LinearInterpolation / IdentityReplication — the two shipped expansions.

#include "atomiverse/cpu/replica/replica_expansion.h"

#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {

LinearInterpolation::LinearInterpolation(cpu::Frame final_state, int n_images)
    : final_(std::move(final_state)), n_images_(n_images) {
    if (n_images < 3) {
        throw std::invalid_argument(
            "LinearInterpolation: n_images must be >= 3 (got "
            + std::to_string(n_images)
            + ") — with two images there is nothing between the endpoints to "
              "relax, "
              "which is not a path calculation");
    }
}

std::vector<Replica>
LinearInterpolation::expand(const cpu::Frame& initial) const {
    const int n = initial.nlocal();
    if (final_.nlocal() != n) {
        throw std::invalid_argument(
            "LinearInterpolation: the final state has "
            + std::to_string(final_.nlocal())
            + " atoms but the initial state has " + std::to_string(n)
            + " — a band between different systems is not a band");
    }

    // Three DISTINCT frames are read/written below (initial / final_ / r.frame) — each
    // keeps its own uniquely-named pointers. Sharing a name across them would silently
    // collapse the interpolation into a copy.
    const StorageReal* __restrict init_x = initial.span(kAtom_x);
    const StorageReal* __restrict init_y = initial.span(kAtom_y);
    const StorageReal* __restrict init_z = initial.span(kAtom_z);
    const StorageReal* __restrict fin_x = final_.span(kAtom_x);
    const StorageReal* __restrict fin_y = final_.span(kAtom_y);
    const StorageReal* __restrict fin_z = final_.span(kAtom_z);

    std::vector<Replica> replicas;
    replicas.reserve(n_images_);
    for (int im = 0; im < n_images_; ++im) {
        // t = 0 at the initial state, 1 at the final state. This is also the replica's
        // parameter: for a band, the thing that distinguishes an image from its siblings
        // IS its position along the path.
        const double t = double(im) / double(n_images_ - 1);

        Replica r{initial,
                  t}; // copies topology, masses, types, box — all shared
        // r.frame is a fresh frame each iteration: its spans resolve here, at this
        // loop's boundary.
        StorageReal* __restrict r_x = r.frame.span(kAtom_x);
        StorageReal* __restrict r_y = r.frame.span(kAtom_y);
        StorageReal* __restrict r_z = r.frame.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            r_x[i] = (1.0 - t) * init_x[i] + t * fin_x[i];
            r_y[i] = (1.0 - t) * init_y[i] + t * fin_y[i];
            r_z[i] = (1.0 - t) * init_z[i] + t * fin_z[i];
        }
        replicas.push_back(std::move(r));
    }
    return replicas;
}

void LinearInterpolation::collapse(const std::vector<Replica>& replicas,
                                   cpu::Frame& out) const {
    if (replicas.empty())
        throw std::logic_error("LinearInterpolation::collapse: no replicas");

    int best = 0;
    for (int i = 1; i < static_cast<int>(replicas.size()); ++i) {
        if (replicas[i].frame.meta.get(kEnergy)
            > replicas[best].frame.meta.get(kEnergy))
            best = i;
    }
    const cpu::Frame& saddle = replicas[best].frame;
    const int n = out.nlocal();
    // Two DISTINCT frames (saddle / out) — separate pointer names, never shared.
    const StorageReal* __restrict sad_x = saddle.span(kAtom_x);
    const StorageReal* __restrict sad_y = saddle.span(kAtom_y);
    const StorageReal* __restrict sad_z = saddle.span(kAtom_z);
    StorageReal* __restrict out_x = out.span(kAtom_x);
    StorageReal* __restrict out_y = out.span(kAtom_y);
    StorageReal* __restrict out_z = out.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        out_x[i] = sad_x[i];
        out_y[i] = sad_y[i];
        out_z[i] = sad_z[i];
    }
    out.meta.copy_serializable_from(saddle.meta);
}

IdentityReplication::IdentityReplication(std::vector<double> parameters)
    : parameters_(std::move(parameters)) {
    if (parameters_.size() < 2) {
        throw std::invalid_argument(
            "IdentityReplication: needs >= 2 replicas (got "
            + std::to_string(parameters_.size())
            + ") — one replica is not a replica set, it is a Frame");
    }
}

std::vector<Replica>
IdentityReplication::expand(const cpu::Frame& initial) const {
    std::vector<Replica> replicas;
    replicas.reserve(parameters_.size());
    for (const double p : parameters_)
        replicas.push_back(Replica{initial, p});
    return replicas;
}

void IdentityReplication::collapse(const std::vector<Replica>& replicas,
                                   cpu::Frame& out) const {
    if (replicas.empty())
        throw std::logic_error("IdentityReplication::collapse: no replicas");
    const cpu::Frame& r0 = replicas[0].frame;
    const int n = out.nlocal();
    // Two DISTINCT frames (r0 / out) — separate pointer names, never shared.
    const StorageReal* __restrict r0_x = r0.span(kAtom_x);
    const StorageReal* __restrict r0_y = r0.span(kAtom_y);
    const StorageReal* __restrict r0_z = r0.span(kAtom_z);
    StorageReal* __restrict out_x = out.span(kAtom_x);
    StorageReal* __restrict out_y = out.span(kAtom_y);
    StorageReal* __restrict out_z = out.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        out_x[i] = r0_x[i];
        out_y[i] = r0_y[i];
        out_z[i] = r0_z[i];
    }
    out.meta.copy_serializable_from(r0.meta);
}

} // namespace cpu
} // namespace ATV
