/// @file src/cpu/gpw/gpw_evaluator.cpp
/// @brief Implementation of the GPW energy evaluator component (see header).

#include "atomiverse/cpu/gpw/gpw_evaluator.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {
constexpr double kAng2Bohr = 1.8897259885789;

const Kind& find_kind(const std::vector<Kind>& kinds, int z) {
    for (const auto& k : kinds)
        if (static_cast<int>(k.element) == z)
            return k;
    throw std::runtime_error(
        "GPWEvaluator: no basis/potential configured for Z="
        + std::to_string(z));
}
} // namespace

namespace {
/// Assemble the GpwAtom set at the frame's current geometry (Å → Bohr).
std::vector<GpwAtom> assemble_atoms(const cpu::Frame& frame,
                                    const GpwEvaluatorConfig& config) {
    const int n = frame.nlocal();
    const int* __restrict sp_type = frame.span(kAtom_type);
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    std::vector<GpwAtom> atoms;
    atoms.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Kind& k = find_kind(config.kinds, sp_type[i]);
        const double pos[3] = {sp_x[i] * kAng2Bohr, sp_y[i] * kAng2Bohr,
                               sp_z[i] * kAng2Bohr};
        atoms.push_back(GpwAtom(pos, k.basis, k.pot, sp_type[i]));
    }
    return atoms;
}
} // namespace

void GPWEvaluator::build(const cpu::BuildContext& ctx) {
    // Validate that every atom in the frame has a configured element, and
    // count valence electrons for the closed-shell occupation.
    const int n_build = ctx.frame.nlocal();
    const int* __restrict sp_type_build = ctx.frame.span(kAtom_type);
    double z_val_total = 0.0;
    for (int i = 0; i < n_build; ++i)
        z_val_total += static_cast<double>(
            find_kind(config_.kinds, sp_type_build[i]).pot.zion);

    // Assemble the solver configuration from the build-time frame. The
    // ATV_GPW_OT diagnostic switch is read once here — re-reading an
    // environment variable per launch had no meaning.
    GpwConfig cfg;
    cfg.cell[0] = ctx.frame.simbox.length(0) * kAng2Bohr;
    cfg.cell[1] = ctx.frame.simbox.length(1) * kAng2Bohr;
    cfg.cell[2] = ctx.frame.simbox.length(2) * kAng2Bohr;
    cfg.cutoff_ry = config_.cutoff_ry;
    cfg.n_occ = static_cast<int>(std::lround(z_val_total / 2.0));
    cfg.max_iter = config_.max_iter;
    cfg.tol = config_.tol;
    cfg.mix = config_.mix;
    cfg.xc = config_.xc;
    cfg.nonperiodic = config_.nonperiodic;
    cfg.d3 = config_.d3;
    if (std::getenv("ATV_GPW_OT"))
        cfg.use_ot = true; // diagnostic: force OT path
    built_cell_ = {cfg.cell[0], cfg.cell[1], cfg.cell[2]};

    solver_ = std::make_unique<GPWSolver>(std::move(cfg));
    solver_->build(assemble_atoms(ctx.frame, config_));
}

void GPWEvaluator::launch(const cpu::LaunchContext& ctx) {
    cpu::Frame& frame = ctx.frame;
    const int n = frame.nlocal();

    // GPW does not support a variable cell: the grids, lattice images and
    // Ewald-like electrostatics are sized from the build-time cell. The old
    // implementation silently solved in the new box — undefined behavior.
    for (int d = 0; d < 3; ++d) {
        const double now = frame.simbox.length(d) * kAng2Bohr;
        if (std::abs(now - built_cell_[static_cast<std::size_t>(d)])
            > 1e-9
                  * std::max(
                      1.0,
                      std::abs(built_cell_[static_cast<std::size_t>(d)]))) {
            throw std::runtime_error(
                "GPWEvaluator::launch: frame box changed since build() "
                "(GPW does not support a variable cell)");
        }
    }

    const auto atoms = assemble_atoms(frame, config_);
    last_ = solver_->run(atoms);
    frame.meta.get(kEnergy) = last_.e_total * ATV::units::kHartreeInEv;

    if (config_.compute_forces) {
        const auto F = gpw_forces_analytic(atoms, solver_->config());
        // [Hartree/Bohr]; one SCF + gradient (was 6N+1 FD)
        AccReal* __restrict sp_fx = frame.span(kAtom_fx);
        AccReal* __restrict sp_fy = frame.span(kAtom_fy);
        AccReal* __restrict sp_fz = frame.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            sp_fx[i] = F[static_cast<std::size_t>(i)][0]
                       * ATV::units::kHartreePerBohrInEvPerAngstrom;
            sp_fy[i] = F[static_cast<std::size_t>(i)][1]
                       * ATV::units::kHartreePerBohrInEvPerAngstrom;
            sp_fz[i] = F[static_cast<std::size_t>(i)][2]
                       * ATV::units::kHartreePerBohrInEvPerAngstrom;
        }
    }
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
