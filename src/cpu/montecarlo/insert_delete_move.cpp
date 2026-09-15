/// @file cpu/montecarlo/insert_delete_move.cpp
/// @brief cpu::mc::InsertDeleteMove implementation.

#include "atomiverse/cpu/montecarlo/insert_delete_move.h"

#include <random>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV {
namespace cpu {
namespace mc {

InsertDeleteMove::InsertDeleteMove(int type, double mass_amu)
    : type_(type), mass_amu_(mass_amu) {
    if (type <= 0)
        throw std::invalid_argument("InsertDeleteMove: type must be > 0");
    if (!(mass_amu > 0.0))
        throw std::invalid_argument("InsertDeleteMove: mass must be > 0 amu");
}

void InsertDeleteMove::launch(const cpu::LaunchContext& ctx) {
    if (ctx.step < 0)
        return; // energy-only bootstrap pass

    std::mt19937_64* rng = ctx.runtime.mc.rng;
    mc::Transaction* txn = ctx.runtime.mc.proposal;
    if (rng == nullptr || txn == nullptr) {
        throw std::runtime_error(
            "InsertDeleteMove::launch: ctx.mc.rng / ctx.mc.proposal not "
            "published — InsertDeleteMove must run under an MC preset");
    }

    std::uniform_real_distribution<double> u01(0.0, 1.0);

    if (u01(*rng) < 0.5) {
        // ── Insert at a uniform fractional position ─────────────────────
        txn->begin(Transaction::Verb::Insert, ctx.frame);
        const int nlocal = ctx.frame.nlocal();
        if (nlocal + 1 > ctx.frame.capacity())
            ctx.frame.grow(nlocal + 1);
        // Resolved AFTER grow() -- grow() invalidates any span/pointer taken
        // before it (generation-aware Span contract, cpu/frame/block.h).
        StorageReal* __restrict sp_x = ctx.frame.span(kAtom_x);
        StorageReal* __restrict sp_y = ctx.frame.span(kAtom_y);
        StorageReal* __restrict sp_z = ctx.frame.span(kAtom_z);
        StorageReal* __restrict sp_vx = ctx.frame.span(kAtom_vx);
        StorageReal* __restrict sp_vy = ctx.frame.span(kAtom_vy);
        StorageReal* __restrict sp_vz = ctx.frame.span(kAtom_vz);
        AccReal* __restrict sp_fx = ctx.frame.span(kAtom_fx);
        AccReal* __restrict sp_fy = ctx.frame.span(kAtom_fy);
        AccReal* __restrict sp_fz = ctx.frame.span(kAtom_fz);
        StorageReal* __restrict sp_charge = ctx.frame.span(kAtom_charge);
        Span<std::int8_t> sp_image = ctx.frame.span(kAtom_image);
        StorageReal* __restrict sp_mass = ctx.frame.span(kAtom_mass);
        int* __restrict sp_type = ctx.frame.span(kAtom_type);
        TagInt* __restrict sp_tag = ctx.frame.span(kAtom_tag);

        const int slot = nlocal;
        const double u0 = u01(*rng), u1 = u01(*rng), u2 = u01(*rng);
        const auto& c = ctx.frame.box.cell();
        sp_x[slot] = static_cast<StorageReal>(u0 * c(0, 0) + u1 * c(1, 0)
                                              + u2 * c(2, 0));
        sp_y[slot] = static_cast<StorageReal>(u0 * c(0, 1) + u1 * c(1, 1)
                                              + u2 * c(2, 1));
        sp_z[slot] = static_cast<StorageReal>(u0 * c(0, 2) + u1 * c(1, 2)
                                              + u2 * c(2, 2));
        // Zero everything else — this slot may hold a stale deleted row.
        sp_vx[slot] = 0;
        sp_vy[slot] = 0;
        sp_vz[slot] = 0;
        sp_fx[slot] = 0;
        sp_fy[slot] = 0;
        sp_fz[slot] = 0;
        sp_charge[slot] = 0;
        sp_image(slot, 0) = 0;
        sp_image(slot, 1) = 0;
        sp_image(slot, 2) = 0;
        if (ctx.frame.has_bonded()) {
            auto num_bond = ctx.frame.span(kAtom_num_bond);
            auto num_angle = ctx.frame.span(kAtom_num_angle);
            auto num_dihedral = ctx.frame.span(kAtom_num_dihedral);
            auto num_improper = ctx.frame.span(kAtom_num_improper);
            auto nspecial = ctx.frame.span(kAtom_nspecial);
            num_bond[slot] = 0;
            num_angle[slot] = 0;
            num_dihedral[slot] = 0;
            num_improper[slot] = 0;
            nspecial(slot, 0) = 0;
            nspecial(slot, 1) = 0;
            nspecial(slot, 2) = 0;
        }
        sp_mass[slot] = static_cast<StorageReal>(mass_amu_);
        sp_type[slot] = type_;
        sp_tag[slot] = ctx.frame.acquire_tag();
        ctx.frame.set_nlocal(nlocal + 1);
    } else {
        // ── Delete a uniformly chosen atom ──────────────────────────────
        const int nlocal = ctx.frame.nlocal();
        if (nlocal == 0) {
            // "A trial deletion from N = 0 is simply rejected" — empty
            // proposal; the policy rolls it back without a Boltzmann draw.
            txn->begin(Transaction::Verb::None, ctx.frame);
            return;
        }
        std::uniform_int_distribution<int> pick(0, nlocal - 1);
        const int victim = pick(*rng);
        txn->begin(Transaction::Verb::Delete, ctx.frame);
        txn->record_touch(ctx.frame, victim); // full row, topology included
        ctx.frame.copy_row(victim, nlocal - 1);
        ctx.frame.set_nlocal(nlocal - 1);
    }
}

} // namespace mc
} // namespace cpu
} // namespace ATV
