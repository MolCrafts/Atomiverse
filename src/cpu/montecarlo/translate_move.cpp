/// @file cpu/montecarlo/translate_move.cpp
/// @brief cpu::mc::TranslateMove implementation.

#include "atomiverse/cpu/montecarlo/translate_move.h"

#include <random>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV {
namespace cpu {
namespace mc {

TranslateMove::TranslateMove(double delta) : delta_(delta) {
    if (!(delta > 0.0)) {
        throw std::invalid_argument("TranslateMove: delta must be > 0 Å");
    }
}

void TranslateMove::launch(const cpu::LaunchContext& ctx) {
    // step < 0 is the driver's energy-only bootstrap pass — propose nothing so
    // the evaluator sees the unperturbed initial configuration.
    if (ctx.step < 0)
        return;

    std::mt19937_64* rng = ctx.runtime.mc.rng;
    mc::Transaction* txn = ctx.runtime.mc.proposal;
    // Both are published by cpu::mc::MC::run(); a null here means the move was driven
    // outside an MC run, which is a usage error.
    if (rng == nullptr || txn == nullptr) {
        throw std::runtime_error(
            "TranslateMove::launch: ctx.mc.rng / ctx.mc.proposal not "
            "published — TranslateMove must run under cpu::mc::MC");
    }

    const int n = ctx.frame.nlocal();
    if (n == 0) {
        // Reachable under GCMC: a chain at a very negative μ can delete its way
        // to an empty cell, and the shared MoveSelector still hands the next
        // trial to this move. There is no atom to pick, so this is an empty
        // proposal — stamp `Verb::None`, exactly as InsertDeleteMove does for a
        // deletion drawn on N = 0 (insert_delete_move.cpp), and let the
        // OutcomePolicy dispose of the trial with no Boltzmann draw.
        //
        // Without the guard `pick(0, n - 1)` below is built with a > b, whose
        // precondition the standard leaves undefined: the draw is unspecified
        // and is then used as a row index. A draw landing inside a frame that
        // still has spare capacity from earlier insertions would snapshot and
        // displace a dead row — a trial move on an atom that is not in the
        // system, judged and possibly committed, with no diagnostic at all.
        txn->begin(Transaction::Verb::None, ctx.frame);
        return;
    }

    // Hoisted once at the launch boundary (the 13.7x rule) -- dominates both
    // the read below and the wrapped write-back.
    StorageReal* __restrict sp_x = ctx.frame.span(kAtom_x);
    StorageReal* __restrict sp_y = ctx.frame.span(kAtom_y);
    StorageReal* __restrict sp_z = ctx.frame.span(kAtom_z);

    std::uniform_int_distribution<int> pick(0, n - 1);
    std::uniform_real_distribution<double> disp(-delta_, delta_);

    const int i = pick(*rng);
    // Snapshot the row into the patch BEFORE mutating it.
    txn->begin(Transaction::Verb::Displace, ctx.frame);
    txn->record_touch(ctx.frame, i);

    double nx = static_cast<double>(sp_x[i]) + disp(*rng);
    double ny = static_cast<double>(sp_y[i]) + disp(*rng);
    double nz = static_cast<double>(sp_z[i]) + disp(*rng);

    // Wrap the displaced atom back into the primary cell so coordinates stay
    // bounded over long runs — the COM otherwise random-walks off to infinity
    // (a displacement perpendicular to a pair bond is ΔE=0 and always accepted),
    // which eventually drives an atom outside the box and blows up the cell
    // list. Energy is unaffected: the evaluator uses the minimum image. A
    // degenerate (zero-length) box is left as-is (wrap_axis is a no-op).
    ctx.frame.box.wrap_into(nx, ny, nz);

    sp_x[i] = static_cast<StorageReal>(nx);
    sp_y[i] = static_cast<StorageReal>(ny);
    sp_z[i] = static_cast<StorageReal>(nz);
}

} // namespace mc
} // namespace cpu
} // namespace ATV
