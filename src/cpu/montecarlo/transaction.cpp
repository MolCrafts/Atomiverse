/// @file src/cpu/mc/transaction.cpp
/// @brief cpu::mc::Transaction implementation — Frame-patch record/rollback
///        over the cross-Frame row copy.

#include "atomiverse/cpu/montecarlo/transaction.h"

#include <algorithm>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace mc {

void Transaction::begin(Verb v, const cpu::Frame& f) {
    clear();
    verb = v;
    nlocal_before_ = f.nlocal();
    source_frame_ = &f;

    // A volume move records every owned row. Reserve here, once, before its
    // particle loop. Single-row verbs get the same boundary preparation;
    // Insert and None carry no pre-mutation row.
    if (v == Verb::Box)
        prepare_patch_(f, f.nlocal());
    else if (v == Verb::Displace || v == Verb::Delete)
        prepare_patch_(f, 1);
}

void Transaction::record_touch(const cpu::Frame& f, int row) {
    const int k = static_cast<int>(touched_.size());
    if (source_frame_ != &f)
        throw std::logic_error(
            "mc::Transaction::record_touch: source Frame differs from begin()");
    prepare_patch_(f, k + 1);
    patch_.copy_row_from(f, k, row);
    touched_.push_back(static_cast<LocalInt>(row));
    patch_.set_nlocal(k + 1);
}

void Transaction::record_box(const cpu::Frame& f) {
    patch_.box = f.box;
    box_recorded_ = true;
}

void Transaction::rollback(cpu::Frame& f) {
    if (nlocal_before_ < 0)
        throw std::logic_error(
            "mc::Transaction::rollback: no proposal recorded (begin() never "
            "ran)");
    // Population first (an Insert's appended rows fall off the end), then the
    // touched rows (a Delete's victim gets its original content back — the
    // compaction survivor copied into its slot is overwritten, and the
    // survivor's own end-row content was never destroyed by copy_row).
    f.set_nlocal(nlocal_before_);
    for (std::size_t k = 0; k < touched_.size(); ++k)
        f.copy_row_from(patch_, touched_[k], static_cast<int>(k));
    if (box_recorded_)
        f.box = patch_.box;
}

void Transaction::clear() {
    verb = Verb::None;
    touched_.clear();
    patch_.set_nlocal(0);
    nlocal_before_ = -1;
    box_recorded_ = false;
}

void Transaction::prepare_patch_(const cpu::Frame& source, int required_rows) {
    if (required_rows < 0)
        throw std::invalid_argument(
            "mc::Transaction: required patch rows must be non-negative");

    const cpu::Block& atoms = source.block("atoms");
    if (cloned_from_ != &atoms || cloned_generation_ != atoms.generation()) {
        patch_ = source.clone_structure(std::max(1, required_rows));
        cloned_from_ = &atoms;
        cloned_generation_ = atoms.generation();
    } else if (patch_.capacity() < required_rows) {
        patch_.grow(required_rows);
    }
}

} // namespace mc
} // namespace cpu
} // namespace ATV
