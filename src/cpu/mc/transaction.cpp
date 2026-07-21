/// @file src/cpu/mc/transaction.cpp
/// @brief cpu::mc::Transaction implementation — Frame-patch record/rollback
///        over the migration row serializers.

#include "atomiverse/cpu/mc/transaction.h"

#include <algorithm>
#include <stdexcept>

#include "atomiverse/cpu/comm/atom_pack.h"

namespace ATV {
namespace cpu {
namespace mc {

void Transaction::begin(Verb v, const cpu::Frame& f) {
    clear();
    verb = v;
    nlocal_before_ = f.nlocal();
    source_frame_ = &f;

    // A volume move records every owned row. Reserve/compile here, once,
    // before its particle loop. Single-row verbs get the same boundary
    // preparation; Insert and None carry no pre-mutation row.
    if (v == Verb::Box)
        prepare_plans_(f, f.nlocal());
    else if (v == Verb::Displace || v == Verb::Delete)
        prepare_plans_(f, 1);
}

void Transaction::record_touch(const cpu::Frame& f, int row) {
    const int k = static_cast<int>(touched_.size());
    if (source_frame_ != &f)
        throw std::logic_error(
            "mc::Transaction::record_touch: source Frame differs from begin()");
    prepare_plans_(f, k + 1);
    comm::require_schema_fingerprint(patch_plan_, source_plan_.fingerprint,
                                     source_plan_.version);
    buf_.resize(comm::max_exchange_bytes(f, source_plan_));
    const std::size_t size =
        comm::pack_exchange(f, source_plan_, row, buf_.data(), buf_.size());
    comm::unpack_exchange(patch_, patch_plan_, k, buf_.data(), size);
    touched_.push_back(static_cast<LocalInt>(row));
    patch_.set_nlocal(k + 1);
}

void Transaction::record_box(const cpu::Frame& f) {
    patch_.simbox = f.simbox;
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
    if (!touched_.empty()) {
        if (!source_plan_.valid_for(f) || !source_plan_.writable)
            source_plan_ = comm::make_exchange_plan(f);
        if (!patch_plan_.valid_for(patch_))
            patch_plan_ = comm::make_exchange_plan(patch_);
        comm::require_schema_fingerprint(source_plan_, patch_plan_.fingerprint,
                                         patch_plan_.version);
        buf_.resize(comm::max_exchange_bytes(patch_, patch_plan_));
        for (std::size_t k = 0; k < touched_.size(); ++k) {
            const std::size_t size =
                comm::pack_exchange(patch_, patch_plan_, static_cast<int>(k),
                                    buf_.data(), buf_.size());
            comm::unpack_exchange(f, source_plan_, touched_[k], buf_.data(),
                                  size);
        }
    }
    if (box_recorded_)
        f.simbox = patch_.simbox;
}

void Transaction::clear() {
    verb = Verb::None;
    touched_.clear();
    patch_.set_nlocal(0);
    nlocal_before_ = -1;
    box_recorded_ = false;
}

void Transaction::prepare_plans_(const cpu::Frame& source, int required_rows) {
    if (required_rows < 0)
        throw std::invalid_argument(
            "mc::Transaction: required patch rows must be non-negative");

    if (!source_plan_.valid_for(source))
        source_plan_ = comm::make_exchange_plan(source);

    if (!patch_plan_.valid_for(patch_))
        patch_plan_ = comm::make_exchange_plan(patch_);

    if (patch_plan_.version != source_plan_.version
        || patch_plan_.fingerprint != source_plan_.fingerprint) {
        patch_ = source.clone_structure(std::max(1, required_rows));
        patch_plan_ = comm::make_exchange_plan(patch_);
    } else if (patch_.capacity() < required_rows) {
        patch_.grow(required_rows);
        patch_plan_ = comm::make_exchange_plan(patch_);
    }
}

} // namespace mc
} // namespace cpu
} // namespace ATV
