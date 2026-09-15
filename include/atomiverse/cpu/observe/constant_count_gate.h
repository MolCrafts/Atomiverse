/// @file cpu/observe/constant_count_gate.h
/// @brief Shared measure-side skeleton for trajectory observers: the
/// constant-atom-count admission gate and the blocked-x|y|z snapshot packer.
///
/// The async trajectory observers (RDF/MSD/VACF) require a constant atom count
/// across the captured window (single-rank MVP; no migration). Each previously
/// copy-pasted the first-frame latch + mismatch drop; this header is the single
/// implementation. Dropped frames are counted so `finalize()` can warn instead
/// of losing data silently.
#pragma once

#include <cstddef>
#include <vector>

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {

/// @brief Latches the first snapshot's atom count and rejects (while counting)
/// frames whose count differs.
class ConstantCountGate {
  public:
    /// @brief Admit or reject one snapshot.
    /// @param n atom count of the incoming snapshot.
    /// @return true when admitted (first frame latches `n`; later frames must
    /// match it).
    bool admit(int n) {
        if (n_atoms_ == 0) {
            n_atoms_ = n;
            return true;
        }
        if (n != n_atoms_) {
            ++n_dropped_;
            return false;
        }
        return true;
    }

    /// @brief Latched atom count (0 before the first admitted frame).
    int n_atoms() const { return n_atoms_; }
    /// @brief Number of rejected snapshots (changed atom count).
    BigInt dropped() const { return n_dropped_; }

  private:
    int n_atoms_ = 0;
    BigInt n_dropped_ = 0;
};

/// @brief Pack one snapshot's three per-atom fields into a reusable blocked
/// `a|b|c` scratch buffer (`scratch[0..n) = a`, `[n..2n) = b`, `[2n..3n) = c`).
///
/// @param scratch reused across frames — resized once, no steady-state
/// allocation. @param a,b,c per-atom source fields (e.g. x/y/z or vx/vy/vz).
/// @param n number of owned atoms to copy from each field.
inline void pack_blocked(std::vector<double>& scratch,
                         const std::vector<double>& a,
                         const std::vector<double>& b,
                         const std::vector<double>& c, int n) {
    const auto un = static_cast<std::size_t>(n);
    scratch.resize(3 * un);
    for (std::size_t i = 0; i < un; ++i) {
        scratch[i] = a[i];
        scratch[un + i] = b[i];
        scratch[2 * un + i] = c[i];
    }
}

} // namespace cpu
} // namespace ATV
