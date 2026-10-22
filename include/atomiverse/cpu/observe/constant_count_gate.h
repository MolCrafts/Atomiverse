/// @file cpu/observe/constant_count_gate.h
/// @brief Shared measure-side skeleton for trajectory observers: the
/// constant-atom-count admission gate, the molrs-rejection tally and the
/// blocked-x|y|z / cell-matrix snapshot packers.
///
/// The async trajectory observers (RDF/MSD/VACF) require a constant atom count
/// across the captured window (single-rank MVP; no migration). Each previously
/// copy-pasted the first-frame latch + mismatch drop; this header is the single
/// implementation. Dropped and molrs-rejected frames are counted so
/// `finalize()` can warn instead of losing data silently.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/box.h"

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

/// @brief Tally of snapshots the molrs accumulator refused.
///
/// `on_snapshot` runs on an analysis worker thread, where an escaping
/// exception would terminate the process; an observer catches the molrs
/// refusal there, records it here, and reports the tally in `finalize()`.
class RejectedFrames {
  public:
    /// @brief Count one refused frame; the first reason is kept verbatim.
    /// @param why the exception molrs raised for the frame.
    void record(const std::exception& why) {
        if (count_ == 0)
            first_reason_ = why.what();
        ++count_;
    }

    /// @brief Number of refused frames.
    BigInt count() const { return count_; }
    /// @brief The molrs message of the first refused frame (empty if none).
    const std::string& first_reason() const { return first_reason_; }

  private:
    BigInt count_ = 0;
    std::string first_reason_;
};

/// @brief Keep correlator rows tied to stable atom tags and equally spaced captures.
/// Atom replacement or a dropped capture invalidates a time correlation rather
/// than silently joining unrelated atoms or compressing its time axis.
class CorrelationInput {
  public:
    void pack(std::vector<double>& out, const std::vector<std::int64_t>& tags,
              const std::vector<double>& a, const std::vector<double>& b,
              const std::vector<double>& c, int n, int step, int stride) {
        const auto count = static_cast<std::size_t>(n);
        if (tags.size() < count || a.size() < count || b.size() < count
            || c.size() < count)
            throw std::runtime_error(
                "correlator snapshot lacks tags or captured fields");
        if (ready_ && static_cast<std::int64_t>(step) - last_step_ != stride)
            throw std::runtime_error(
                "correlator snapshots are not equally spaced");
        order_.resize(count);
        std::iota(order_.begin(), order_.end(), 0);
        std::sort(order_.begin(), order_.end(),
                  [&](int i, int j) { return tags[i] < tags[j]; });
        for (std::size_t i = 0; i < count; ++i) {
            const auto tag = tags[order_[i]];
            if (tag <= 0 || (i && tag == tags[order_[i - 1]]))
                throw std::runtime_error(
                    "correlator requires unique positive atom tags");
            if (ready_ && (count != tags_.size() || tag != tags_[i]))
                throw std::runtime_error("correlator atom identities changed");
        }
        if (!ready_) {
            tags_.resize(count);
            for (std::size_t i = 0; i < count; ++i)
                tags_[i] = tags[order_[i]];
        }
        out.resize(3 * count);
        for (std::size_t i = 0; i < count; ++i) {
            const int row = order_[i];
            out[i] = a[row];
            out[count + i] = b[row];
            out[2 * count + i] = c[row];
        }
        last_step_ = step;
        ready_ = true;
    }

  private:
    std::vector<int> order_;
    std::vector<std::int64_t> tags_;
    std::int64_t last_step_ = 0;
    bool ready_ = false;
};

/// @brief A snapshot's cell matrix H as molrs takes it: row-major, widened to
/// double.
/// @param box the snapshot's cell [Å].
/// @return the nine values of `box.cell()` [Å], row-major.
inline std::array<double, 9> cell_matrix(const Box& box) {
    std::array<double, 9> h{};
    const auto& c = box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            h[static_cast<std::size_t>(i * 3 + j)] =
                static_cast<double>(c(i, j));
    return h;
}

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
