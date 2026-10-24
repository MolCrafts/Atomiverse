/// @file core/evaluation_history.h
/// @brief Author-side commit boundary for state produced by physical evaluation.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ATV {

/// A method keeps its accepted history separate from a prepared candidate.
/// Candidate preparation may fail; final publication and discard must not.
/// Participants must remain alive until their enclosing transaction ends.
class EvaluationHistory {
  public:
    virtual ~EvaluationHistory() = default;
    virtual void commit_evaluation() noexcept = 0;
    virtual void discard_evaluation() noexcept = 0;
};

/// Backend-owned participant ledger, reused by all nested method solves.
/// This stores no Frame or scientific data. A physical step owns a Scope;
/// inner SCF loops and trial replay only borrow that step's transaction.
class EvaluationTransaction {
  public:
    bool active() const noexcept { return active_; }
    bool empty() const noexcept { return participants_.empty(); }
    void reserve(std::size_t count) { participants_.reserve(count); }

    /// Register before preparing a candidate. May allocate/throw; duplicates
    /// are ignored. A participant whose candidate was invalidated remains
    /// registered and makes its eventual commit a no-op.
    void enlist(EvaluationHistory& participant) {
        if (!active_)
            throw std::logic_error(
                "evaluation history requires an active transaction");
        if (std::find(participants_.begin(), participants_.end(), &participant)
            == participants_.end())
            participants_.push_back(&participant);
    }

    class Scope {
      public:
        explicit Scope(EvaluationTransaction& transaction) noexcept
            : transaction_(transaction), owner_(!transaction.active_) {
            if (owner_)
                transaction_.active_ = true;
        }
        ~Scope() {
            if (owner_)
                transaction_.finish_(false);
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

        /// Call only after the complete physical step, including trailing
        /// components and device error checks, succeeds. Nested scopes never
        /// publish their parent's candidates.
        void accept() noexcept {
            if (owner_) {
                transaction_.finish_(true);
                owner_ = false;
            }
        }

      private:
        EvaluationTransaction& transaction_;
        bool owner_;
    };

  private:
    void finish_(bool accept) noexcept {
        for (auto* participant : participants_) {
            if (accept)
                participant->commit_evaluation();
            else
                participant->discard_evaluation();
        }
        participants_.clear();
        active_ = false;
    }

    std::vector<EvaluationHistory*> participants_;
    bool active_ = false;
};

} // namespace ATV
