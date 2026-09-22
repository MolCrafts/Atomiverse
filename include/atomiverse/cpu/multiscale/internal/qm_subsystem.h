/// @file cpu/multiscale/internal/qm_subsystem.h
/// @brief Private QM subsystem: mask resolve, persistent frame, engine slot,
///        site collection, force scatter (esmm-qmmm-03).
#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/frame/frame.h"
namespace ATV {
namespace cpu {
namespace multiscale {
namespace internal {

/// @brief Resolved link bond (user-frame rows + private-frame H index).
struct ResolvedLink {
    int qm_row = -1;    ///< User-frame row of QM host.
    int mm_row = -1;    ///< User-frame row of MM partner.
    int fqm_h_row = -1; ///< Private QM frame row of link H.
};

/// @brief Persistent QM subsystem owned by a multiscale coupling.
///
/// Not a Component. Allocates the private QM `Frame` once in `build()`;
/// `refresh()` copies geometry each launch without reallocation.
class QmSubsystem {
  public:
    QmSubsystem() = default;

    /// @brief Own an embedding-capable engine (heap-stable address).
    template <typename E,
              std::enable_if_t<!std::is_lvalue_reference_v<E>, int> = 0>
    void set_engine(E&& e) {
        auto owned = std::make_unique<std::decay_t<E>>(std::forward<E>(e));
        engine_ = owned.get();
        owned_engine_ = std::move(owned);
    }

    /// @brief Borrow an embedding-capable engine.
    template <typename E>
    void set_engine_ref(E& e) {
        owned_engine_.reset();
        engine_ = &e;
    }

    template <typename E>
    void set_engine(E&) = delete;

    /// @brief Link-H scale g ∈ (0,1). Parameter, not a region list.
    double link_scale = 0.709;

    double& embed_cutoff() { return embed_cutoff_; }
    double embed_cutoff() const { return embed_cutoff_; }

    bool& reset_history_each_launch() { return reset_history_each_launch_; }

    /// @brief Resolve mask, derive links from bond topology, allocate private
    ///        frame, prepare engine.
    /// @param frame User frame with `kAtomQmZ` filled. Covalent QM/MM cuts are
    ///        **derived** from `num_bond`/`bond_atom` × mask (no bonds ⇒ zero
    ///        links). There is no `set_links` / `links()` user API.
    /// @param mpi_size Communicator size (v1 requires 1).
    /// @param require_embedding When true (electrostatic/polarizable), the
    ///        engine must implement `as_embedding()` and QM charges must be 0.
    ///        When false (mechanical), any `QmEngine` is accepted and QM
    ///        charges may be nonzero.
    /// @throws std::runtime_error / invalid_argument on validation failure.
    void build(Frame& frame, int mpi_size, bool require_embedding = true);

    /// @brief Per-launch: refresh geometry, collect sites, evaluate, scatter.
    /// @param frame User frame.
    /// @param forces_requested Honor LaunchContext force gate.
    /// @param[out] e_qm QM energy written this step [eV] (before scatter).
    void launch(Frame& frame, bool forces_requested, double& e_qm);

    int last_n_links() const {
        return static_cast<int>(resolved_links_.size());
    }
    int n_qm() const { return n_qm_real_; }
    /// @brief User-frame rows of real QM atoms (valid after build/launch).
    const std::vector<int>& qm_rows() const { return qm_rows_; }

    /// @brief Net force residual after scatter [eV/Å] (third-law witness).
    const std::array<double, 3>& last_force_residual() const {
        return force_residual_;
    }

    const Frame* private_frame() const { return qm_frame_.get(); }
    QmEngine* engine() const { return engine_; }

  private:
    void resolve_mask_(const Frame& frame);
    void allocate_private_frame_();
    void refresh_(Frame& frame);
    void collect_sites_(const Frame& frame);
    void scatter_(Frame& frame, bool forces_requested, double e_qm);

    std::unique_ptr<QmEngine> owned_engine_;
    QmEngine* engine_ = nullptr;
    EmbeddingEngine* embedding_ = nullptr;

    double embed_cutoff_ = 12.0;
    bool reset_history_each_launch_ = false;

    std::vector<int> qm_rows_; ///< User-frame rows of real QM atoms.
    std::vector<int> qm_z_;    ///< Atomic numbers from mask.
    std::vector<ResolvedLink> resolved_links_;
    std::unordered_set<TagInt> link_mm_tags_;

    int n_qm_real_ = 0;
    int n_fqm_ = 0;
    int built_n_qm_ = -1; ///< Frozen QM count after first build.

    std::unique_ptr<Frame> qm_frame_;
    std::vector<EmbeddingSite> sites_;
    std::vector<int> site_user_rows_; ///< User-frame row per collected site.

    std::array<double, 3> force_residual_{{0, 0, 0}};
    bool built_ = false;
};

} // namespace internal
} // namespace multiscale
} // namespace cpu
} // namespace ATV
