/// @file cpu/component/qm_engine.h
/// @brief QM engine protocols for multiscale couplings + embedding site types.
///
/// Two protocols, one header (the neighbor_protocol.h precedent). Neither is a
/// Component: implementors are Drivers (`MeanFieldMethod<D>`), plain method
/// classes (`AM1`), or Components (`DFTB2`).
///
/// Contract on the frame handed to prepare/evaluate: it is the coupling's
/// **private** QM frame (QM atoms + link caps only), pre-zeroed on energy and
/// forces — an engine may overwrite or accumulate, both are correct. The
/// engine MUST NOT consult `ctx.runtime.pair_list` or halo state.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/component.h" // BuildContext / LaunchContext

namespace ATV {
namespace cpu {

/// @brief One external MM-like embedding charge (public currency).
///
/// Positions in Å, charge in e. Used by mean-field `point_charges` and by
/// the private embedding gradient path. Not a Config bag — a plain value type.
struct EmbeddingSite {
    double x = 0; ///< Position x [Å].
    double y = 0; ///< Position y [Å].
    double z = 0; ///< Position z [Å].
    double q = 0; ///< Charge [e].
};

/// @brief Force on one embedding site (public currency).
///
/// Components in eV/Å. Filled by the analytic embedding gradient when
/// `point_charges` is non-empty; empty otherwise.
struct EmbeddingForce {
    double fx = 0; ///< Force x [eV/Å].
    double fy = 0; ///< Force y [eV/Å].
    double fz = 0; ///< Force z [eV/Å].
};

class EmbeddingEngine; // capability probe target

/// @brief Erased electronic-structure engine driven by multiscale couplings.
///
/// Lifecycle matches Component (`prepare` ≈ build, `evaluate` ≈ launch) but
/// is **not** registered into a Plan — couplings call these methods directly.
class QmEngine {
  public:
    virtual ~QmEngine() = default;

    /// @brief Engine label for diagnostics / exception messages.
    virtual std::string qm_engine_name() const = 0;

    /// @brief One-time setup against the QM reference geometry.
    /// @param qm_ctx Build context whose frame is the private QM subsystem.
    virtual void prepare(const BuildContext& qm_ctx) = 0;

    /// @brief One QM evaluation on the private QM frame.
    ///
    /// Writes total energy [eV] to `kEnergy` (and electronic share). When
    /// `qm_ctx.forces_requested`, also writes forces [eV/Å] to `kAtom_f*`.
    /// Engines that can compute per-atom charges write them [e] into
    /// `kAtom_charge`.
    /// @param qm_ctx Launch context on the private QM frame.
    virtual void evaluate(const LaunchContext& qm_ctx) = 0;

    /// @brief Electrostatic-embedding capability, or nullptr.
    virtual EmbeddingEngine* as_embedding() { return nullptr; }

    /// @brief Drop warm-start / density-extrapolation history (MC posture).
    virtual void reset_history() {}
};

/// @brief Opt-in electrostatic embedding capability for a `QmEngine`.
///
/// Probed via `QmEngine::as_embedding()` (default nullptr). Site forces of the
/// last forces-enabled evaluate satisfy Σ_QM F + Σ_MM F = 0 when site
/// self-energy is off.
class EmbeddingEngine {
  public:
    virtual ~EmbeddingEngine() = default;

    /// @brief Replace the external charge set (positions [Å], charges [e]).
    virtual void set_embedding_sites(std::vector<EmbeddingSite> sites) = 0;

    /// @brief Site forces [eV/Å] of the last forces-enabled evaluate, ordered
    ///        as given to `set_embedding_sites`.
    virtual const std::vector<EmbeddingForce>&
    embedding_site_forces() const = 0;

    /// @brief false ⇒ MM–MM Coulomb among the sites is excluded (additive
    ///        QM/MM; MM–MM belongs to the force field). Default true.
    virtual void set_site_self_energy(bool enabled) = 0;
};

} // namespace cpu
} // namespace ATV
