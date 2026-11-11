/// @file cpu/semiempirical/dftb0.h
/// @brief Non-SCC DFTB (DFTB0) method — static in-tree SK tables, CPU.
#pragma once

#include <limits>
#include <memory>
#include <span>
#include <string>

#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Non-SCC DFTB / DFTB0 energy (+ optional forces) for H/C/N/O.
///
/// Electronic tight-binding without self-consistent charge: Slater–Koster
/// H0/S from static in-tree tables, generalized eigen, band energy + pair
/// repulsive. Forces, when requested, are the closed-form non-SCC gradient
/// (Hellmann–Feynman + Pulay + repulsive) — not a finite difference, and
/// with no Hubbard / γ / Δq term (contrast @ref DFTB2). Tables:
/// all ordered H/C/N/O pairs from transcribed DFTB.org mio-1-1
/// (`basis_data/dftb_mio_sk_tables.h` via `internal/mio_sk_table.h` /
/// `MioSketchPair`). Supported Z: 1, 6, 7, 8; any other element throws.
///
/// Oracle for pins: **DFTB+** 24.1 (non-SCC / DFTB0 path). For a homonuclear
/// diatomic the Mulliken fluctuations vanish by symmetry, so DFTB2 reduces to
/// DFTB0 and both pin the same DFTB+ number — see
/// `regressions/semiempirical/dftb0/`. Units on the public surface:
/// energy [eV], positions [Å], forces [eV/Å].
///
/// @code
///   cpu::semiempirical::DFTB0 dftb0;
///   dftb0.forces(frame);
///   eval.on(evaluate::Phase::evaluate).add_ref(dftb0);
/// @endcode
class DFTB0 : public cpu::Semiempirical {
  public:
    /// @brief Construct the neutral closed-shell mio-1-1 method.
    DFTB0();
    /// @brief Release the retained numerical workspace.
    ~DFTB0() override;
    /// @brief Transfer the method and its retained workspace.
    /// @param other Source method.
    DFTB0(DFTB0&& other) noexcept;
    /// @brief Transfer the method and its retained workspace.
    /// @param other Source method.
    /// @return This method.
    DFTB0& operator=(DFTB0&& other) noexcept;

    /// @brief Overwrite energy [eV] and charges [e]; preserve existing forces.
    /// @param frame Input geometry and result destination.
    void run(Frame& frame);
    /// @brief Evaluate energy and charges without modifying the input Frame.
    /// @param frame Input geometry; results remain on the method.
    void run(const Frame& frame);
    /// @brief Overwrite energy [eV], charges [e] and analytic forces [eV/Å].
    /// @param frame Input geometry and result destination.
    void forces(Frame& frame);

    std::string name() const override { return "dftb0"; }
    /// @brief Declare the Mulliken charge publication to the parent Runtime.
    /// @return True.
    bool updates_owned_charges() const override { return true; }

    /// @brief Allocate or rebuild scratch for the Frame's atom identities.
    /// @param ctx Parent build context.
    void build(const BuildContext& ctx) override;
    /// @brief Accumulate energy and requested forces; overwrite atomic charges.
    /// @param ctx Parent launch context.
    void launch(const LaunchContext& ctx) override;

    /// @brief Last electronic+repulsive energy [eV]; NaN before success or after failure.
    /// @return Total energy [eV].
    double last_energy() const { return energy_; }

    /// @brief Last successful Mulliken charges; empty before a solve or after failure.
    /// @return Charges [e], in Frame atom order, valid until the next evaluation.
    std::span<const double> last_charges() const;

  private:
    struct Workspace;
    std::unique_ptr<Workspace> workspace_;
    double energy_ = std::numeric_limits<double>::quiet_NaN();
    void prepare(const Frame& frame);
    void evaluate(const Frame& frame, bool want_forces);
    void publish(Frame& frame, bool want_forces, bool accumulate) const;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
