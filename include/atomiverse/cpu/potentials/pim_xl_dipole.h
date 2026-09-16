/// @file cpu/potentials/pim_xl_dipole.h
/// @brief Extended-Lagrangian PIM induced-dipole dynamics (iEL/SCF) — CPU.
///
/// Cold-starting the induced-dipole self-consistent cycle (SCC) from μ = 0
/// every molecular-dynamics step wastes iterations and — when the SCC is
/// truncated — pumps energy into the trajectory, because the converged dipole
/// always lags the nuclei. The extended-Lagrangian cure gives the SCC guess
/// its own time-reversible dynamics: an auxiliary dipole set is propagated by
/// the dissipative Niklasson recursion (the classical-polarization analogue of
/// XL-BOMD) and handed to the SCC as the starting point each step.
///
/// This is the iEL/SCF scheme of Albaugh, Demerdash & Head-Gordon,
/// J. Chem. Phys. **143**, 174104 (2015), built by composing two shipped
/// pieces: `cpu::PimInducedDipole` (Thole-damped bulk PIM SCC + Hellmann–
/// Feynman forces) and `cpu::electronic::XlBomd` (Niklasson Table-I
/// dissipative recursion, JCP **130**, 214109 (2009)) acting element-wise on
/// the nlocal×3 dipole array. MetalWalls capability map **§2 "XL dipoles"**;
/// MW's own mass-zero route is MaZe (Coretti, Bonella & Ciccotti, JCP
/// **149**, 191102 (2018)) — the true SHAKE-like constrained propagation is
/// deferred exactly as `ConstPotMaZe` defers it for electrode charges.
/// Units: e, Å, eV, eV/Å.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/electronic/xl_bomd.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

namespace ATV {
namespace cpu {

/// @brief Extended-Lagrangian bulk PIM dipole Potential (iEL/SCF).
///
/// Thin owner: each `launch` warm-starts the owned `PimInducedDipole` SCC
/// from the auxiliary dipoles, converges it, then advances the auxiliary
/// dynamics with the converged result. Energy and forces on Frame are exactly
/// the owned solver's (SCC-converged Hellmann–Feynman), so the fixed point at
/// a stationary geometry is the plain SCF PIM answer.
class PimXlDipole : public Potential {
  public:
    /// @param cutoff Real-space multipole cutoff [Å] (must be > 0).
    /// @param alpha_by_type Isotropic polarizability [Å³] per atom type index
    ///        (`kAtom_type`); types outside the vector → α = 0.
    /// @param thole_a Thole exponential damping factor (dimensionless; 0 =
    ///        off; typical 0.39).
    /// @param xl_order Niklasson dissipation order K ∈ [3, 9] (Table I of
    ///        JCP 130, 214109 (2009); history depth K+1).
    /// @param pol_tol Induced-dipole SCC threshold on max |Δμ| [e·Å].
    /// @param pol_max_iter Maximum SCC iterations per step.
    PimXlDipole(double cutoff, std::vector<double> alpha_by_type,
                double thole_a = 0.39, int xl_order = 5, double pol_tol = 1e-10,
                int pol_max_iter = 100);

    std::string name() const override { return "pim_xl_dipole"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief Owned PIM SCC solver — mutate its public members
    ///        (`thole_a`, `tt_b`, `pol_tol`, …) after construction.
    /// @return The composed `PimInducedDipole`.
    PimInducedDipole& solver() { return pim_; }
    /// @copydoc solver()
    const PimInducedDipole& solver() const { return pim_; }

    /// @brief Niklasson auxiliary-dipole propagator (read-only view).
    /// @return The composed `electronic::XlBomd`.
    const electronic::XlBomd& propagator() const { return xl_; }

    /// @brief Polarization energy of the last launch [eV].
    double last_pol_energy() const { return pim_.last_pol_energy(); }
    /// @brief SCC iterations used on the last launch.
    int last_pol_iters() const { return pim_.last_pol_iters(); }
    /// @brief Whether the SCC converged on the last launch.
    bool last_pol_converged() const { return pim_.last_pol_converged(); }
    /// @brief Converged induced dipoles of the last launch (packed xyz, e·Å).
    const std::vector<double>& last_mu() const { return pim_.last_mu(); }
    /// @brief Whether the last launch was warm-started from the auxiliary
    ///        dipoles (false on the first launch and after `reset`).
    bool last_warm() const { return last_warm_; }
    /// @brief Max |μ_aux − μ_scc| of the last warm launch [e·Å] — how far the
    ///        auxiliary dynamics sits from the Born–Oppenheimer dipole
    ///        surface (0 when the last launch was cold).
    double last_aux_error() const { return aux_err_; }

    /// @brief Drop the auxiliary-dipole history (geometry discontinuity /
    ///        restart): the next launch cold-starts and re-seeds. Atom-count
    ///        changes reset automatically.
    void reset();

  private:
    PimInducedDipole pim_;
    electronic::XlBomd xl_;
    double aux_err_ = 0.0;
    bool last_warm_ = false;
    long n_prev_ = -1;
};

} // namespace cpu
} // namespace ATV
