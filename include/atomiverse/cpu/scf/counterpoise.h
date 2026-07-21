/// @file cpu/scf/counterpoise.h
/// @brief Boys–Bernardi counterpoise (CP) correction for BSSE.
///
/// Pure orchestration over a concrete mean-field method: five serial SCF
/// runs on a dimer Frame (dimer, monomer A, monomer B, A-in-AB-basis with B
/// ghost, B-in-AB-basis with A ghost). No Driver subclass, no Component, no
/// Plan registration. Ghost markers live on the method (`ghost_atoms`),
/// never on Frame.
///
/// Sign convention (locked; matches CounterpoiseResult doxygen, unit tests,
/// and the tutorial BSSE section):
///   bsse = (E_A^{AB} − E_A^A) + (E_B^{AB} − E_B^B) ≤ 0
/// Monomer energy necessarily lowers in the larger dimer basis; |bsse| is
/// the basis-set superposition error magnitude.
///
/// Pitfalls (Jensen, *Introduction to Computational Chemistry*, basis/BSSE):
/// - Small bases often **over-correct** with CP.
/// - Site–site CP for multi-body clusters is not size-consistent with
///   full-cluster CP.
/// - Monomer deformation (geometry relaxation) is a separate correction;
///   this class evaluates CP at fixed dimer-fragment geometries only.
///
/// Citation: Boys & Bernardi, *Mol. Phys.* **19**, 553 (1970),
/// DOI 10.1080/00268977000101561.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Boys–Bernardi counterpoise energies [Hartree].
///
/// All fields are total electronic energies (including nuclear repulsion)
/// in Hartree from the five SCF runs, plus derived interaction / BSSE
/// quantities. Sign convention locked:
///   bsse = (e_a_ghost − e_a) + (e_b_ghost − e_b) ≤ 0
///   e_int_cp  = e_dimer − e_a_ghost − e_b_ghost
///   e_int_raw = e_dimer − e_a − e_b
struct CounterpoiseResult {
    double e_dimer = 0;    ///< E_AB^{AB} dimer energy [Hartree].
    double e_a = 0;        ///< E_A^A monomer A in its own basis [Hartree].
    double e_b = 0;        ///< E_B^B monomer B in its own basis [Hartree].
    double e_a_ghost = 0;  ///< E_A^{AB} A in dimer basis (B ghost) [Hartree].
    double e_b_ghost = 0;  ///< E_B^{AB} B in dimer basis (A ghost) [Hartree].
    double e_int_cp = 0;   ///< CP-corrected interaction energy [Hartree].
    double e_int_raw = 0;  ///< Uncorrected interaction energy [Hartree].
    /// @brief Basis-set superposition error [Hartree],
    ///        (e_a_ghost − e_a) + (e_b_ghost − e_b); ≤ 0 by construction.
    double bsse = 0;
};

/// @brief Boys–Bernardi counterpoise orchestrator for a bimolecular complex.
///
/// Construct with the local atom-index sets of monomers A and B (disjoint
/// partitions of the dimer Frame). `run` borrows a user-configured method
/// object, temporarily rewrites its `ghost_atoms`, runs five const-geometry
/// SCFs, restores `ghost_atoms`, and returns the energy components.
///
/// Sub-fragment Frames for free monomers are built with the public Frame
/// API (row copy of positions + Z + simbox). Method `charge` is left
/// untouched for the dimer run; free-monomer and ghosted runs force
/// charge 0 (neutral monomers; charged monomers are out of scope).
///
/// @code
///   cpu::scf::RHF rhf;
///   rhf.basis = BasisSet::Pople631Gs;
///   rhf.tol = 1e-10;
///   cpu::scf::Counterpoise cp({0, 1, 2}, {3, 4, 5}); // water dimer
///   auto r = cp.run(dimer_frame, rhf);
///   // r.e_int_cp, r.bsse [Hartree]; rhf.ghost_atoms restored
/// @endcode
class Counterpoise {
  public:
    /// @brief Bind monomer index sets (local Frame indices of the dimer).
    /// @param monomer_a Local atom indices of fragment A (dimensionless).
    /// @param monomer_b Local atom indices of fragment B (dimensionless).
    /// @throws std::invalid_argument when either set is empty.
    Counterpoise(std::vector<int> monomer_a, std::vector<int> monomer_b)
        : monomer_a_(std::move(monomer_a)), monomer_b_(std::move(monomer_b)) {
        if (monomer_a_.empty() || monomer_b_.empty())
            throw std::invalid_argument(
                "cpu::scf::Counterpoise: both monomer index sets must be "
                "non-empty");
    }

    /// @brief Run the five-SCF Boys–Bernardi counterpoise sequence.
    ///
    /// Temporarily rewrites `m.ghost_atoms` (and, for the four monomer
    /// calculations, `m.charge = 0`); both members are restored on exit
    /// even if an SCF throws. Geometry is const — uses
    /// `Method::run(const Frame&)`. Energies are read from `m.energy()`
    /// [Hartree].
    ///
    /// @tparam Method Concrete mean-field type (e.g. `RHF`) exposing
    ///         `ghost_atoms`, `charge`, `run(const Frame&)`, `energy()`,
    ///         and `converged()`.
    /// @param dimer Full complex geometry (Å); left untouched.
    /// @param m     User-configured method (basis, tol, … already set).
    /// @return CounterpoiseResult with all components [Hartree].
    /// @throws std::runtime_error when any of the five SCFs fails to
    ///         converge (no fallback).
    template <typename Method>
    CounterpoiseResult run(const Frame& dimer, Method& m) const {
        const auto saved_ghost = m.ghost_atoms;
        const int saved_charge = m.charge;

        // RAII restore of method members (ghost_atoms + charge).
        struct Restore {
            Method& m;
            std::vector<int> ghost;
            int charge;
            ~Restore() {
                m.ghost_atoms = std::move(ghost);
                m.charge = charge;
            }
        } restore{m, saved_ghost, saved_charge};

        CounterpoiseResult r;

        // 1. Dimer in dimer basis (user charge applies only here).
        m.ghost_atoms.clear();
        m.charge = saved_charge;
        m.run(dimer);
        if (!m.converged())
            throw std::runtime_error(
                "cpu::scf::Counterpoise: dimer SCF did not converge");
        r.e_dimer = m.energy();

        // Free monomers always neutral (spec: charged monomers out of scope).
        m.charge = 0;

        // 2. Monomer A in its own basis.
        {
            Frame fa = sub_frame_(dimer, monomer_a_);
            m.ghost_atoms.clear();
            m.run(fa);
            if (!m.converged())
                throw std::runtime_error(
                    "cpu::scf::Counterpoise: monomer-A SCF did not converge");
            r.e_a = m.energy();
        }

        // 3. Monomer B in its own basis.
        {
            Frame fb = sub_frame_(dimer, monomer_b_);
            m.ghost_atoms.clear();
            m.run(fb);
            if (!m.converged())
                throw std::runtime_error(
                    "cpu::scf::Counterpoise: monomer-B SCF did not converge");
            r.e_b = m.energy();
        }

        // 4. A in AB basis (B ghosted).
        m.ghost_atoms = monomer_b_;
        m.run(dimer);
        if (!m.converged())
            throw std::runtime_error(
                "cpu::scf::Counterpoise: A-in-AB-basis (B ghost) SCF did "
                "not converge");
        r.e_a_ghost = m.energy();

        // 5. B in AB basis (A ghosted).
        m.ghost_atoms = monomer_a_;
        m.run(dimer);
        if (!m.converged())
            throw std::runtime_error(
                "cpu::scf::Counterpoise: B-in-AB-basis (A ghost) SCF did "
                "not converge");
        r.e_b_ghost = m.energy();

        r.e_int_raw = r.e_dimer - r.e_a - r.e_b;
        r.e_int_cp = r.e_dimer - r.e_a_ghost - r.e_b_ghost;
        // Locked sign: monomer energy lowers in the larger basis → bsse ≤ 0.
        r.bsse = (r.e_a_ghost - r.e_a) + (r.e_b_ghost - r.e_b);
        return r;
    }

  private:
    std::vector<int> monomer_a_;
    std::vector<int> monomer_b_;

    /// @brief Build a sub-Frame from selected dimer rows (public Frame API).
    /// @param dimer Full complex geometry.
    /// @param idx   Local atom indices to copy (must be in range).
    /// @return New Frame with `idx.size()` atoms and the dimer's simbox.
    static Frame sub_frame_(const Frame& dimer, const std::vector<int>& idx) {
        const int n_full = static_cast<int>(dimer.nlocal());
        const int n = static_cast<int>(idx.size());
        for (int i : idx) {
            if (i < 0 || i >= n_full)
                throw std::out_of_range(
                    "cpu::scf::Counterpoise: monomer atom index "
                    + std::to_string(i) + " outside [0, "
                    + std::to_string(n_full) + ")");
        }
        Frame sub(n);
        const auto sx = dimer.span(kAtom_x);
        const auto sy = dimer.span(kAtom_y);
        const auto sz = dimer.span(kAtom_z);
        const auto st = dimer.span(kAtom_type);
        auto dx = sub.span(kAtom_x);
        auto dy = sub.span(kAtom_y);
        auto dz = sub.span(kAtom_z);
        auto dt = sub.span(kAtom_type);
        for (int k = 0; k < n; ++k) {
            const int i = idx[static_cast<std::size_t>(k)];
            dx[k] = sx[i];
            dy[k] = sy[i];
            dz[k] = sz[i];
            dt[k] = st[i];
        }
        sub.simbox = dimer.simbox;
        return sub;
    }
};

} // namespace scf
} // namespace cpu
} // namespace ATV
