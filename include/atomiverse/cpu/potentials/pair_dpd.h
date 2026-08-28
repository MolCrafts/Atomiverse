/// @file cpu/potentials/pair_dpd.h
/// @brief Groot–Warren DPD: soft conservative + pairwise D/R (CPU).
#pragma once

#include <array>
#include <span>

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Dissipative particle dynamics pair force (Groot–Warren 1997).
///
/// *Dissipative particle dynamics* (DPD) is a coarse-grained method: each
/// particle stands for a blob of fluid rather than an atom, so the pair force
/// is soft (finite at \(r=0\)) and the degrees of freedom that were coarsened
/// away are put back as a pairwise thermostat.
///
/// Conservative: \(F^C = a\,w(r)\hat r\) with \(w=1-r/r_c\) and \(\hat r\) the
/// unit vector along \(r_i - r_j\), from \(U^C = \tfrac12 a\,r_c\,w^2\) [eV].
/// Multi-type: an optional symmetric `a_matrix[type_i][type_j]`, keyed by
/// atomic number — see the second constructor.
/// Dissipative / random with \(\sigma^2 = 2\gamma k_B T\) and weight \(w\).
///
/// **Stress:** `kStress` carries the conservative virial
/// \(\sum \Delta r\otimes F^C\) only. The dissipative and random forces are
/// thermostat terms — zero mean at equilibrium, and \(F^R\propto 1/\sqrt{dt}\)
/// — so tallying them would put timestep-dependent noise into every reported
/// pressure without changing its average. See `src/cpu/potentials/pair_dpd.cpp`.
///
/// **Random stream:** \(F^R\) is redrawn every loop step from the pairwise
/// stream \(\{\text{seed}, \text{step}, i, j\}\), and the step is the one the
/// launch carries (`LaunchContext::step`, plus the `set_step` shift that is 0
/// by default) — the only step a Component sees when a Driver owns the loop. A
/// stream frozen at one draw would turn the thermostat into a constant,
/// position-independent kick: \(F^R\) is what injects the energy \(F^D\)
/// removes, so the system would heat or cool monotonically instead of sampling
/// `temperature`.
///
/// `i` and `j` in that stream are LOCAL ATOM INDICES, not global tags, so
/// bit-for-bit reproducibility holds for a given seed, step budget and atom
/// ordering — reorder the frame and the same run draws a different (equally
/// valid) noise realization.
class PairDpd : public Pair {
  public:
    /// @brief DPD's dissipative term is \(v_j - v_i\), so ghost rows carry
    /// velocities. This is the ONLY style in the tree that reads a neighbour
    /// velocity; a run without it does not ship them.
    ///
    /// `each_step`, not `stable`: what DPD needs is not "send velocities
    /// once per rebuild" but "\(v_j\) is the owner's current value when I
    /// compute \(F^D\)". Under a `stable` guarantee it would be up to
    /// `every` steps old for the whole reneighbour interval — a
    /// systematically wrong friction that no energy-conservation check
    /// catches, because DPD is not energy-conserving by construction.
    /// @return This style's ghost declarations.
    static constexpr std::array kGhosts = {
        GhostField::read_each_step(kAtom_vx, "PairDpd"),
        GhostField::read_each_step(kAtom_vy, "PairDpd"),
        GhostField::read_each_step(kAtom_vz, "PairDpd"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    /// @brief Single-type: one amplitude `a` for every pair.
    ///
    /// @param a           Soft repulsion amplitude [eV/Å] — a FORCE, not an
    ///                    energy: it is the value of \(F^C\) at contact.
    /// @param gamma       Dissipative friction γ [eV·fs/Å²] (≥ 0). 0 switches
    ///                    the dissipative AND the random term off, since
    ///                    \(\sigma=\sqrt{2\gamma k_B T}\).
    /// @param temperature Thermostat setpoint T [K] (> 0).
    /// @param cutoff      Pair cutoff r_c [Å] (> 0); `w` and both thermostat
    ///                    weights vanish there.
    /// @param dt          Timestep [fs] (> 0) — must be the one the
    ///                    integrator uses, because \(F^R \propto 1/\sqrt{dt}\).
    /// @param seed        Random-stream seed (dimensionless).
    /// @throws std::invalid_argument on `cutoff <= 0`, `dt <= 0`,
    ///         `temperature <= 0` or `gamma < 0`.
    PairDpd(double a, double gamma, double temperature, double cutoff,
            double dt, std::uint64_t seed = 1);

    /// @brief Multi-component: a per-type-pair amplitude matrix.
    ///
    /// The matrix is indexed by the frame's `kAtom_type` values AS-IS. In this
    /// tree that column carries the ATOMIC NUMBER (`cpu::from_molrs` fills it
    /// from molrs `atomic_number`), so the matrix must be square of side
    /// max(Z) + 1 — an argon/sodium mixture needs 19 rows, not 2 — and a type
    /// outside it throws at `launch` rather than falling back to a default.
    ///
    /// @param a_matrix    Square amplitude table `a_matrix[t_i][t_j]` [eV/Å].
    ///                    Must be supplied SYMMETRIC: the kernel is not
    ///                    symmetrized — it reads `a_matrix[type[i]][type[j]]`
    ///                    for the pair whose lower ATOM INDEX is `i` — so an
    ///                    asymmetric table makes the energy depend on the
    ///                    frame's atom ordering. Only squareness is checked.
    /// @param gamma       Dissipative friction γ [eV·fs/Å²] (≥ 0).
    /// @param temperature Thermostat setpoint T [K] (> 0).
    /// @param cutoff      Pair cutoff r_c [Å] (> 0).
    /// @param dt          Timestep [fs] (> 0).
    /// @param seed        Random-stream seed (dimensionless).
    /// @throws std::invalid_argument on the same parameter ranges as the
    ///         single-type constructor, on an empty `a_matrix`, or on a
    ///         non-square one.
    PairDpd(std::vector<std::vector<double>> a_matrix, double gamma,
            double temperature, double cutoff, double dt,
            std::uint64_t seed = 1);

    std::string name() const override { return "dpd"; }

    /// @brief Shift the random stream so loop step 0 draws at `step`.
    ///
    /// The stream index is `step + LaunchContext::step`, so the default 0
    /// makes the loop step the stream index outright — which is what a Driver
    /// run wants, and what every in-tree registration gets.
    ///
    /// It is a shift rather than the stream index itself because the loop step
    /// restarts at 0 on every `run()` (each `run()` is an independent
    /// simulation). Two consecutive `run()` calls on one DPD component would
    /// therefore replay the *identical* sequence of random forces — the
    /// LAMMPS `run N` continuation problem. Handing the second run the number
    /// of steps already taken continues the stream instead. The other caller is
    /// the component-author side: a unit test that drives one
    /// `launch(…, step = 0)` directly and wants a chosen stream index
    /// (`tests/cpu/potentials/test_pair_dpd.cpp`). Nothing inside the engine
    /// calls it, so under a Driver the stream index simply IS the loop step.
    ///
    /// @param step Stream index that loop step 0 draws at [count; ≥ 0]. Not
    ///             validated — a negative value is just a backwards shift.
    void set_step(int step) { stream_origin_ = step; }

    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double a_default_ = 0;
    std::vector<std::vector<double>> a_mat_;
    bool multi_ = false;
    double gamma_ = 0;
    double temperature_ = 0;
    double dt_ = 1;
    double cutoff_sq_ = 0;
    double sigma_ = 0;
    std::uint64_t seed_ = 1;
    /// Stream index of loop step 0; see `set_step`.
    int stream_origin_ = 0;

    double a_of_(int ti, int tj) const;
};

} // namespace cpu
} // namespace ATV
