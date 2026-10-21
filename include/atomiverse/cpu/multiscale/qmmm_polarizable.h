/// @file cpu/multiscale/qmmm_polarizable.h
/// @brief Polarizable QM/MM: embedding engine + induced MM dipoles.
#pragma once

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/multiscale/internal/qm_mm_vdw.h"
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"
#include "atomiverse/cpu/potentials/lj_type_coeff.h"

namespace ATV {
namespace cpu {

/// @brief Solve isotropic induced-dipole SCC (Applequist/Thole).
/// @param n Number of polarizable sites.
/// @param pos Site positions, length 3n [Å].
/// @param alpha Isotropic polarizability per site [Å³].
/// @param e_perm Permanent field per site, length 3n [e/Å²].
/// @param thole_a Thole damping (0 disables).
/// @param tol Max |Δμ| [e·Å].
/// @param max_iter Maximum SCC iterations.
/// @param[out] mu Induced dipoles, length 3n [e·Å].
/// @param[out] iters_used Optional iteration count.
/// @return Polarization energy [eV].
double solve_induced_dipoles(int n, const double* pos, const double* alpha,
                             const double* e_perm, double thole_a, double tol,
                             int max_iter, std::vector<double>& mu,
                             int* iters_used = nullptr);

/// @brief Electrostatic embedding + isotropic induced MM dipoles.
///
/// Permanent field at MM sites is built from the engine-written
/// `kAtom_charge` on the private QM frame (Mulliken / AM1 charges), not bare
/// nuclear Z. Throws if that column is all-zero after evaluate.
class QmmmPolarizable : public Potential {
  public:
    /// @param cross_lj QM–MM LJ table.
    /// @param lj_cutoff LJ cutoff [Å].
    /// @param embed_cutoff Embedding site cutoff [Å].
    QmmmPolarizable(std::vector<LJTypeCoeff> cross_lj, double lj_cutoff,
                    double embed_cutoff);

    std::string name() const override { return "qmmm_polarizable"; }

    /// @brief Isotropic polarizability applied to every MM site [Å³].
    double alpha = 1.0;
    /// @brief Thole damping factor (dimensionless).
    double thole_a = 0.39;
    double pol_tol = 1e-8; ///< [e·Å]
    int pol_max_iter = 50;

    /// @brief Link-H scale g ∈ (0,1). Region cuts come from Frame bonds×mask.
    double& link_scale() { return sub_.link_scale; }
    bool& reset_history_each_launch() {
        return sub_.reset_history_each_launch();
    }

    template <typename E,
              std::enable_if_t<!std::is_lvalue_reference_v<E>, int> = 0>
    QmmmPolarizable& set_engine(E&& e) {
        sub_.set_engine(std::forward<E>(e));
        return *this;
    }
    template <typename E>
    QmmmPolarizable& set_engine_ref(E& e) {
        sub_.set_engine_ref(e);
        return *this;
    }
    template <typename E>
    QmmmPolarizable& set_engine(E&) = delete;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double last_qm_energy() const { return e_qm_; }
    double last_mm_energy() const { return e_vdw_; }
    double last_pol_energy() const { return e_pol_; }
    int last_pol_iters() const { return last_pol_iters_; }

  private:
    multiscale::internal::QmSubsystem sub_;
    multiscale::internal::QmMmVdw vdw_;
    double e_qm_ = 0, e_vdw_ = 0, e_pol_ = 0;
    int last_pol_iters_ = 0;
};

} // namespace cpu
} // namespace ATV
