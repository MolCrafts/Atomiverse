/// @file cpu/scf/nevpt2.cpp
/// @brief Strongly contracted NEVPT2 on a closed-shell CASSCF reference.
///
/// Angeli, Cimiraglia, Evangelisti, Leininger, Malrieu, J. Chem. Phys. **114**,
/// 10252 (2001); Angeli, Cimiraglia, Malrieu, J. Chem. Phys. **117**, 9138
/// (2002). Product path matches PySCF `mrpt.NEVPT(mc).kernel()`.
///
/// The eight SC perturber classes are:
///   Sr, Si, Sij, Srs, Sir, Sijr, Srsi, Sijrs
/// (see PySCF `pyscf.mrpt.nevpt2`). Each class contracts ERIs with active-space
/// RDMs and Dyall denominators. When the inactive core and external virtual
/// spaces are both empty (full-valence CAS spanning all MOs), every class has
/// a vanishing index range and the PT2 correction is identically zero.

#include "atomiverse/cpu/scf/nevpt2.h"
#include "atomiverse/cpu/scf/internal/casscf_pack.h"
#include "atomiverse/cpu/scf/internal/nevpt2_pack.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

NEVPT2Result nevpt2_energy(const internal::IntegralView& ints,
                           const SlaterDeterminant& det, double e_scf_ha,
                           int ncas, int nelecas, double orb_tol,
                           int max_macro) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("nevpt2_energy: ERI is not materialized; "
                                    "call with compute_eri=true.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "nevpt2_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (ncas <= 0) {
        throw std::invalid_argument("nevpt2_energy: ncas must be positive (got "
                                    + std::to_string(ncas) + ")");
    }
    if (nelecas < 0 || (nelecas % 2) != 0) {
        throw std::invalid_argument(
            "nevpt2_energy: closed-shell CAS requires even nelecas (got "
            + std::to_string(nelecas) + ")");
    }

    // Zeroth-order: CASSCF (Roos). Same bookkeeping as PySCF mcscf.CASSCF then
    // mrpt.NEVPT.
    const CASSCFResult cas =
        casscf_energy(ints, det, e_scf_ha, ncas, nelecas, orb_tol, max_macro);

    const int n_mo = static_cast<int>(det.coefficients(Spin::alpha).shape(1));
    const int ncore = cas.ncore;
    const int nvirt = n_mo - ncore - cas.ncas;
    if (nvirt < 0) {
        throw std::invalid_argument("nevpt2_energy: nvirt="
                                    + std::to_string(nvirt)
                                    + " is negative (ncore+ncas exceeds n_mo)");
    }

    NEVPT2Result out;
    out.e_casscf = cas.e_casscf;
    out.ncas = cas.ncas;
    out.nelecas = cas.nelecas;
    out.ncore = ncore;
    out.nvirt = nvirt;
    out.converged = cas.converged;

    // SC-NEVPT2 energy correction = Σ_class E_class over the eight Angeli
    // subspaces. Each subspace contracts against at least one core or virtual
    // orbital index; empty ranges contribute exactly zero.
    //
    // Full-valence CAS (ncore = nvirt = 0): all eight classes vanish → e_corr = 0
    // and E_NEVPT2 = E_CASSCF. This is the H₂/STO-3G CAS(2,2) foothold and matches
    // PySCF mrpt.NEVPT to machine precision.
    //
    // Non-empty external space requires 1/2/3-RDMs + Dyall intermediates
    // (make_a* / Sr, Si, …). That path is productized separately; refuse rather
    // than silently return a partial (wrong) sum.
    if (ncore == 0 && nvirt == 0) {
        out.e_corr = 0.0;
        out.e_nevpt2 = cas.e_casscf;
        return out;
    }

    // Partial non-empty bookkeeping already known; implement SC classes that
    // are identically zero when one of the spaces vanishes, and refuse when a
    // non-zero class would require the 3-RDM intermediate stack.
    //
    // Class → requires:
    //   Sr     : nvirt > 0  (+ 3-RDM)
    //   Si     : ncore > 0  (+ 3-RDM)
    //   Sij    : ncore ≥ 1  (+ 3-RDM; PySCF returns 0 when ncore==0)
    //   Srs    : nvirt ≥ 1  (+ 3-RDM; PySCF returns 0 when nvirt==0)
    //   Sir    : ncore > 0 and nvirt > 0 (+ 3-RDM)
    //   Sijr   : ncore ≥ 1 and nvirt ≥ 1 (+ 2-RDM)
    //   Srsi   : ncore ≥ 1 and nvirt ≥ 1 (+ 2-RDM)
    //   Sijrs  : ncore ≥ 1 and nvirt ≥ 1 (no active RDM; pure (ia|jb) doubles)
    //
    // Any non-empty space on this foothold build still needs the RDM stack —
    // throw so the caller cannot confuse a stub zero with a real PT2 energy.
    throw std::invalid_argument(
        "nevpt2_energy: SC-NEVPT2 with non-empty core/virtual space "
        "(ncore="
        + std::to_string(ncore) + ", nvirt=" + std::to_string(nvirt)
        + ") requires 3-RDM Dyall intermediates not yet on the product path. "
          "Full-valence CAS (ncore=nvirt=0) is supported (e_corr ≡ 0).");
}

} // namespace scf
} // namespace cpu
} // namespace ATV
