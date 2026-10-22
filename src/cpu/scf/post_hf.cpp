/// @file cpu/scf/post_hf.cpp
/// @brief Post-HF entry points. Algorithms stay in the internal packs
///        (mp2.kernel, CCSD.kernel / ccsd_t, CASCI/CASSCF, NEVPT). This file
///        only moves the dispatch off MeanFieldMethod.

#include "atomiverse/cpu/scf/post_hf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/scf/internal/casci_pack.h"
#include "atomiverse/cpu/scf/internal/casscf_pack.h"
#include "atomiverse/cpu/scf/internal/ccsd_gradient.h"
#include "atomiverse/cpu/scf/internal/ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/dispersion.h"
#include "atomiverse/cpu/scf/internal/eom_cc2_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ea_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ee_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ip_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_sf_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/fci_pack.h"
#include "atomiverse/cpu/scf/internal/mp2_gradient.h"
#include "atomiverse/cpu/scf/internal/mp2_pack.h"
#include "atomiverse/cpu/scf/internal/mp3_pack.h"
#include "atomiverse/cpu/scf/internal/mp4_pack.h"
#include "atomiverse/cpu/scf/internal/nevpt2_pack.h"
#include "atomiverse/cpu/scf/methods.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

void refuse_ks(MeanFieldType method, const char* entry) {
    if (method == MeanFieldType::KS)
        throw std::invalid_argument(
            std::string(entry)
            + ": a Kohn–Sham reference is refused. CC, CI, CAS, and MP3/MP4 "
              "take a Hartree–Fock mean field (PySCF cc.ccsd.CCSD / "
              "mcscf.CASCI / fci.FCI on scf.RHF). The explicit KS product is "
              "MP2 on RKS (double hybrid), not this entry.");
}

void write_gradient(Frame& frame, const GradientResult& gr) {
    AccReal* __restrict fx = frame.span(kAtom_fx);
    AccReal* __restrict fy = frame.span(kAtom_fy);
    AccReal* __restrict fz = frame.span(kAtom_fz);
    const int natm = static_cast<int>(frame.nlocal());
    for (int A = 0; A < natm; ++A) {
        fx[A] = gr.forces(A, 0);
        fy[A] = gr.forces(A, 1);
        fz[A] = gr.forces(A, 2);
    }
    frame.meta.get(kEnergy) = gr.energy;
}

template <typename Method>
void drive_converged(Method& mf, Frame& frame, const SCFInitialGuess* guess,
                     bool materialize_eri, const char* entry) {
    struct RestoreEri {
        bool& value;
        bool saved;
        ~RestoreEri() { value = saved; }
    } restore{mf.compute_eri, mf.compute_eri};
    if (materialize_eri)
        mf.compute_eri = true;
    mf.run(frame, guess);
    mf.compute_eri = restore.saved;
    if (!mf.converged())
        throw std::invalid_argument(
            std::string(entry)
            + ": SCF reference did not converge — refusing to compute a "
              "correlation correction on a non-stationary determinant "
              "(no fallback)");
}

} // namespace

template <typename Method>
MP2<Method>::MP2(Method& mf) : mf_(&mf) {}

template <typename Method>
MP2Result MP2<Method>::run(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    if (mf.method_ == MeanFieldType::KS && mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP2::run: Kohn–Sham MP2 is only the restricted double-hybrid path "
            "(RKS). UKS and ROKS are not that product — use RKS or an HF "
            "reference.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MP2::run: a GHF (spinor) reference is not supported — the MP2 "
            "pack has no spinor transformation. Use RHF or UHF.");
    drive_converged(mf, frame, guess, /*materialize_eri=*/mf.ri == RI::None,
                    "MP2::run");
    if ((mf.spin_ == SpinType::Unrestricted
         || mf.spin_ == SpinType::RestrictedOpen)
        && frozen != 0) {
        throw std::invalid_argument(
            "MP2::run: frozen-core UMP2/ROMP2 is not supported — set "
            "frozen=0, or use closed-shell RHF.");
    }
    if (mf.ri != RI::None) {
        if (mf.spin_ != SpinType::Restricted)
            throw std::invalid_argument(
                "MP2::run: RI-MP2 is closed-shell only — set ri=RI::None for "
                "UMP2/ROMP2, or use RHF.");
        return ri_mp2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                             mf.aux_basis, frozen, os_scale, ss_scale);
    }
    if (mf.spin_ == SpinType::RestrictedOpen) {
        return romp2_energy(mf.integral_view_(), mf.determinant(),
                            mf.F_alpha_uncoupled_, mf.F_beta_uncoupled_,
                            mf.energy(), os_scale, ss_scale);
    }
    if (mf.spin_ == SpinType::Unrestricted)
        return ump2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                           os_scale, ss_scale);
    return mp2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                      frozen, os_scale, ss_scale);
}

template <typename Method>
GradientResult MP2<Method>::forces(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    if (mf.method_ == MeanFieldType::KS && mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP2::forces: Kohn–Sham MP2 is only the restricted double-hybrid "
            "path "
            "(RKS). UKS and ROKS are not that product — use RKS or an HF "
            "reference.");
    if (mf.spin_ != SpinType::Restricted && mf.spin_ != SpinType::Unrestricted)
        throw std::invalid_argument(
            "MP2::forces: RMP2 (RHF) or UMP2 (UHF) analytic gradient only — "
            "ROHF/ROMP2 gradients are a follow-up");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "MP2::forces: RI-MP2 gradients are not implemented — set "
            "ri=RI::None");
    if (frozen != 0)
        throw std::invalid_argument(
            "MP2::forces: frozen-core MP2 gradients are not implemented — set "
            "frozen=0");
    if (std::fabs(os_scale - 1.0) > 0.0 || std::fabs(ss_scale - 1.0) > 0.0)
        throw std::invalid_argument(
            "MP2::forces: spin-component-scaled MP2 gradients are not "
            "implemented — keep os_scale=ss_scale=1");

    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "MP2::forces");

    GradientResult gr;
    if (mf.spin_ == SpinType::Unrestricted) {
        const MP2Result corr =
            ump2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                        /*os_scale=*/1.0, /*ss_scale=*/1.0);
        gr = internal::ump2_analytic_forces(frame, mf.integral_view_(),
                                            mf.determinant(), mf.energy(),
                                            corr.e_corr);
    } else {
        const MP2Result corr =
            mp2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                       /*n_frozen=*/0, /*os_scale=*/1.0, /*ss_scale=*/1.0);
        gr = internal::mp2_analytic_forces(frame, mf.integral_view_(),
                                           mf.determinant(), mf.energy(),
                                           corr.e_corr);
    }
    internal::add_dispersion_forces(frame, mf.d3bj, mf.d4, gr);
    mf.last_grad_ = gr;
    write_gradient(frame, gr);
    return gr;
}

template <typename Method>
MP3<Method>::MP3(Method& mf) : mf_(&mf) {}

template <typename Method>
MP3Result MP3<Method>::run(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "MP3::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP3::run: closed-shell RMP3 only — use RHF (open-shell MP3 is a "
            "follow-up).");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "MP3::run: RI-MP3 is not implemented — set ri=RI::None");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "MP3::run");
    return mp3_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                      frozen);
}

template <typename Method>
MP4<Method>::MP4(Method& mf) : mf_(&mf) {}

template <typename Method>
MP4Result MP4<Method>::run(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "MP4::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP4::run: closed-shell RMP4 only — use RHF (open-shell MP4 is a "
            "follow-up).");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "MP4::run: RI-MP4 is not implemented — set ri=RI::None");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "MP4::run");
    return mp4_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                      frozen);
}

template <typename Method>
FCI<Method>::FCI(Method& mf) : mf_(&mf) {}

template <typename Method>
FCIResult FCI<Method>::run(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "FCI::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "FCI::run: an ROHF reference is not supported — the FCI pack "
            "requires a closed-shell reference. Use RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "FCI::run: a GHF (spinor) reference is not supported — the FCI "
            "pack requires a closed-shell reference. Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "FCI::run");
    return fci_energy(mf.integral_view_(), mf.determinant(), mf.energy());
}

template <typename Method>
CASCI<Method>::CASCI(Method& mf) : mf_(&mf) {}

template <typename Method>
CASCIResult CASCI<Method>::run(Frame& frame, int ncas, int nelecas) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CASCI::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "CASCI::run: an ROHF reference is not supported — closed-shell "
            "CASCI requires RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "CASCI::run: a GHF (spinor) reference is not supported — "
            "closed-shell CASCI requires RHF.");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CASCI::run: closed-shell RHF reference only — UHF CASCI is not "
            "implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CASCI::run");
    return casci_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                        ncas, nelecas);
}

template <typename Method>
CASSCF<Method>::CASSCF(Method& mf) : mf_(&mf) {}

template <typename Method>
CASSCFResult CASSCF<Method>::run(Frame& frame, int ncas, int nelecas) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CASSCF::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "CASSCF::run: an ROHF reference is not supported — closed-shell "
            "CASSCF requires RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "CASSCF::run: a GHF (spinor) reference is not supported — "
            "closed-shell CASSCF requires RHF.");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CASSCF::run: closed-shell RHF reference only — UHF CASSCF is not "
            "implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CASSCF::run");
    return casscf_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                         ncas, nelecas, /*orb_tol=*/std::max(mf.tol, 1e-8),
                         /*max_macro=*/50);
}

template <typename Method>
NEVPT2<Method>::NEVPT2(Method& mf) : mf_(&mf) {}

template <typename Method>
NEVPT2Result NEVPT2<Method>::run(Frame& frame, int ncas, int nelecas) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "NEVPT2::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "NEVPT2::run: an ROHF reference is not supported — closed-shell "
            "SC-NEVPT2 requires RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "NEVPT2::run: a GHF (spinor) reference is not supported — "
            "closed-shell SC-NEVPT2 requires RHF.");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "NEVPT2::run: closed-shell RHF reference only — UHF NEVPT2 is not "
            "implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "NEVPT2::run");
    return nevpt2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                         ncas, nelecas, /*orb_tol=*/std::max(mf.tol, 1e-8),
                         /*max_macro=*/50);
}

template <typename Method>
CCSD<Method>::CCSD(Method& mf) : mf_(&mf) {}

template <typename Method>
CCSDResult CCSD<Method>::run(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::run: closed-shell RHF reference only — UHF / ROHF / GHF "
            "CCSD is not implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CCSD::run");
    return ccsd_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                       /*tol=*/1e-10, /*max_iter=*/100, /*use_diis=*/diis);
}

template <typename Method>
GradientResult CCSD<Method>::forces(Frame& frame,
                                    const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::forces");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::forces: closed-shell RHF reference only — UCCSD gradients "
            "are not implemented (no fallback). Use RHF.");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "CCSD::forces: RI-CCSD gradients are not implemented — set "
            "ri=RI::None");
    if (frozen != 0)
        throw std::invalid_argument(
            "CCSD::forces: frozen-core CCSD gradients are not implemented — "
            "set frozen=0");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "CCSD::forces");
    const CCSDResult corr =
        ccsd_energy(mf.integral_view_(), mf.determinant(), mf.energy());
    GradientResult gr = internal::ccsd_analytic_forces(
        frame, mf.integral_view_(), mf.determinant(), mf.energy(), corr.e_corr);
    internal::add_dispersion_forces(frame, mf.d3bj, mf.d4, gr);
    mf.last_grad_ = gr;
    write_gradient(frame, gr);
    return gr;
}

template <typename Method>
EOMEECCSDResult CCSD<Method>::eom_ee_ccsd(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_ee_ccsd");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_ee_ccsd: closed-shell RHF reference only — EOM-EE "
            "singlet is CCSD.eomee_ccsd_singlet (PySCF), not a UHF / ROHF / "
            "GHF entry (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::eom_ee_ccsd");
    return eom_ee_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
EOMIPCCSDResult CCSD<Method>::eom_ip_ccsd(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_ip_ccsd");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_ip_ccsd: closed-shell RHF reference only — EOM-IP "
            "is CCSD.ipccsd (PySCF), not a UHF / ROHF / GHF entry "
            "(no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::eom_ip_ccsd");
    return eom_ip_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
EOMEACCSDResult CCSD<Method>::eom_ea_ccsd(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_ea_ccsd");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_ea_ccsd: closed-shell RHF reference only — EOM-EA "
            "is CCSD.eaccsd (PySCF), not a UHF / ROHF / GHF entry "
            "(no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::eom_ea_ccsd");
    return eom_ea_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
EOMSFCCSDResult CCSD<Method>::eom_sf_ccsd(Frame& frame, int n_states,
                                          const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_sf_ccsd");
    if (mf.spin_ == SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_sf_ccsd: unrestricted UHF reference required "
            "(PySCF cc.UCCSD.eomsf_ccsd). Closed-shell EE is "
            "CCSD::eom_ee_ccsd.");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true,
                    "CCSD::eom_sf_ccsd");
    return eom_sf_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
CCSDTResult CCSD<Method>::ccsd_t(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::ccsd_t");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::ccsd_t: closed-shell RHF reference only — UHF / ROHF / GHF "
            "CCSD(T) is not implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::ccsd_t");
    return ccsd_t_energy(mf.integral_view_(), mf.determinant(), mf.energy());
}

template <typename Method>
CC2<Method>::CC2(Method& mf) : mf_(&mf) {}

template <typename Method>
CC2Result CC2<Method>::run(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CC2::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CC2::run: closed-shell RHF reference only — UHF / ROHF / GHF CC2 "
            "is not implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CC2::run");
    return cc2_energy(mf.integral_view_(), mf.determinant(), mf.energy());
}

template <typename Method>
EOMCC2Result CC2<Method>::eom_cc2(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CC2::eom_cc2");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CC2::eom_cc2: closed-shell RHF reference only — EOM-CC2 is "
            "the cc2=True EOM-EE singlet (PySCF), not a UHF / ROHF / GHF "
            "entry (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CC2::eom_cc2");
    return eom_cc2_energies(mf.integral_view_(), mf.determinant(), mf.energy(),
                            n_states);
}

#define ATV_INST_POSTHF(T)                                                     \
    template class MP2<T>;                                                     \
    template class MP3<T>;                                                     \
    template class MP4<T>;                                                     \
    template class FCI<T>;                                                     \
    template class CASCI<T>;                                                   \
    template class CASSCF<T>;                                                  \
    template class NEVPT2<T>;                                                  \
    template class CCSD<T>;                                                    \
    template class CC2<T>;

ATV_INST_POSTHF(RHF)
ATV_INST_POSTHF(ROHF)
ATV_INST_POSTHF(UHF)
ATV_INST_POSTHF(GHF)
ATV_INST_POSTHF(RKS)
ATV_INST_POSTHF(UKS)
ATV_INST_POSTHF(ROKS)

#undef ATV_INST_POSTHF

} // namespace scf
} // namespace cpu
} // namespace ATV
