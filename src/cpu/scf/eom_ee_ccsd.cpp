/// @file cpu/scf/eom_ee_ccsd.cpp
/// @brief Spin-orbital EOM-EE-CCSD singlet excitation energies.
///
/// Intermediates follow PySCF `cc.gintermediates` (Gauss & Stanton 1995 Table
/// III, EOM section (b)); the right-hand σ equations are Wang, Tu & Wang,
/// JCTC 10, 5567 (2014) Eqs. (9)–(10) as implemented in `eom_gccsd.eeccsd_matvec`.
/// Packed s4 vector basis: r1 + tril(i>j)×tril(a>b) r2. Singlet roots are
/// selected by α↔β symmetry of the R1 block (closed-shell reference).

#include "atomiverse/cpu/scf/eom_ee_ccsd.h"
#include "atomiverse/cpu/scf/eom_cc2.h"
#include "atomiverse/cpu/scf/eom_sf_ccsd.h"
#include "atomiverse/cpu/scf/internal/eom_cc2_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ee_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_sf_ccsd_pack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/internal/ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ccsd_common.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/uhf.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

using T2 = xt::xtensor<double, 4>;
using T1 = xt::xtensor<double, 2>;

/// Packed vector size: nov + C(no,2)*C(nv,2).
int vector_size(int no, int nv) {
    return no * nv + (no * (no - 1) / 2) * (nv * (nv - 1) / 2);
}

void unpack_s4(const std::vector<double>& v, int no, int nv, T1& r1, T2& r2) {
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    r1 = xt::zeros<double>({No, Nv});
    r2 = xt::zeros<double>({No, No, Nv, Nv});
    const int nov = no * nv;
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a)
            r1(i, a) = v[static_cast<std::size_t>(i * nv + a)];
    int k = nov;
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < i; ++j) // i > j  (tril k=-1)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < a; ++b) { // a > b
                    const double x = v[static_cast<std::size_t>(k++)];
                    r2(i, j, a, b) = x;
                    r2(j, i, a, b) = -x;
                    r2(i, j, b, a) = -x;
                    r2(j, i, b, a) = x;
                }
}

std::vector<double> pack_s4(const T1& r1, const T2& r2, int no, int nv) {
    std::vector<double> v(static_cast<std::size_t>(vector_size(no, nv)), 0.0);
    const int nov = no * nv;
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a)
            v[static_cast<std::size_t>(i * nv + a)] = r1(i, a);
    int k = nov;
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < i; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < a; ++b)
                    v[static_cast<std::size_t>(k++)] = r2(i, j, a, b);
    return v;
}

/// Antisymmetrize T2 in place: ¼ P(ij)P(ab).
void antisym_t2(T2& t2, int no, int nv) {
    T2 s = xt::zeros<double>(
        {static_cast<std::size_t>(no), static_cast<std::size_t>(no),
         static_cast<std::size_t>(nv), static_cast<std::size_t>(nv)});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    s(i, j, a, b) = 0.25
                                    * (t2(i, j, a, b) - t2(i, j, b, a)
                                       - t2(j, i, a, b) + t2(j, i, b, a));
    t2 = s;
}

/// Right-hand EOM-EE-CCSD σ = H̄ r (Wang/Tu/Wang + PySCF eom_gccsd).
std::vector<double> matvec(const EomImds& im, const std::vector<double>& vec) {
    const int no = im.no;
    const int nv = im.nv;
    // Reuse dense tensors across matvecs (Jacobian build calls this dim times).
    // Shape-stable thread_local avoids O(dim) heap churn on tiny O₂ cases.
    thread_local T1 r1, Hr1;
    thread_local T2 r2, Hr2, tmp2, pp;
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    if (r1.shape().size() != 2 || r1.shape(0) != No || r1.shape(1) != Nv) {
        r1 = xt::zeros<double>({No, Nv});
        Hr1 = xt::zeros<double>({No, Nv});
        r2 = xt::zeros<double>({No, No, Nv, Nv});
        Hr2 = xt::zeros<double>({No, No, Nv, Nv});
        tmp2 = xt::zeros<double>({No, No, Nv, Nv});
        pp = xt::zeros<double>({No, No, Nv, Nv});
    } else {
        std::fill(r1.begin(), r1.end(), 0.0);
        std::fill(r2.begin(), r2.end(), 0.0);
        std::fill(Hr1.begin(), Hr1.end(), 0.0);
        std::fill(Hr2.begin(), Hr2.end(), 0.0);
    }
    // unpack into r1/r2
    {
        const int nov = no * nv;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
                r1(i, a) = vec[static_cast<std::size_t>(i * nv + a)];
        int k = nov;
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < i; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < a; ++b) {
                        const double x = vec[static_cast<std::size_t>(k++)];
                        r2(i, j, a, b) = x;
                        r2(j, i, a, b) = -x;
                        r2(i, j, b, a) = -x;
                        r2(j, i, b, a) = x;
                    }
    }

    // Eq. (9) Hr1
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a) {
            double s = 0.0;
            for (int e = 0; e < nv; ++e)
                s += im.Fvv(a, e) * r1(i, e);
            for (int m = 0; m < no; ++m)
                s -= im.Foo(m, i) * r1(m, a);
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    s += im.Fov(m, e) * r2(i, m, a, e);
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    s += im.Wovvo(m, a, e, i) * r1(m, e);
            for (int m = 0; m < no; ++m)
                for (int n = 0; n < no; ++n)
                    for (int e = 0; e < nv; ++e)
                        s -= 0.5 * im.Wooov(m, n, i, e) * r2(m, n, a, e);
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    for (int f = 0; f < nv; ++f)
                        s += 0.5 * im.Wvovv(a, m, e, f) * r2(i, m, e, f);
            Hr1(i, a) = s;
        }

    // Eq. (10) Hr2 — tmpab contribution then P(ab)
    // Precompute Z_eb = Σ_{mnf} Woovv_mnef · r2_mnbf  (O(no² nv³))
    // so  Σ_{mnef} Woovv · t2_ijae · r2_mnbf = Σ_e t2_ijae · Z_eb
    thread_local T1 Z_eb; // [nv, nv]
    if (Z_eb.shape().size() != 2 || Z_eb.shape(0) != Nv || Z_eb.shape(1) != Nv)
        Z_eb = xt::zeros<double>({Nv, Nv});
    else
        std::fill(Z_eb.begin(), Z_eb.end(), 0.0);
    for (int e = 0; e < nv; ++e)
        for (int b = 0; b < nv; ++b) {
            double s = 0.0;
            for (int m = 0; m < no; ++m)
                for (int n = 0; n < no; ++n)
                    for (int f = 0; f < nv; ++f)
                        s += im.Woovv(m, n, e, f) * r2(m, n, b, f);
            Z_eb(e, b) = s;
        }
    // W_am = Σ_{ef} Wvovv_amef · r1_me  → used as Σ_f t2_ijfb · W_am with e-loop
    // Actually: Σ_{mef} Wvovv(a,m,e,f) * t2(i,j,f,b) * r1(m,e)
    //         = Σ_f t2(i,j,f,b) * Σ_{me} Wvovv(a,m,e,f) * r1(m,e)
    thread_local T1
        Wavf; // [nv, nv] for fixed a? reuse: Wavf(a,f)=Σ_me Wvovv(a,m,e,f)r1(m,e)
    if (Wavf.shape().size() != 2 || Wavf.shape(0) != Nv || Wavf.shape(1) != Nv)
        Wavf = xt::zeros<double>({Nv, Nv});
    else
        std::fill(Wavf.begin(), Wavf.end(), 0.0);
    for (int a = 0; a < nv; ++a)
        for (int f = 0; f < nv; ++f) {
            double s = 0.0;
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    s += im.Wvovv(a, m, e, f) * r1(m, e);
            Wavf(a, f) = s;
        }

    std::fill(tmp2.begin(), tmp2.end(), 0.0);
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    double tab = 0.0;
                    for (int e = 0; e < nv; ++e)
                        tab += im.Fvv(b, e) * r2(i, j, a, e);
                    for (int e = 0; e < nv; ++e)
                        tab -= 0.5 * im.t2(i, j, a, e) * Z_eb(e, b);
                    for (int m = 0; m < no; ++m)
                        tab -= im.Wovoo(m, b, i, j) * r1(m, a);
                    for (int f = 0; f < nv; ++f)
                        tab -= im.t2(i, j, f, b) * Wavf(a, f);
                    tmp2(i, j, a, b) = tab;
                }
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    Hr2(i, j, a, b) = tmp2(i, j, a, b) - tmp2(i, j, b, a);

    // tmpij: Σ_{mnef} Woovv_mnef · t2_imab · r2_jnef
    //       = Σ_m t2(i,m,a,b) * Σ_{nef} Woovv(m,n,e,f) * r2(j,n,e,f)
    // Let Y(m,j) = Σ_{nef} Woovv(m,n,e,f) * r2(j,n,e,f)
    thread_local T1 Y_mj;
    if (Y_mj.shape().size() != 2 || Y_mj.shape(0) != No || Y_mj.shape(1) != No)
        Y_mj = xt::zeros<double>({No, No});
    else
        std::fill(Y_mj.begin(), Y_mj.end(), 0.0);
    for (int m = 0; m < no; ++m)
        for (int j = 0; j < no; ++j) {
            double s = 0.0;
            for (int n = 0; n < no; ++n)
                for (int e = 0; e < nv; ++e)
                    for (int f = 0; f < nv; ++f)
                        s += im.Woovv(m, n, e, f) * r2(j, n, e, f);
            Y_mj(m, j) = s;
        }
    // Σ_{mne} Wooov(m,n,i,e) * t2(n,j,a,b) * r1(m,e)
    // = Σ_n t2(n,j,a,b) * Σ_{me} Wooov(m,n,i,e) * r1(m,e)
    // Let U(n,i) = Σ_{me} Wooov(m,n,i,e) * r1(m,e)
    thread_local T1 U_ni;
    if (U_ni.shape().size() != 2 || U_ni.shape(0) != No || U_ni.shape(1) != No)
        U_ni = xt::zeros<double>({No, No});
    else
        std::fill(U_ni.begin(), U_ni.end(), 0.0);
    for (int n = 0; n < no; ++n)
        for (int i = 0; i < no; ++i) {
            double s = 0.0;
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    s += im.Wooov(m, n, i, e) * r1(m, e);
            U_ni(n, i) = s;
        }

    std::fill(tmp2.begin(), tmp2.end(), 0.0);
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    double s = 0.0;
                    for (int m = 0; m < no; ++m)
                        s -= im.Foo(m, j) * r2(i, m, a, b);
                    for (int m = 0; m < no; ++m)
                        s -= 0.5 * im.t2(i, m, a, b) * Y_mj(m, j);
                    for (int e = 0; e < nv; ++e)
                        s += im.Wvvvo(a, b, e, j) * r1(i, e);
                    for (int n = 0; n < no; ++n)
                        s += im.t2(n, j, a, b) * U_ni(n, i);
                    tmp2(i, j, a, b) = s;
                }
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    Hr2(i, j, a, b) += tmp2(i, j, a, b) - tmp2(j, i, a, b);

    // P(ij)P(ab) Wovvo · r2
    std::fill(tmp2.begin(), tmp2.end(), 0.0);
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    double s = 0.0;
                    for (int m = 0; m < no; ++m)
                        for (int e = 0; e < nv; ++e)
                            s += im.Wovvo(m, b, e, j) * r2(i, m, a, e);
                    tmp2(i, j, a, b) = s;
                }
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    pp(i, j, a, b) = tmp2(i, j, a, b) - tmp2(j, i, a, b);
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    Hr2(i, j, a, b) += pp(i, j, a, b) - pp(i, j, b, a);

    // 0.5 Woooo · r2 + 0.5 Wvvvv · r2
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    double s = 0.0;
                    for (int m = 0; m < no; ++m)
                        for (int n = 0; n < no; ++n)
                            s += 0.5 * im.Woooo(m, n, i, j) * r2(m, n, a, b);
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s += 0.5 * im.Wvvvv(a, b, e, f) * r2(i, j, e, f);
                    Hr2(i, j, a, b) += s;
                }

    antisym_t2(Hr2, no, nv);
    return pack_s4(Hr1, Hr2, no, nv);
}

/// Majority spin-flip R1 weight (matches PySCF eomsf on high-spin refs).
/// Layout: occ=[α|β], vir=[α|β]. When n_α ≥ n_β, score the Ms−1 (α→β)
/// channel; otherwise Ms+1 (β→α). Pure-doubles / opposite-flip / conserving
/// roots score low.
double sf_score(const std::vector<double>& vec, int no, int nv, int na, int nb,
                int nva, int nvb) {
    T1 r1;
    T2 r2;
    unpack_s4(vec, no, nv, r1, r2);
    double flip_maj = 0.0, flip_min = 0.0, cons = 0.0;
    // α→α
    for (int i = 0; i < na; ++i)
        for (int a = 0; a < nva; ++a)
            cons += r1(i, a) * r1(i, a);
    // β→β
    for (int i = 0; i < nb; ++i)
        for (int a = 0; a < nvb; ++a)
            cons += r1(na + i, nva + a) * r1(na + i, nva + a);
    // α→β (Ms −1)
    double ab = 0.0;
    for (int i = 0; i < na; ++i)
        for (int a = 0; a < nvb; ++a)
            ab += r1(i, nva + a) * r1(i, nva + a);
    // β→α (Ms +1)
    double ba = 0.0;
    for (int i = 0; i < nb; ++i)
        for (int a = 0; a < nva; ++a)
            ba += r1(na + i, a) * r1(na + i, a);
    if (na >= nb) {
        flip_maj = ab;
        flip_min = ba;
    } else {
        flip_maj = ba;
        flip_min = ab;
    }
    const double tot = flip_maj + flip_min + cons;
    if (tot < 1e-14)
        return -1.0; // pure doubles / null
    return flip_maj / tot;
}

/// Closed-shell singlet score: R1 α-block ≈ β-block (Ms=0 singlet).
/// Spin-orbital layout: occ = [α spatial | β spatial], vir same.
double singlet_score(const std::vector<double>& vec, int no, int nv,
                     int n_occ_sp, int n_vir_sp) {
    T1 r1;
    T2 r2;
    unpack_s4(vec, no, nv, r1, r2);
    double same = 0.0, diff = 0.0, norm = 0.0;
    for (int i = 0; i < n_occ_sp; ++i)
        for (int a = 0; a < n_vir_sp; ++a) {
            const double ra = r1(i, a);
            const double rb = r1(n_occ_sp + i, n_vir_sp + a);
            same += (ra + rb) * (ra + rb);
            diff += (ra - rb) * (ra - rb);
            norm += ra * ra + rb * rb;
        }
    // Also Ms=±1 channels live only in mixed-spin blocks — heavy weight there
    // ⇒ triplet/spin-flip. Penalize if αα/ββ is weak relative to mixed.
    double mixed = 0.0;
    for (int i = 0; i < n_occ_sp; ++i)
        for (int a = 0; a < n_vir_sp; ++a) {
            mixed += r1(i, n_vir_sp + a) * r1(i, n_vir_sp + a);
            mixed += r1(n_occ_sp + i, a) * r1(n_occ_sp + i, a);
        }
    if (norm + mixed < 1e-14)
        return -1.0; // pure doubles / null
    // Singlet Ms=0: same ≫ diff and little mixed R1.
    return same / (same + diff + mixed + 1e-30);
}

/// Dense EOM-EE Jacobian on a converged (CCSD or CC2) amplitude set; return
/// the lowest `n_states` singlet roots [Hartree]. Shared by EOM-EE-CCSD and
/// EOM-CC2 (PySCF evaluates the same σ at CC2 T₁/T₂ when `cc2=True`).
std::vector<double> eom_ee_singlet_roots_ha(const CCSDSolution& sol,
                                            int n_states,
                                            const char* method_tag) {
    if (sol.nv <= 0 || sol.no <= 0) {
        throw std::invalid_argument(std::string(method_tag)
                                    + ": no virtual spin-orbitals — empty EE "
                                      "space.");
    }

    const int dim = vector_size(sol.no, sol.nv);
    if (n_states > dim)
        throw std::invalid_argument(
            std::string(method_tag) + ": n_states=" + std::to_string(n_states)
            + " exceeds EOM dimension " + std::to_string(dim) + ".");

    const EomImds im(sol);

    // Dense Jacobian H[:,j] = σ(e_j)
    std::vector<std::vector<double>> H(
        static_cast<std::size_t>(dim),
        std::vector<double>(static_cast<std::size_t>(dim), 0.0));
    for (int j = 0; j < dim; ++j) {
        std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
        e[static_cast<std::size_t>(j)] = 1.0;
        const std::vector<double> sig = matvec(im, e);
        for (int i = 0; i < dim; ++i)
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                sig[static_cast<std::size_t>(i)];
    }

    auto eigs = eig_dense(H, "eom_ee_ccsd", true);

    // Collect real eigenvalues with singlet character.
    struct Root {
        double w;
        double score;
    };
    std::vector<Root> singlets;
    for (const auto& ep : eigs) {
        if (std::abs(ep.wi) > 1e-8)
            continue;
        if (!(ep.wr > 1e-8)) // drop zero / negative / null
            continue;
        const double sc =
            singlet_score(ep.vr, sol.no, sol.nv, sol.n_occ_sp, sol.n_vir_sp);
        if (sc >= 0.5) // predominantly singlet Ms=0
            singlets.push_back(Root{ep.wr, sc});
    }
    std::sort(singlets.begin(), singlets.end(),
              [](const Root& a, const Root& b) { return a.w < b.w; });

    // Fallback: if singlet filter is too strict (tiny systems / pure doubles
    // singlets), take all positive real roots sorted and rely on the FCI
    // coincidence for 2e — but still prefer high singlet score.
    if (static_cast<int>(singlets.size()) < n_states) {
        singlets.clear();
        for (const auto& ep : eigs) {
            if (std::abs(ep.wi) > 1e-8 || !(ep.wr > 1e-8))
                continue;
            const double sc = singlet_score(ep.vr, sol.no, sol.nv, sol.n_occ_sp,
                                            sol.n_vir_sp);
            singlets.push_back(Root{ep.wr, sc});
        }
        // Prefer high singlet score, then energy.
        std::sort(singlets.begin(), singlets.end(),
                  [](const Root& a, const Root& b) {
                      if (std::abs(a.score - b.score) > 0.05)
                          return a.score > b.score;
                      return a.w < b.w;
                  });
        // Re-sort the top candidates by energy after score filter:
        // take roots with score ≥ 0.3 first; if still short, fill by energy.
        std::vector<Root> preferred;
        for (const auto& r : singlets)
            if (r.score >= 0.3)
                preferred.push_back(r);
        if (static_cast<int>(preferred.size()) >= n_states) {
            std::sort(preferred.begin(), preferred.end(),
                      [](const Root& a, const Root& b) { return a.w < b.w; });
            singlets = preferred;
        } else {
            // All positive real roots by energy (includes triplets).
            std::sort(singlets.begin(), singlets.end(),
                      [](const Root& a, const Root& b) { return a.w < b.w; });
            // Drop near-degenerate triplet clones: keep unique energies.
            std::vector<Root> uniq;
            for (const auto& r : singlets) {
                bool dup = false;
                for (const auto& u : uniq)
                    if (std::abs(u.w - r.w) < 1e-8) {
                        // keep the higher singlet score
                        if (r.score > u.score)
                            const_cast<Root&>(u) = r;
                        dup = true;
                        break;
                    }
                if (!dup)
                    uniq.push_back(r);
            }
            // Prefer unique roots ordered by descending singlet score among
            // those with score≥0.5, else by energy.
            std::vector<Root> s_only;
            for (const auto& r : uniq)
                if (r.score >= 0.5)
                    s_only.push_back(r);
            if (static_cast<int>(s_only.size()) >= n_states) {
                std::sort(
                    s_only.begin(), s_only.end(),
                    [](const Root& a, const Root& b) { return a.w < b.w; });
                singlets = s_only;
            } else {
                // For H2, unique energies are T, S, S' — take those with the
                // best singlet score first.
                std::sort(uniq.begin(), uniq.end(),
                          [](const Root& a, const Root& b) {
                              return a.score > b.score;
                          });
                singlets = uniq;
                std::sort(singlets.begin(), singlets.end(),
                          [](const Root& a, const Root& b) {
                              // Primary: score; secondary: energy for ties
                              if (std::abs(a.score - b.score) > 0.1)
                                  return a.score > b.score;
                              return a.w < b.w;
                          });
            }
        }
    }

    if (static_cast<int>(singlets.size()) < n_states)
        throw std::runtime_error(std::string(method_tag) + ": found only "
                                 + std::to_string(singlets.size())
                                 + " usable EE roots; requested n_states="
                                 + std::to_string(n_states));

    // Final order: ascending ω among the selected singlets.
    // Re-pick: among roots with score ≥ 0.5, sort by energy; take n_states.
    {
        std::vector<Root> good;
        for (const auto& r : eigs) {
            if (std::abs(r.wi) > 1e-8 || !(r.wr > 1e-8))
                continue;
            const double sc =
                singlet_score(r.vr, sol.no, sol.nv, sol.n_occ_sp, sol.n_vir_sp);
            if (sc >= 0.5)
                good.push_back(Root{r.wr, sc});
        }
        // Dedup energies
        std::sort(good.begin(), good.end(),
                  [](const Root& a, const Root& b) { return a.w < b.w; });
        std::vector<Root> dedup;
        for (const auto& r : good) {
            if (!dedup.empty() && std::abs(dedup.back().w - r.w) < 1e-8)
                continue;
            dedup.push_back(r);
        }
        if (static_cast<int>(dedup.size()) >= n_states)
            singlets = dedup;
    }

    std::vector<double> roots_ha;
    roots_ha.reserve(static_cast<std::size_t>(n_states));
    for (int k = 0; k < n_states; ++k)
        roots_ha.push_back(singlets[static_cast<std::size_t>(k)].w);
    return roots_ha;
}

} // namespace

EOMEECCSDResult eom_ee_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states, double tol,
                                     int max_iter) {
    if (n_states < 1)
        throw std::invalid_argument(
            "eom_ee_ccsd: n_states must be ≥ 1 (dimensionless).");

    CCSDSolution sol =
        ccsd_solve(ints, det, e_scf_ha, tol, max_iter, /*cc2=*/false);
    EOMEECCSDResult out;
    out.e_corr_ha = sol.result.e_corr;
    out.e_ccsd_ha = sol.result.e_tot;
    out.ccsd_iterations = sol.result.iterations;
    out.converged = sol.result.converged;

    const std::vector<double> roots_ha =
        eom_ee_singlet_roots_ha(sol, n_states, "eom_ee_ccsd");
    out.energy_ev.reserve(roots_ha.size());
    for (double w : roots_ha)
        out.energy_ev.push_back(w * units::kHartreeInEv);
    out.converged = true;
    return out;
}

EOMCC2Result eom_cc2_energies(const internal::IntegralView& ints,
                              const SlaterDeterminant& det, double e_scf_ha,
                              int n_states, double tol, int max_iter) {
    if (n_states < 1)
        throw std::invalid_argument(
            "eom_cc2: n_states must be ≥ 1 (dimensionless).");

    CCSDSolution sol =
        ccsd_solve(ints, det, e_scf_ha, tol, max_iter, /*cc2=*/true);
    EOMCC2Result out;
    out.e_corr_ha = sol.result.e_corr;
    out.e_cc2_ha = sol.result.e_tot;
    out.cc2_iterations = sol.result.iterations;
    out.converged = sol.result.converged;

    const std::vector<double> roots_ha =
        eom_ee_singlet_roots_ha(sol, n_states, "eom_cc2");
    out.energy_ev.reserve(roots_ha.size());
    for (double w : roots_ha)
        out.energy_ev.push_back(w * units::kHartreeInEv);
    out.converged = true;
    return out;
}

EOMEECCSDResult RHF::eom_ee_ccsd(Frame& frame, int n_states) {
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "RHF::eom_ee_ccsd: SCF reference did not converge — refusing to "
            "compute EOM-EE-CCSD on a non-stationary determinant (no "
            "fallback).");
    return eom_ee_ccsd_energies(integral_view_(), determinant(), energy(),
                                n_states);
}

EOMCC2Result RHF::eom_cc2(Frame& frame, int n_states) {
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "RHF::eom_cc2: SCF reference did not converge — refusing to "
            "compute EOM-CC2 on a non-stationary determinant (no fallback).");
    return eom_cc2_energies(integral_view_(), determinant(), energy(),
                            n_states);
}

EOMSFCCSDResult eom_sf_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states, double tol,
                                     int max_iter) {
    if (n_states < 1)
        throw std::invalid_argument(
            "eom_sf_ccsd: n_states must be ≥ 1 (dimensionless).");
    if (det.restricted())
        throw std::invalid_argument(
            "eom_sf_ccsd: unrestricted UHF reference required (use "
            "eom_ee_ccsd for closed-shell EE).");
    if (det.generalized())
        throw std::invalid_argument(
            "eom_sf_ccsd: GHF reference is not supported.");

    CCSDSolution sol = uccsd_solve(ints, det, e_scf_ha, tol, max_iter);
    EOMSFCCSDResult out;
    out.e_corr_ha = sol.result.e_corr;
    out.e_ccsd_ha = sol.result.e_tot;
    out.ccsd_iterations = sol.result.iterations;
    out.converged = sol.result.converged;

    if (sol.nv <= 0 || sol.no <= 0)
        throw std::invalid_argument(
            "eom_sf_ccsd: no virtual spin-orbitals — empty SF space.");

    const int dim = vector_size(sol.no, sol.nv);
    if (n_states > dim)
        throw std::invalid_argument(
            "eom_sf_ccsd: n_states=" + std::to_string(n_states)
            + " exceeds EOM dimension " + std::to_string(dim) + ".");

    const EomImds im(sol);

    // Majority SF subspace (ΔMs = −1 when n_α ≥ n_β): packed basis vectors
    // whose unit excitation changes 2Ms by −2. Cuts the dense Jacobian from
    // full EE (~784 on O₂/STO-3G) to the SF block (~PySCF eomsf size class).
    const int na = sol.n_alpha;
    const int nb = sol.n_beta;
    const int nva = sol.n_vir_a;
    const int nvb = sol.n_vir_b;
    auto occ_spin = [&](int i) -> int { return (i < na) ? 0 : 1; };
    auto vir_spin = [&](int a) -> int { return (a < nva) ? 0 : 1; };
    // 2Ms: α=+1, β=−1. Target Δ(2Ms)=−2 for majority flip when na≥nb.
    const int dms_target = (na >= nb) ? -2 : +2;

    std::vector<int> sf_idx;
    sf_idx.reserve(static_cast<std::size_t>(dim / 2 + 8));
    // Singles
    for (int i = 0; i < sol.no; ++i)
        for (int a = 0; a < sol.nv; ++a) {
            const int dms =
                (vir_spin(a) == 0 ? 1 : -1) - (occ_spin(i) == 0 ? 1 : -1);
            if (dms == dms_target)
                sf_idx.push_back(i * sol.nv + a);
        }
    // Doubles (packed i>j, a>b)
    int kpack = sol.no * sol.nv;
    for (int i = 0; i < sol.no; ++i)
        for (int j = 0; j < i; ++j)
            for (int a = 0; a < sol.nv; ++a)
                for (int b = 0; b < a; ++b) {
                    const int dms = (vir_spin(a) == 0 ? 1 : -1)
                                    + (vir_spin(b) == 0 ? 1 : -1)
                                    - (occ_spin(i) == 0 ? 1 : -1)
                                    - (occ_spin(j) == 0 ? 1 : -1);
                    if (dms == dms_target)
                        sf_idx.push_back(kpack);
                    ++kpack;
                }

    const int nsf = static_cast<int>(sf_idx.size());
    if (nsf < 1)
        throw std::invalid_argument("eom_sf_ccsd: empty majority SF subspace.");
    if (n_states > nsf)
        throw std::invalid_argument(
            "eom_sf_ccsd: n_states=" + std::to_string(n_states)
            + " exceeds SF dimension " + std::to_string(nsf) + ".");

    std::vector<std::vector<double>> H(
        static_cast<std::size_t>(nsf),
        std::vector<double>(static_cast<std::size_t>(nsf), 0.0));
    std::vector<double> e(static_cast<std::size_t>(dim), 0.0);
    for (int j = 0; j < nsf; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[static_cast<std::size_t>(sf_idx[static_cast<std::size_t>(j)])] = 1.0;
        const std::vector<double> sig = matvec(im, e);
        for (int i = 0; i < nsf; ++i)
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                sig[static_cast<std::size_t>(
                    sf_idx[static_cast<std::size_t>(i)])];
    }

    auto eigs = eig_dense(H, "eom_ee_ccsd", true);

    struct Root {
        double w;
    };
    std::vector<Root> sfs;
    for (const auto& ep : eigs) {
        if (std::abs(ep.wi) > 1e-8)
            continue;
        if (ep.wr < -1e-6)
            continue;
        sfs.push_back(Root{ep.wr});
    }
    std::sort(sfs.begin(), sfs.end(),
              [](const Root& a, const Root& b) { return a.w < b.w; });

    if (static_cast<int>(sfs.size()) < n_states)
        throw std::runtime_error("eom_sf_ccsd: found only "
                                 + std::to_string(sfs.size())
                                 + " usable SF roots; requested n_states="
                                 + std::to_string(n_states));

    out.energy_ev.reserve(static_cast<std::size_t>(n_states));
    for (int k = 0; k < n_states; ++k)
        out.energy_ev.push_back(sfs[static_cast<std::size_t>(k)].w
                                * units::kHartreeInEv);
    out.converged = true;
    return out;
}

EOMSFCCSDResult UHF::eom_sf_ccsd(Frame& frame, int n_states,
                                 const SCFInitialGuess* guess) {
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame, guess);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "UHF::eom_sf_ccsd: SCF reference did not converge — refusing to "
            "compute EOM-SF-CCSD on a non-stationary determinant (no "
            "fallback).");
    return eom_sf_ccsd_energies(integral_view_(), determinant(), energy(),
                                n_states);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
