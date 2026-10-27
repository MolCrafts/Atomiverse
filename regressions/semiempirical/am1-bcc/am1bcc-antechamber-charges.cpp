/// @file regressions/semiempirical/am1bcc-antechamber-charges/reg_am1bcc-antechamber-charges.cpp
/// @brief AM1-BCC and ABCG2 charges of phenol and acetate vs AmberTools
///        antechamber.
///
/// Oracle: **AmberTools 26.1 antechamber** `-c bcc` / `-c abcg2`, with sqm
/// (SQM version 19) supplying the AM1 base charges. Run at the fixed input
/// geometry (`maxcyc=0`). Inputs, the run script and the captured `.ac` /
/// `sqm.out` are in `tools/bcc_oracle/`. Method: Jakalian, Jack & Bayly,
/// *J. Comput. Chem.* **23**, 1623 (2002).
///
/// Nothing here executes an oracle binary; every reference number is a
/// transcribed constant. See REFERENCE.md for the tolerance rationale.
///
/// Exit non-zero on failure.

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/semiempirical/am1_bcc_charges.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "harness/registry.h"

namespace atv_reg_semiempirical_am1bcc_antechamber_charges {

namespace {

using ATV::cpu::semiempirical::AM1;
using ATV::cpu::semiempirical::AM1BCCCharges;

struct Atom {
    int z;
    double x, y, z_ang;
};

struct Molecule {
    const char* name;
    int net_charge;
    std::vector<Atom> atoms;
    std::vector<std::array<int, 2>> bonds;    // 0-based rows
    std::vector<double> sqm_mulliken;         // sqm.out "Mulliken Charge"
    std::vector<double> bcc;                  // .ac column 9, -c bcc
    std::vector<double> abcg2;                // .ac column 9, -c abcg2
    std::vector<std::vector<int>> equivalent; // rows antechamber averaged
};

// ── Pinned antechamber / sqm output (tools/bcc_oracle/outputs/) ─────────────
Molecule phenol() {
    return {
        "phenol",
        0,
        {{6, 1.390, 0.000, 0.0},
         {6, 0.695, 1.204, 0.0},
         {6, -0.695, 1.204, 0.0},
         {6, -1.390, 0.000, 0.0},
         {6, -0.695, -1.204, 0.0},
         {6, 0.695, -1.204, 0.0},
         {8, 2.750, 0.000, 0.0},
         {1, 1.235, 2.139, 0.0},
         {1, -1.235, 2.139, 0.0},
         {1, -2.470, 0.000, 0.0},
         {1, -1.235, -2.139, 0.0},
         {1, 1.235, -2.139, 0.0},
         {1, 3.063, 0.908, 0.0}},
        {{0, 1},
         {1, 2},
         {2, 3},
         {3, 4},
         {4, 5},
         {5, 0},
         {0, 6},
         {1, 7},
         {2, 8},
         {3, 9},
         {4, 10},
         {5, 11},
         {6, 12}},
        {0.091, -0.224, -0.088, -0.167, -0.092, -0.159, -0.252, 0.129, 0.131,
         0.131, 0.134, 0.144, 0.222},
        {0.136100, -0.191500, -0.090000, -0.167000, -0.090000, -0.191500,
         -0.498100, 0.136500, 0.132500, 0.131000, 0.132500, 0.136500, 0.423000},
        {0.136100, -0.173500, -0.072000, -0.149000, -0.072000, -0.173500,
         -0.537100, 0.118500, 0.114500, 0.113000, 0.114500, 0.118500, 0.462000},
        {{1, 5}, {2, 4}, {7, 11}, {8, 10}},
    };
}

Molecule acetate() {
    return {
        "acetate",
        -1,
        {{6, 0.000, 0.000, 0.000},
         {6, 1.520, 0.000, 0.000},
         {8, 2.145, 1.083, 0.000},
         {8, 2.145, -1.083, 0.000},
         {1, -0.363, 0.000, 1.028},
         {1, -0.363, -0.890, -0.514},
         {1, -0.363, 0.890, -0.514}},
        {{0, 1}, {1, 2}, {1, 3}, {0, 4}, {0, 5}, {0, 6}},
        {-0.272, 0.319, -0.596, -0.596, 0.048, 0.049, 0.049},
        {-0.204100, 0.899600, -0.861300, -0.861300, 0.009367, 0.009367,
         0.009367},
        {-0.204100, 0.689000, -0.756000, -0.756000, 0.009367, 0.009367,
         0.009367},
        {{2, 3}, {4, 5, 6}},
    };
}

// ── Tolerances (see REFERENCE.md § Tolerance rationale) ─────────────────────
// sqm prints Mulliken charges to 1e-3 (half-ULP 5e-4) and antechamber feeds
// that rounded table to the BCC stage; 1e-4 more covers the CODATA-vs-frozen
// constant floor measured against MOPAC (am1-water-mopac-energy).
constexpr double kBaseTolE = 6.0e-4;
// Final charge = averaged base + increment; the increment is exact (same
// BCCPARM rows), so the final residual inherits the base bound.
constexpr double kChargeTolE = 6.0e-4;
constexpr double kEquivalentTolE = 1.0e-12;
constexpr double kNetChargeTolE = 1.0e-6;

ATV::cpu::Frame build(const Molecule& m) {
    const int n = static_cast<int>(m.atoms.size());
    ATV::cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        f.span(ATV::kAtom_type)[k] = m.atoms[k].z;
        f.span(ATV::kAtom_x)[k] = static_cast<ATV::StorageReal>(m.atoms[k].x);
        f.span(ATV::kAtom_y)[k] = static_cast<ATV::StorageReal>(m.atoms[k].y);
        f.span(ATV::kAtom_z)[k] =
            static_cast<ATV::StorageReal>(m.atoms[k].z_ang);
        f.span(ATV::kAtom_tag)[k] = i;
    }
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (const auto& b : m.bonds)
        f.add_bond(b[0], b[1], 0);
    f.box = ATV::cpu::Box::cubic(40.0); // isolation only
    return f;
}

bool check_per_atom(const char* what, const Molecule& m,
                    const std::vector<double>& got,
                    const std::vector<double>& want, double tol) {
    double worst = 0.0;
    int worst_i = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d =
            std::isfinite(got[i]) ? std::abs(got[i] - want[i]) : INFINITY;
        if (d > worst) {
            worst = d;
            worst_i = static_cast<int>(i);
        }
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
        std::printf("  %-8s %-6s atom %2zu  ATV %+.6f  oracle %+.6f  Δ %+.2e\n",
                    m.name, what, i + 1, got[i], want[i], got[i] - want[i]);
    }
    if (!(worst <= tol)) {
        std::fprintf(stderr, "FAIL: %s %s atom %d: |Δ| = %.3e > %.1e e\n",
                     m.name, what, worst_i + 1, worst, tol);
        return false;
    }
    std::printf("%s %s: max|Δ| = %.2e e (tol %.1e)\n", m.name, what, worst,
                tol);
    return true;
}

bool run_case(const Molecule& m) {
    bool ok = true;
    const int n = static_cast<int>(m.atoms.size());

    for (const char* set : {"bcc", "abcg2"}) {
        const bool is_bcc = std::string(set) == "bcc";
        const char* label = is_bcc ? "bcc" : "abcg2";

        auto frame = build(m);
        AM1 am1{m.net_charge};
        am1.density_tol = 1.0e-10;
        AM1BCCCharges model{am1, set};
        auto transport = molrs::frame_new();
        ATV::cpu::to_molpy(frame, *transport);
        auto molecule = molrs::atomistic_from_frame(*transport);
        auto q = model.assign(*molecule);

        if (is_bcc) {
            // Base stage: ATV AM1 Mulliken vs sqm's, before any averaging.
            ok &= check_per_atom("am1", m,
                                 model.last_am1_result().mulliken_charges,
                                 m.sqm_mulliken, kBaseTolE);
        }

        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            total += q[static_cast<std::size_t>(i)];
        }
        ok &=
            check_per_atom(label, m, q, is_bcc ? m.bcc : m.abcg2, kChargeTolE);

        // BCC increments are antisymmetric per bond, so Σq is the AM1 total.
        if (!(std::abs(total - m.net_charge) <= kNetChargeTolE)) {
            std::fprintf(stderr, "FAIL: %s %s Σq = %.9f, net charge %d\n",
                         m.name, label, total, m.net_charge);
            ok = false;
        }
        // antechamber -eq 1: topologically equivalent atoms share a charge.
        for (const auto& cls : m.equivalent) {
            for (int a : cls) {
                const double d =
                    std::abs(q[static_cast<std::size_t>(a)]
                             - q[static_cast<std::size_t>(cls[0])]);
                if (!(d <= kEquivalentTolE)) {
                    std::fprintf(stderr,
                                 "FAIL: %s %s atoms %d and %d are equivalent "
                                 "but differ by %.3e e\n",
                                 m.name, label, cls[0] + 1, a + 1, d);
                    ok = false;
                }
            }
        }
    }
    return ok;
}

} // namespace

int run() {
    bool ok = true;
    ok &= run_case(phenol());
    ok &= run_case(acetate());
    if (!ok)
        return 1;
    std::printf("PASS: AM1-BCC / ABCG2 phenol + acetate vs AmberTools 26.1 "
                "antechamber (Jakalian JCC 23, 1623 (2002))\n");
    return 0;
}

} // namespace atv_reg_semiempirical_am1bcc_antechamber_charges

ATV_REGRESSION("semiempirical/am1-bcc", "am1bcc-antechamber-charges",
               atv_reg_semiempirical_am1bcc_antechamber_charges::run);
