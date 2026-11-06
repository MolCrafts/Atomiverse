/// @file regressions/forcefield/gaff-fourier-cvff-lammps/reg_gaff-fourier-cvff-lammps.cpp
/// @brief GAFF2 proper torsions (`dihedral_style fourier`) and impropers
///        (`improper_style cvff`) of N-methylacetamide, ethylene carbonate
///        and toluene vs an offline LAMMPS single point.
///
/// Public headers only. Reads the data file and coefficient include through
/// `load_lammps_data` / `load_lammps_coeffs`, so the loaders' Impropers
/// section, Type Labels sections and fourier/cvff rows are exercised too.
///
/// oracle: lammps (30 Sep 2026 develop, a0e2bd6) — see REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_coeffs.h"
#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/cpu/potentials/dihedral_fourier.h"
#include "atomiverse/cpu/potentials/improper_cvff.h"
#include "harness/registry.h"

namespace atv_reg_forcefield_gaff_fourier_cvff_lammps {

using namespace ATV;

int run() {
    const std::string dir =
        std::string(ATV_REG_DATA_DIR) + "/gaff_fourier_cvff";

    // LAMMPS thermo line (kcal/mol): step pe edihed eimp.
    const double edihed_ref = 3.6499054964912958e+01;
    const double eimp_ref = 2.0257736233620535e-01;
    const double tol_E = 1e-10; // kcal/mol
    const double tol_F = 1e-10; // kcal/mol/Å

    cpu::Frame f = cpu::load_lammps_data(dir + "/system.data");
    const cpu::LammpsCoeffs ff = cpu::load_lammps_coeffs(dir + "/torsion.ff");
    if (!ff.has_fourier || !ff.has_improper_cvff) {
        std::printf("FAIL: torsion.ff lost its fourier/cvff styles\n");
        return 1;
    }

    // The coefficient file is LAMMPS `units real`; so is the run.
    cpu::Evaluate eval;
    eval.set_units("real");
    eval.set_evaluator(cpu::DihedralFourier{ff.dihedral_fourier});
    eval.set_evaluator(
        cpu::ImproperCvff{ff.improper_K, ff.improper_d, ff.improper_n});
    eval.build();
    eval.run(f);

    const double edihed = f.meta.get(kEnergyDihedral);
    const double eimp = f.meta.get(kEnergyImproper);
    std::printf("edihed = %.16e (ref %.16e)\neimp   = %.16e (ref %.16e)\n",
                edihed, edihed_ref, eimp, eimp_ref);
    bool ok = std::fabs(edihed - edihed_ref) <= tol_E
              && std::fabs(eimp - eimp_ref) <= tol_E;

    // Per-atom forces as LAMMPS printed them (write_dump, sorted by id).
    std::ifstream dump(dir + "/lammps_torsion_forces.dump");
    std::string line;
    while (std::getline(dump, line) && line.rfind("ITEM: ATOMS", 0) != 0) {
    }
    const auto tag = f.span(kAtom_tag);
    std::vector<int> row_of(static_cast<std::size_t>(f.nlocal()) + 1, -1);
    for (int i = 0; i < f.nlocal(); ++i)
        row_of[static_cast<std::size_t>(tag[i]) + 1] = i; // tag = id - 1
    double worst = 0.0;
    int rows = 0;
    while (std::getline(dump, line)) {
        std::istringstream iss(line);
        int id = 0;
        double ref[3];
        if (!(iss >> id >> ref[0] >> ref[1] >> ref[2]))
            continue;
        const int i = row_of.at(static_cast<std::size_t>(id));
        const double got[3] = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                               f.span(kAtom_fz)[i]};
        for (int d = 0; d < 3; ++d)
            worst = std::max(worst, std::fabs(got[d] - ref[d]));
        ++rows;
    }
    std::printf("forces: %d rows, worst |dF| = %.3e kcal/mol/A (tol %.0e)\n",
                rows, worst, tol_F);
    ok = ok && rows == f.nlocal() && worst <= tol_F;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

} // namespace atv_reg_forcefield_gaff_fourier_cvff_lammps

ATV_REGRESSION("forcefield/gaff", "gaff-fourier-cvff-lammps",
               atv_reg_forcefield_gaff_fourier_cvff_lammps::run);
