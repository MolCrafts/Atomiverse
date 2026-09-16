/// @file cpu/scf/basis_data/ecp_def2.h
/// @brief def2-svp effective-core-potential table (auto-generated;
///        do not edit).
///
/// Emitted by `tools/gen_basis_data.py` from PySCF's
/// `gto.basis.load_ecp("def2-svp", El)` data, so numerical values
/// match the PySCF ground-truth path bit-for-bit. Channel storage
/// follows the PySCF/def2 convention: projector channels hold
/// U_l - U_L; the LAST channel of each AtomEcp is the local part U_L
/// (PySCF l = -1, relabeled to l = L = 1 + max projector l).
/// Exponents/coefficients in atomic units (Bohr^-2 / Hartree).
///
/// Supported elements: Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn.
/// All other Z (def2 convention: Z <= 36) are all-electron — their
/// rows are empty AtomEcp entries.
///
/// Regenerate:
///   python tools/gen_basis_data.py --ecp def2-svp
#pragma once

#include <vector>

#include "atomiverse/cpu/scf/basis_data/basis_table.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace basis_data {

/// @brief Get the def2-svp ECP table, indexed by atomic number Z.
/// @return Reference to a static table with AtomEcp entries for
///         Z=1..86; position 0 and any all-electron Z are empty.
inline const std::vector<AtomEcp>& ecp_def2() {
    static const std::vector<AtomEcp> data = []() {
        return std::vector<AtomEcp>{
            AtomEcp{}, // Z=0 unused
            AtomEcp{}, // Z=1 all-electron
            AtomEcp{}, // Z=2 all-electron
            AtomEcp{}, // Z=3 all-electron
            AtomEcp{}, // Z=4 all-electron
            AtomEcp{}, // Z=5 all-electron
            AtomEcp{}, // Z=6 all-electron
            AtomEcp{}, // Z=7 all-electron
            AtomEcp{}, // Z=8 all-electron
            AtomEcp{}, // Z=9 all-electron
            AtomEcp{}, // Z=10 all-electron
            AtomEcp{}, // Z=11 all-electron
            AtomEcp{}, // Z=12 all-electron
            AtomEcp{}, // Z=13 all-electron
            AtomEcp{}, // Z=14 all-electron
            AtomEcp{}, // Z=15 all-electron
            AtomEcp{}, // Z=16 all-electron
            AtomEcp{}, // Z=17 all-electron
            AtomEcp{}, // Z=18 all-electron
            AtomEcp{}, // Z=19 all-electron
            AtomEcp{}, // Z=20 all-electron
            AtomEcp{}, // Z=21 all-electron
            AtomEcp{}, // Z=22 all-electron
            AtomEcp{}, // Z=23 all-electron
            AtomEcp{}, // Z=24 all-electron
            AtomEcp{}, // Z=25 all-electron
            AtomEcp{}, // Z=26 all-electron
            AtomEcp{}, // Z=27 all-electron
            AtomEcp{}, // Z=28 all-electron
            AtomEcp{}, // Z=29 all-electron
            AtomEcp{}, // Z=30 all-electron
            AtomEcp{}, // Z=31 all-electron
            AtomEcp{}, // Z=32 all-electron
            AtomEcp{}, // Z=33 all-electron
            AtomEcp{}, // Z=34 all-electron
            AtomEcp{}, // Z=35 all-electron
            AtomEcp{}, // Z=36 all-electron
            // Z=37 (Rb): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 5.036551, 89.500198},
                                       EcpGaussianTerm{2, 1.970849, 0.493761},
                                       EcpGaussianTerm{2, 3.843114, 12.3169},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 4.258341, 58.568974},
                                       EcpGaussianTerm{2, 1.470709, 0.431791},
                                       EcpGaussianTerm{2, 3.843114, 12.3169},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 3.023127, 26.224898},
                                       EcpGaussianTerm{2, 0.650383, 0.962839},
                                       EcpGaussianTerm{2, 3.843114, 12.3169},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 3.843114, -12.3169},
                                   }},
                    }},
            // Z=38 (Sr): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 7.400074, 135.47943},
                                       EcpGaussianTerm{2, 3.606379, 17.534463},
                                       EcpGaussianTerm{2, 4.633975, 15.805992},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 6.484868, 88.359709},
                                       EcpGaussianTerm{2, 3.288053, 15.394372},
                                       EcpGaussianTerm{2, 4.633975, 15.805992},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 4.622841, 29.888987},
                                       EcpGaussianTerm{2, 2.246904, 6.659414},
                                       EcpGaussianTerm{2, 4.633975, 15.805992},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 4.633975, -15.805992},
                                   }},
                    }},
            // Z=39 (Y): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 7.4880494, 135.15384412},
                                   EcpGaussianTerm{2, 3.7440247, 15.5524413},
                                   EcpGaussianTerm{2, 6.584212, 19.12219811},
                                   EcpGaussianTerm{2, 3.292106, 2.43637543},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 6.4453772, 87.78499167},
                                   EcpGaussianTerm{2, 3.2226886, 11.56406599},
                                   EcpGaussianTerm{2, 6.584212, 19.12219811},
                                   EcpGaussianTerm{2, 3.292106, 2.43637543},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 4.6584472, 29.70100072},
                                   EcpGaussianTerm{2, 2.3292236, 5.53996847},
                                   EcpGaussianTerm{2, 6.584212, 19.12219811},
                                   EcpGaussianTerm{2, 3.292106, 2.43637543},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 6.584212, -19.12219811},
                                   EcpGaussianTerm{2, 3.292106, -2.43637543},
                               }},
                }},
            // Z=40 (Zr): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 8.2, 150.26759106},
                                   EcpGaussianTerm{2, 4.0897278, 18.9762165},
                                   EcpGaussianTerm{2, 7.54, 21.09377605},
                                   EcpGaussianTerm{2, 3.77, 3.08069427},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 7.11, 99.62212372},
                                   EcpGaussianTerm{2, 3.596798, 14.16873329},
                                   EcpGaussianTerm{2, 7.54, 21.09377605},
                                   EcpGaussianTerm{2, 3.77, 3.08069427},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 5.35, 35.04512355},
                                   EcpGaussianTerm{2, 2.4918215, 6.11125948},
                                   EcpGaussianTerm{2, 7.54, 21.09377605},
                                   EcpGaussianTerm{2, 3.77, 3.08069427},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 7.54, -21.09377605},
                                   EcpGaussianTerm{2, 3.77, -3.08069427},
                               }},
                }},
            // Z=41 (Nb): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 8.9, 165.17914349},
                                       EcpGaussianTerm{2, 4.43, 21.99297437},
                                       EcpGaussianTerm{2, 8.49, 22.92954996},
                                       EcpGaussianTerm{2, 4.25, 3.66630986},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 7.77, 111.79441445},
                                       EcpGaussianTerm{2, 3.96, 16.63348326},
                                       EcpGaussianTerm{2, 8.49, 22.92954996},
                                       EcpGaussianTerm{2, 4.25, 3.66630986},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 6.05, 38.1122488},
                                       EcpGaussianTerm{2, 2.84, 8.03916727},
                                       EcpGaussianTerm{2, 8.49, 22.92954996},
                                       EcpGaussianTerm{2, 4.25, 3.66630986},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 8.49, -22.92954996},
                                       EcpGaussianTerm{2, 4.25, -3.66630986},
                                   }},
                    }},
            // Z=42 (Mo): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 9.7145938, 180.1031085},
                                   EcpGaussianTerm{2, 4.6805004, 24.99722791},
                                   EcpGaussianTerm{2, 9.45, 24.80517707},
                                   EcpGaussianTerm{2, 4.72, 4.15378155},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 8.1421366, 123.77275231},
                                   EcpGaussianTerm{2, 4.6259863, 19.530228},
                                   EcpGaussianTerm{2, 9.45, 24.80517707},
                                   EcpGaussianTerm{2, 4.72, 4.15378155},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 6.6184148, 48.37502229},
                                   EcpGaussianTerm{2, 3.2487516, 8.89205274},
                                   EcpGaussianTerm{2, 9.45, 24.80517707},
                                   EcpGaussianTerm{2, 4.72, 4.15378155},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 9.45, -24.80517707},
                                   EcpGaussianTerm{2, 4.72, -4.15378155},
                               }},
                }},
            // Z=43 (Tc): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 10.4223462, 195.15916591},
                                   EcpGaussianTerm{2, 5.036516, 28.09260333},
                                   EcpGaussianTerm{2, 10.4, 26.56244747},
                                   EcpGaussianTerm{2, 5.2, 4.58568054},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 8.9504494, 135.28456622},
                                   EcpGaussianTerm{2, 4.8544394, 21.8065043},
                                   EcpGaussianTerm{2, 10.4, 26.56244747},
                                   EcpGaussianTerm{2, 5.2, 4.58568054},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 6.9456968, 54.32972942},
                                   EcpGaussianTerm{2, 3.9705849, 11.15506795},
                                   EcpGaussianTerm{2, 10.4, 26.56244747},
                                   EcpGaussianTerm{2, 5.2, 4.58568054},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 10.4, -26.56244747},
                                   EcpGaussianTerm{2, 5.2, -4.58568054},
                               }},
                }},
            // Z=44 (Ru): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 11.1052693, 209.82297122},
                                   EcpGaussianTerm{2, 5.4147454, 30.65472642},
                                   EcpGaussianTerm{2, 11.36, 28.34061627},
                                   EcpGaussianTerm{2, 5.68, 4.94462923},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 9.7712707, 146.33618228},
                                   EcpGaussianTerm{2, 5.0739908, 24.12787723},
                                   EcpGaussianTerm{2, 11.36, 28.34061627},
                                   EcpGaussianTerm{2, 5.68, 4.94462923},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 7.6714231, 67.51589667},
                                   EcpGaussianTerm{2, 4.1365647, 9.87010415},
                                   EcpGaussianTerm{2, 11.36, 28.34061627},
                                   EcpGaussianTerm{2, 5.68, 4.94462923},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 11.36, -28.34061627},
                                   EcpGaussianTerm{2, 5.68, -4.94462923},
                               }},
                }},
            // Z=45 (Rh): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 11.72, 225.34775353},
                                       EcpGaussianTerm{2, 5.82, 32.82318898},
                                       EcpGaussianTerm{2, 12.31, 30.09345572},
                                       EcpGaussianTerm{2, 6.16, 5.21848192},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 10.42, 158.70941159},
                                       EcpGaussianTerm{2, 5.45, 26.44410049},
                                       EcpGaussianTerm{2, 12.31, 30.09345572},
                                       EcpGaussianTerm{2, 6.16, 5.21848192},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 8.82, 62.75862572},
                                       EcpGaussianTerm{2, 3.87, 10.97871947},
                                       EcpGaussianTerm{2, 12.31, 30.09345572},
                                       EcpGaussianTerm{2, 6.16, 5.21848192},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 12.31, -30.09345572},
                                       EcpGaussianTerm{2, 6.16, -5.21848192},
                                   }},
                    }},
            // Z=46 (Pd): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 12.43, 240.22904033},
                                   EcpGaussianTerm{2, 6.1707594, 35.17194347},
                                   EcpGaussianTerm{2, 13.27, 31.92955431},
                                   EcpGaussianTerm{2, 6.63, 5.39821694},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 11.08, 170.41727605},
                                   EcpGaussianTerm{2, 5.8295541, 28.47213287},
                                   EcpGaussianTerm{2, 13.27, 31.92955431},
                                   EcpGaussianTerm{2, 6.63, 5.39821694},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 9.51, 69.01384488},
                                   EcpGaussianTerm{2, 4.1397811, 11.75086158},
                                   EcpGaussianTerm{2, 13.27, 31.92955431},
                                   EcpGaussianTerm{2, 6.63, 5.39821694},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 13.27, -31.92955431},
                                   EcpGaussianTerm{2, 6.63, -5.39821694},
                               }},
                }},
            // Z=47 (Ag): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 13.13, 255.13936452},
                                       EcpGaussianTerm{2, 6.51, 36.86612154},
                                       EcpGaussianTerm{2, 14.22, 33.68992012},
                                       EcpGaussianTerm{2, 7.11, 5.53112021},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 11.74, 182.18186871},
                                       EcpGaussianTerm{2, 6.2, 30.35775148},
                                       EcpGaussianTerm{2, 14.22, 33.68992012},
                                       EcpGaussianTerm{2, 7.11, 5.53112021},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 10.21, 73.71926087},
                                       EcpGaussianTerm{2, 4.38, 12.50211712},
                                       EcpGaussianTerm{2, 14.22, 33.68992012},
                                       EcpGaussianTerm{2, 7.11, 5.53112021},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 14.22, -33.68992012},
                                       EcpGaussianTerm{2, 7.11, -5.53112021},
                                   }},
                    }},
            // Z=48 (Cd): n_core=28, 3 projector channels + local
            AtomEcp{
                28,
                {
                    EcpChannel{0,
                               {
                                   EcpGaussianTerm{2, 13.8358689, 270.00948324},
                                   EcpGaussianTerm{2, 6.8572704, 38.76730798},
                                   EcpGaussianTerm{2, 15.1847957, 35.47662555},
                                   EcpGaussianTerm{2, 7.5923978, 5.61767685},
                               }},
                    EcpChannel{1,
                               {
                                   EcpGaussianTerm{2, 12.404971, 193.82962939},
                                   EcpGaussianTerm{2, 6.5677995, 31.89652523},
                                   EcpGaussianTerm{2, 15.1847957, 35.47662555},
                                   EcpGaussianTerm{2, 7.5923978, 5.61767685},
                               }},
                    EcpChannel{2,
                               {
                                   EcpGaussianTerm{2, 10.8969253, 79.193647},
                                   EcpGaussianTerm{2, 4.6411649, 13.23082674},
                                   EcpGaussianTerm{2, 15.1847957, 35.47662555},
                                   EcpGaussianTerm{2, 7.5923978, 5.61767685},
                               }},
                    EcpChannel{3,
                               {
                                   // local U_L
                                   EcpGaussianTerm{2, 15.1847957, -35.47662555},
                                   EcpGaussianTerm{2, 7.5923978, -5.61767685},
                               }},
                }},
            // Z=49 (In): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 15.392822, 281.12235},
                                       EcpGaussianTerm{2, 8.055864, 61.90147},
                                       EcpGaussianTerm{2, 12.539056, 13.728078},
                                       EcpGaussianTerm{2, 12.552561, 18.206866},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 13.928672, 67.462154},
                                       EcpGaussianTerm{2, 13.347234, 134.94925},
                                       EcpGaussianTerm{2, 7.614132, 14.74614},
                                       EcpGaussianTerm{2, 7.318365, 29.639262},
                                       EcpGaussianTerm{2, 12.539056, 13.728078},
                                       EcpGaussianTerm{2, 12.552561, 18.206866},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 14.034715, 35.493254},
                                       EcpGaussianTerm{2, 14.511616, 53.178773},
                                       EcpGaussianTerm{2, 5.55055, 9.177281},
                                       EcpGaussianTerm{2, 5.059415, 12.39241},
                                       EcpGaussianTerm{2, 12.539056, 13.728078},
                                       EcpGaussianTerm{2, 12.552561, 18.206866},
                                   }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 12.539056, -13.728078},
                                EcpGaussianTerm{2, 12.552561, -18.206866},
                            }},
                    }},
            // Z=50 (Sn): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 17.420414, 279.988682},
                                EcpGaussianTerm{2, 7.631155, 62.37781},
                                EcpGaussianTerm{2, 12.282348, 12.576333},
                                EcpGaussianTerm{2, 12.27215, 16.595944},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 16.131024, 66.162523},
                                EcpGaussianTerm{2, 15.628077, 132.174396},
                                EcpGaussianTerm{2, 7.325608, 16.339417},
                                EcpGaussianTerm{2, 6.942519, 32.488959},
                                EcpGaussianTerm{2, 12.282348, 12.576333},
                                EcpGaussianTerm{2, 12.27215, 16.595944},
                            }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 15.514976, 36.387441},
                                       EcpGaussianTerm{2, 15.18816, 54.507841},
                                       EcpGaussianTerm{2, 5.456024, 8.696823},
                                       EcpGaussianTerm{2, 5.363105, 12.840208},
                                       EcpGaussianTerm{2, 12.282348, 12.576333},
                                       EcpGaussianTerm{2, 12.27215, 16.595944},
                                   }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 12.282348, -12.576333},
                                EcpGaussianTerm{2, 12.27215, -16.595944},
                            }},
                    }},
            // Z=51 (Sb): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 16.330865, 281.071581},
                                EcpGaussianTerm{2, 8.556542, 61.716604},
                                EcpGaussianTerm{2, 14.444978, 15.366801},
                                EcpGaussianTerm{2, 14.449295, 20.296138},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 14.470337, 67.45738},
                                EcpGaussianTerm{2, 13.816194, 134.933503},
                                EcpGaussianTerm{2, 8.424924, 14.716344},
                                EcpGaussianTerm{2, 8.092728, 29.518512},
                                EcpGaussianTerm{2, 14.444978, 15.366801},
                                EcpGaussianTerm{2, 14.449295, 20.296138},
                            }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 14.886331, 35.447815},
                                       EcpGaussianTerm{2, 15.146319, 53.143466},
                                       EcpGaussianTerm{2, 5.908267, 9.179223},
                                       EcpGaussianTerm{2, 5.594322, 13.240253},
                                       EcpGaussianTerm{2, 14.444978, 15.366801},
                                       EcpGaussianTerm{2, 14.449295, 20.296138},
                                   }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 14.444978, -15.366801},
                                EcpGaussianTerm{2, 14.449295, -20.296138},
                            }},
                    }},
            // Z=52 (Te): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 16.814473, 281.045843},
                                EcpGaussianTerm{2, 8.793526, 61.620656},
                                EcpGaussianTerm{2, 15.206168, 15.74545},
                                EcpGaussianTerm{2, 15.201702, 20.742448},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 14.877801, 67.449464},
                                EcpGaussianTerm{2, 14.269731, 134.904304},
                                EcpGaussianTerm{2, 8.724435, 14.689547},
                                EcpGaussianTerm{2, 8.291515, 29.415063},
                                EcpGaussianTerm{2, 15.206168, 15.74545},
                                EcpGaussianTerm{2, 15.201702, 20.742448},
                            }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 15.205008, 35.432057},
                                       EcpGaussianTerm{2, 15.225848, 53.135687},
                                       EcpGaussianTerm{2, 6.071769, 9.069802},
                                       EcpGaussianTerm{2, 5.80476, 13.122304},
                                       EcpGaussianTerm{2, 15.206168, 15.74545},
                                       EcpGaussianTerm{2, 15.201702, 20.742448},
                                   }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 15.206168, -15.74545},
                                EcpGaussianTerm{2, 15.201702, -20.742448},
                            }},
                    }},
            // Z=53 (I): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 40.015835, 49.994293},
                                EcpGaussianTerm{2, 17.429747, 281.025317},
                                EcpGaussianTerm{2, 9.005484, 61.573326},
                                EcpGaussianTerm{2, 19.458609, 21.84204},
                                EcpGaussianTerm{2, 19.34926, 28.468191},
                                EcpGaussianTerm{2, 4.823767, 0.243713},
                                EcpGaussianTerm{2, 4.884315, 0.320804},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 15.355466, 67.442841},
                                EcpGaussianTerm{2, 14.971833, 134.881137},
                                EcpGaussianTerm{2, 8.960164, 14.675051},
                                EcpGaussianTerm{2, 8.259096, 29.375666},
                                EcpGaussianTerm{2, 19.458609, 21.84204},
                                EcpGaussianTerm{2, 19.34926, 28.468191},
                                EcpGaussianTerm{2, 4.823767, 0.243713},
                                EcpGaussianTerm{2, 4.884315, 0.320804},
                            }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 15.068908, 35.439529},
                                       EcpGaussianTerm{2, 14.555322, 53.176057},
                                       EcpGaussianTerm{2, 6.718647, 9.067195},
                                       EcpGaussianTerm{2, 6.456393, 13.206937},
                                       EcpGaussianTerm{2, 1.191779, 0.089335},
                                       EcpGaussianTerm{2, 1.291157, 0.05238},
                                       EcpGaussianTerm{2, 19.458609, 21.84204},
                                       EcpGaussianTerm{2, 19.34926, 28.468191},
                                       EcpGaussianTerm{2, 4.823767, 0.243713},
                                       EcpGaussianTerm{2, 4.884315, 0.320804},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 19.458609, -21.84204},
                                       EcpGaussianTerm{2, 19.34926, -28.468191},
                                       EcpGaussianTerm{2, 4.823767, -0.243713},
                                       EcpGaussianTerm{2, 4.884315, -0.320804},
                                   }},
                    }},
            // Z=54 (Xe): n_core=28, 3 projector channels + local
            AtomEcp{28,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 40.005184, 49.997962},
                                EcpGaussianTerm{2, 17.812214, 281.013303},
                                EcpGaussianTerm{2, 9.30415, 61.538255},
                                EcpGaussianTerm{2, 20.881557, 23.089295},
                                EcpGaussianTerm{2, 20.783443, 30.074475},
                                EcpGaussianTerm{2, 5.253389, 0.288227},
                                EcpGaussianTerm{2, 5.361188, 0.386924},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 15.701772, 67.439142},
                                EcpGaussianTerm{2, 15.258608, 134.874711},
                                EcpGaussianTerm{2, 9.292184, 14.6633},
                                EcpGaussianTerm{2, 8.559003, 29.35473},
                                EcpGaussianTerm{2, 20.881557, 23.089295},
                                EcpGaussianTerm{2, 20.783443, 30.074475},
                                EcpGaussianTerm{2, 5.253389, 0.288227},
                                EcpGaussianTerm{2, 5.361188, 0.386924},
                            }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 15.1856, 35.436908},
                                       EcpGaussianTerm{2, 14.2845, 53.195772},
                                       EcpGaussianTerm{2, 7.121889, 9.046232},
                                       EcpGaussianTerm{2, 6.991963, 13.223681},
                                       EcpGaussianTerm{2, 0.623946, 0.084853},
                                       EcpGaussianTerm{2, 0.647284, 0.044155},
                                       EcpGaussianTerm{2, 20.881557, 23.089295},
                                       EcpGaussianTerm{2, 20.783443, 30.074475},
                                       EcpGaussianTerm{2, 5.253389, 0.288227},
                                       EcpGaussianTerm{2, 5.361188, 0.386924},
                                   }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 20.881557, -23.089295},
                                EcpGaussianTerm{2, 20.783443, -30.074475},
                                EcpGaussianTerm{2, 5.253389, -0.288227},
                                EcpGaussianTerm{2, 5.361188, -0.386924},
                            }},
                    }},
            // Z=55 (Cs): n_core=46, 3 projector channels + local
            AtomEcp{46,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 4.07975, 84.54773},
                                       EcpGaussianTerm{2, 2.417406, 16.654173},
                                       EcpGaussianTerm{2, 3.123269, 28.884309},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 5.51408, 157.049059},
                                       EcpGaussianTerm{2, 2.160316, 26.423307},
                                       EcpGaussianTerm{2, 3.123269, 28.884309},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 1.80741, 13.172753},
                                       EcpGaussianTerm{2, 0.858182, 3.342833},
                                       EcpGaussianTerm{2, 3.123269, 28.884309},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 3.123269, -28.884309},
                                   }},
                    }},
            // Z=56 (Ba): n_core=46, 3 projector channels + local
            AtomEcp{46,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 9.526986, 427.845816},
                                       EcpGaussianTerm{2, 4.48751, 204.41753},
                                       EcpGaussianTerm{2, 3.589465, 33.473174},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 8.31593, 293.605864},
                                       EcpGaussianTerm{2, 4.292217, 294.193316},
                                       EcpGaussianTerm{2, 3.589465, 33.473174},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 5.916108, 112.550402},
                                       EcpGaussianTerm{2, 2.874842, 181.782621},
                                       EcpGaussianTerm{2, 3.589465, 33.473174},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 3.589465, -33.473174},
                                   }},
                    }},
            // Z=57 (La): n_core=46, 3 projector channels + local
            AtomEcp{46,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 3.3099, 91.932177},
                                       EcpGaussianTerm{2, 1.655, -3.788764},
                                       EcpGaussianTerm{2, 4.0286, 36.010016},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 2.8368, 63.759486},
                                       EcpGaussianTerm{2, 1.4184, -0.647958},
                                       EcpGaussianTerm{2, 4.0286, 36.010016},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 2.0213, 36.116173},
                                       EcpGaussianTerm{2, 1.0107, 0.219114},
                                       EcpGaussianTerm{2, 4.0286, 36.010016},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 4.0286, -36.010016},
                                   }},
                    }},
            AtomEcp{}, // Z=58 all-electron
            AtomEcp{}, // Z=59 all-electron
            AtomEcp{}, // Z=60 all-electron
            AtomEcp{}, // Z=61 all-electron
            AtomEcp{}, // Z=62 all-electron
            AtomEcp{}, // Z=63 all-electron
            AtomEcp{}, // Z=64 all-electron
            AtomEcp{}, // Z=65 all-electron
            AtomEcp{}, // Z=66 all-electron
            AtomEcp{}, // Z=67 all-electron
            AtomEcp{}, // Z=68 all-electron
            AtomEcp{}, // Z=69 all-electron
            AtomEcp{}, // Z=70 all-electron
            AtomEcp{}, // Z=71 all-electron
            // Z=72 (Hf): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 14.769959, 1499.28471073},
                                EcpGaussianTerm{2, 7.3849794, 40.28210136},
                                EcpGaussianTerm{2, 1.78576984, -10.04672251},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 9.8494895, 397.73300533},
                                EcpGaussianTerm{2, 4.9247445, 19.31640586},
                                EcpGaussianTerm{2, 1.78576984, -10.04672251},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 6.0967564, 101.32980526},
                                EcpGaussianTerm{2, 3.0483782, 5.87343821},
                                EcpGaussianTerm{2, 1.78576984, -10.04672251},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 1.78576984, 10.04672251},
                            }},
                    }},
            // Z=73 (Ta): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 14.5464077, 1345.88064703},
                                EcpGaussianTerm{2, 7.2732038, 36.7668062},
                                EcpGaussianTerm{2, 2.01788111, -12.01796094},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 9.93556529, 378.42530145},
                                EcpGaussianTerm{2, 4.96778243, 22.29309086},
                                EcpGaussianTerm{2, 2.01788111, -12.01796094},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 6.34737691, 104.88395571},
                                EcpGaussianTerm{2, 3.17368846, 8.75584805},
                                EcpGaussianTerm{2, 2.01788111, -12.01796094},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 2.01788111, 12.01796094},
                            }},
                    }},
            // Z=74 (W): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 14.3228564, 1192.39588226},
                                EcpGaussianTerm{2, 7.1614281, 32.52293315},
                                EcpGaussianTerm{2, 2.25888846, -14.15257947},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.0216411, 359.03196711},
                                EcpGaussianTerm{2, 5.0108204, 24.03038019},
                                EcpGaussianTerm{2, 2.25888846, -14.15257947},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 6.59799743, 108.30134897},
                                EcpGaussianTerm{2, 3.29899871, 10.98252827},
                                EcpGaussianTerm{2, 2.25888846, -14.15257947},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 2.25888846, 14.15257947},
                            }},
                    }},
            // Z=75 (Re): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 14.0993051, 1038.95157226},
                                EcpGaussianTerm{2, 7.0496525, 29.5617383},
                                EcpGaussianTerm{2, 2.50865059, -16.44985227},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.1077169, 339.54350965},
                                EcpGaussianTerm{2, 5.0538583, 24.91369646},
                                EcpGaussianTerm{2, 2.50865059, -16.44985227},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 6.84861794, 111.69965275},
                                EcpGaussianTerm{2, 3.42430897, 12.62432927},
                                EcpGaussianTerm{2, 2.50865059, -16.44985227},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 2.50865059, 16.44985227},
                            }},
                    }},
            // Z=76 (Os): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 13.8757539, 885.40571914},
                                EcpGaussianTerm{2, 6.9378769, 25.96704014},
                                EcpGaussianTerm{2, 2.7670751, -18.90945701},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.1937926, 320.08390185},
                                EcpGaussianTerm{2, 5.0968962, 26.14876493},
                                EcpGaussianTerm{2, 2.7670751, -18.90945701},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 7.09923846, 115.04484313},
                                EcpGaussianTerm{2, 3.54961923, 13.62257457},
                                EcpGaussianTerm{2, 2.7670751, -18.90945701},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 2.7670751, 18.90945701},
                            }},
                    }},
            // Z=77 (Ir): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 13.6522026, 732.26919978},
                                EcpGaussianTerm{2, 6.8261013, 26.48472087},
                                EcpGaussianTerm{2, 3.03407192, -21.53103107},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.2798684, 299.48947357},
                                EcpGaussianTerm{2, 5.1399341, 26.46623354},
                                EcpGaussianTerm{2, 3.03407192, -21.53103107},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 7.34985897, 124.45759451},
                                EcpGaussianTerm{2, 3.67492949, 14.03599518},
                                EcpGaussianTerm{2, 3.03407192, -21.53103107},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 3.03407192, 21.53103107},
                            }},
                    }},
            // Z=78 (Pt): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 13.4286513, 579.22386092},
                                EcpGaussianTerm{2, 6.7143256, 29.66949062},
                                EcpGaussianTerm{2, 3.30956857, -24.31437573},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.3659442, 280.86077422},
                                EcpGaussianTerm{2, 5.1829721, 26.74538204},
                                EcpGaussianTerm{2, 3.30956857, -24.31437573},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 7.60047949, 120.39644429},
                                EcpGaussianTerm{2, 3.80023974, 15.81092058},
                                EcpGaussianTerm{2, 3.30956857, -24.31437573},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 3.30956857, 24.31437573},
                            }},
                    }},
            // Z=79 (Au): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 13.2051, 426.8466792},
                                       EcpGaussianTerm{2, 6.60255, 37.00708285},
                                       EcpGaussianTerm{2, 4.78982, -30.4900889},
                                       EcpGaussianTerm{2, 2.39491, -5.17107381},
                                   }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.45202, 261.19958038},
                                EcpGaussianTerm{2, 5.22601, 26.96249604},
                                EcpGaussianTerm{2, 4.78982, -30.4900889},
                                EcpGaussianTerm{2, 2.39491, -5.17107381},
                            }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 7.8511, 124.79066561},
                                       EcpGaussianTerm{2, 3.92555, 16.30072573},
                                       EcpGaussianTerm{2, 4.78982, -30.4900889},
                                       EcpGaussianTerm{2, 2.39491, -5.17107381},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 4.78982, 30.4900889},
                                       EcpGaussianTerm{2, 2.39491, 5.17107381},
                                   }},
                    }},
            // Z=80 (Hg): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 12.9815487, 275.73721174},
                                EcpGaussianTerm{2, 6.4907744, 49.08921249},
                                EcpGaussianTerm{2, 3.88579112, -30.36499643},
                            }},
                        EcpChannel{
                            1,
                            {
                                EcpGaussianTerm{2, 10.5380958, 241.54007398},
                                EcpGaussianTerm{2, 5.2690479, 27.39659081},
                                EcpGaussianTerm{2, 3.88579112, -30.36499643},
                            }},
                        EcpChannel{
                            2,
                            {
                                EcpGaussianTerm{2, 8.10172051, 127.86700761},
                                EcpGaussianTerm{2, 4.05086026, 16.60831151},
                                EcpGaussianTerm{2, 3.88579112, -30.36499643},
                            }},
                        EcpChannel{
                            3,
                            {
                                // local U_L
                                EcpGaussianTerm{2, 3.88579112, 30.36499643},
                            }},
                    }},
            // Z=81 (Tl): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 12.167805, 281.284663},
                                EcpGaussianTerm{2, 8.294909, 62.434251},
                                EcpGaussianTerm{2, 5.626399, -15.825488},
                                EcpGaussianTerm{2, 5.548952, -21.104021},
                                EcpGaussianTerm{2, 2.874946, -2.915127},
                                EcpGaussianTerm{2, 2.821451, -3.896903},
                            }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 7.151492, 4.633408},
                                       EcpGaussianTerm{2, 5.172865, 9.341756},
                                       EcpGaussianTerm{2, 9.891072, 72.299253},
                                       EcpGaussianTerm{2, 9.003391, 144.558037},
                                       EcpGaussianTerm{2, 5.626399, -15.825488},
                                       EcpGaussianTerm{2, 5.548952, -21.104021},
                                       EcpGaussianTerm{2, 2.874946, -2.915127},
                                       EcpGaussianTerm{2, 2.821451, -3.896903},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 7.130218, 35.943039},
                                       EcpGaussianTerm{2, 6.926906, 53.909593},
                                       EcpGaussianTerm{2, 5.41757, 10.381939},
                                       EcpGaussianTerm{2, 5.138681, 15.583822},
                                       EcpGaussianTerm{2, 5.626399, -15.825488},
                                       EcpGaussianTerm{2, 5.548952, -21.104021},
                                       EcpGaussianTerm{2, 2.874946, -2.915127},
                                       EcpGaussianTerm{2, 2.821451, -3.896903},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 5.626399, 15.825488},
                                       EcpGaussianTerm{2, 5.548952, 21.104021},
                                       EcpGaussianTerm{2, 2.874946, 2.915127},
                                       EcpGaussianTerm{2, 2.821451, 3.896903},
                                   }},
                    }},
            // Z=82 (Pb): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 12.296303, 281.285499},
                                EcpGaussianTerm{2, 8.632634, 62.520217},
                                EcpGaussianTerm{2, 3.887512, -12.209892},
                                EcpGaussianTerm{2, 3.811963, -16.190291},
                            }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 10.24179, 72.276897},
                                       EcpGaussianTerm{2, 8.924176, 144.591083},
                                       EcpGaussianTerm{2, 6.581342, 4.758693},
                                       EcpGaussianTerm{2, 6.255403, 9.940621},
                                       EcpGaussianTerm{2, 3.887512, -12.209892},
                                       EcpGaussianTerm{2, 3.811963, -16.190291},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 7.754336, 35.848507},
                                       EcpGaussianTerm{2, 7.720281, 53.724342},
                                       EcpGaussianTerm{2, 4.970264, 10.115256},
                                       EcpGaussianTerm{2, 4.563789, 14.833731},
                                       EcpGaussianTerm{2, 3.887512, -12.209892},
                                       EcpGaussianTerm{2, 3.811963, -16.190291},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 3.887512, 12.209892},
                                       EcpGaussianTerm{2, 3.811963, 16.190291},
                                   }},
                    }},
            // Z=83 (Bi): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 13.04309, 283.264227},
                                       EcpGaussianTerm{2, 8.221682, 62.471959},
                                       EcpGaussianTerm{2, 4.214546, -13.713383},
                                       EcpGaussianTerm{2, 4.1334, -18.194308},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 10.467777, 72.001499},
                                       EcpGaussianTerm{2, 9.118901, 144.002277},
                                       EcpGaussianTerm{2, 6.754791, 5.007945},
                                       EcpGaussianTerm{2, 6.252592, 9.99155},
                                       EcpGaussianTerm{2, 4.214546, -13.713383},
                                       EcpGaussianTerm{2, 4.1334, -18.194308},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 8.081474, 36.396259},
                                       EcpGaussianTerm{2, 7.890595, 54.597664},
                                       EcpGaussianTerm{2, 4.955556, 9.984294},
                                       EcpGaussianTerm{2, 4.704559, 14.981485},
                                       EcpGaussianTerm{2, 4.214546, -13.713383},
                                       EcpGaussianTerm{2, 4.1334, -18.194308},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 4.214546, 13.713383},
                                       EcpGaussianTerm{2, 4.1334, 18.194308},
                                   }},
                    }},
            // Z=84 (Po): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 13.277227, 283.244706},
                                EcpGaussianTerm{2, 8.399518, 62.396461},
                                EcpGaussianTerm{2, 5.01327, -17.428295},
                                EcpGaussianTerm{2, 4.98464, -23.380353},
                                EcpGaussianTerm{2, 1.32676, -0.163392},
                                EcpGaussianTerm{2, 1.528758, -0.324566},
                            }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 10.665682, 71.991716},
                                       EcpGaussianTerm{2, 9.283753, 143.971871},
                                       EcpGaussianTerm{2, 6.872749, 4.949615},
                                       EcpGaussianTerm{2, 6.32615, 9.740499},
                                       EcpGaussianTerm{2, 5.01327, -17.428295},
                                       EcpGaussianTerm{2, 4.98464, -23.380353},
                                       EcpGaussianTerm{2, 1.32676, -0.163392},
                                       EcpGaussianTerm{2, 1.528758, -0.324566},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 8.214866, 36.378383},
                                       EcpGaussianTerm{2, 8.008696, 54.562715},
                                       EcpGaussianTerm{2, 5.055227, 9.889499},
                                       EcpGaussianTerm{2, 4.782553, 14.693877},
                                       EcpGaussianTerm{2, 5.01327, -17.428295},
                                       EcpGaussianTerm{2, 4.98464, -23.380353},
                                       EcpGaussianTerm{2, 1.32676, -0.163392},
                                       EcpGaussianTerm{2, 1.528758, -0.324566},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 5.01327, 17.428295},
                                       EcpGaussianTerm{2, 4.98464, 23.380353},
                                       EcpGaussianTerm{2, 1.32676, 0.163392},
                                       EcpGaussianTerm{2, 1.528758, 0.324566},
                                   }},
                    }},
            // Z=85 (At): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{
                            0,
                            {
                                EcpGaussianTerm{2, 30.200832, 49.957158},
                                EcpGaussianTerm{2, 13.612306, 283.210371},
                                EcpGaussianTerm{2, 8.52934, 62.281052},
                                EcpGaussianTerm{2, 5.812163, -19.870198},
                                EcpGaussianTerm{2, 5.753715, -26.416452},
                                EcpGaussianTerm{2, 2.513472, -0.99497},
                                EcpGaussianTerm{2, 2.536261, -1.490701},
                            }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 10.854065, 71.982371},
                                       EcpGaussianTerm{2, 9.468229, 143.903532},
                                       EcpGaussianTerm{2, 7.031114, 4.871759},
                                       EcpGaussianTerm{2, 6.143858, 8.983059},
                                       EcpGaussianTerm{2, 5.812163, -19.870198},
                                       EcpGaussianTerm{2, 5.753715, -26.416452},
                                       EcpGaussianTerm{2, 2.513472, -0.99497},
                                       EcpGaussianTerm{2, 2.536261, -1.490701},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 8.313515, 36.363237},
                                       EcpGaussianTerm{2, 7.998965, 54.54897},
                                       EcpGaussianTerm{2, 5.179966, 9.776285},
                                       EcpGaussianTerm{2, 4.942226, 14.264755},
                                       EcpGaussianTerm{2, 5.812163, -19.870198},
                                       EcpGaussianTerm{2, 5.753715, -26.416452},
                                       EcpGaussianTerm{2, 2.513472, -0.99497},
                                       EcpGaussianTerm{2, 2.536261, -1.490701},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 5.812163, 19.870198},
                                       EcpGaussianTerm{2, 5.753715, 26.416452},
                                       EcpGaussianTerm{2, 2.513472, 0.99497},
                                       EcpGaussianTerm{2, 2.536261, 1.490701},
                                   }},
                    }},
            // Z=86 (Rn): n_core=60, 3 projector channels + local
            AtomEcp{60,
                    {
                        EcpChannel{0,
                                   {
                                       EcpGaussianTerm{2, 30.151242, 49.965551},
                                       EcpGaussianTerm{2, 14.521241, 283.07},
                                       EcpGaussianTerm{2, 8.052038, 62.00287},
                                       EcpGaussianTerm{2, 6.348571, -21.79729},
                                       EcpGaussianTerm{2, 6.295949, -28.946805},
                                       EcpGaussianTerm{2, 2.882118, -1.447365},
                                       EcpGaussianTerm{2, 2.908048, -2.177964},
                                   }},
                        EcpChannel{1,
                                   {
                                       EcpGaussianTerm{2, 11.009942, 71.969119},
                                       EcpGaussianTerm{2, 9.617625, 143.860559},
                                       EcpGaussianTerm{2, 7.336008, 4.714761},
                                       EcpGaussianTerm{2, 6.406253, 9.013065},
                                       EcpGaussianTerm{2, 6.348571, -21.79729},
                                       EcpGaussianTerm{2, 6.295949, -28.946805},
                                       EcpGaussianTerm{2, 2.882118, -1.447365},
                                       EcpGaussianTerm{2, 2.908048, -2.177964},
                                   }},
                        EcpChannel{2,
                                   {
                                       EcpGaussianTerm{2, 8.36922, 36.368365},
                                       EcpGaussianTerm{2, 8.116975, 54.551761},
                                       EcpGaussianTerm{2, 5.353656, 9.634487},
                                       EcpGaussianTerm{2, 5.097212, 14.387902},
                                       EcpGaussianTerm{2, 6.348571, -21.79729},
                                       EcpGaussianTerm{2, 6.295949, -28.946805},
                                       EcpGaussianTerm{2, 2.882118, -1.447365},
                                       EcpGaussianTerm{2, 2.908048, -2.177964},
                                   }},
                        EcpChannel{3,
                                   {
                                       // local U_L
                                       EcpGaussianTerm{2, 6.348571, 21.79729},
                                       EcpGaussianTerm{2, 6.295949, 28.946805},
                                       EcpGaussianTerm{2, 2.882118, 1.447365},
                                       EcpGaussianTerm{2, 2.908048, 2.177964},
                                   }},
                    }},
        };
    }();
    return data;
}

} // namespace basis_data
} // namespace scf
} // namespace cpu
} // namespace ATV
