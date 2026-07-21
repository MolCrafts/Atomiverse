/// @file cpu/gpw/basis_data/gpw_tables.h
/// @brief GTH pseudopotential + MOLOPT/PADE Gaussian-basis static data tables.
///
/// GENERATED ONCE by tools/gen_gpw_tables.cpp from the published parameter
/// sets and committed as plain C++ data — nothing is parsed at build or run
/// time. Provenance: GTH-PBE — Krack, TCA 114, 145 (2005); GTH-PADE —
/// Goedecker-Teter-Hutter, PRB 54, 1703 (1996) / Hartwigsen et al., PRB 58,
/// 3641 (1998); MOLOPT bases — VandeVondele & Hutter, JCP 127, 114105 (2007);
/// GTH-PADE bases — the CP2K BASIS_SET distribution of the same papers.
/// Consumed by the GthPotential / GaussianBasis lookup constructors only.
#pragma once

namespace ATV {
namespace cpu {
namespace gpw {
namespace basis_data {

/// @brief One GTH nonlocal channel: l, r_l [Bohr], nproj, h matrix (row-major).
struct GthChannelRow {
    int l;
    double r;
    int nproj;
    const double* h;
    unsigned long h_len;
};

/// @brief One GTH pseudopotential entry (Z + name + local/nonlocal data).
struct GthPotentialRow {
    int z;
    const char* symbol;
    const char* name;
    int zion;
    double rloc;
    const double* cloc;
    unsigned long n_cloc;
    const GthChannelRow* channels;
    unsigned long n_channels;
};

/// @brief One contraction set: shared exponents + row-major coefficients.
struct GaussianBasisSetRow {
    int n;
    int lmin;
    int lmax;
    int ncoef;
    const double* exps;
    unsigned long n_exps;
    const int* nshell;
    unsigned long n_nshell;
    const double* coef;
    unsigned long n_coef;
};

/// @brief One contracted Gaussian basis entry (Z + name + sets).
struct GaussianBasisRow {
    int z;
    const char* symbol;
    const char* name;
    const GaussianBasisSetRow* sets;
    unsigned long n_sets;
};

inline constexpr double kPot_H_GTH_PBE_q1_cloc[] = {-4.1789004399999996,
                                                    0.72446330999999997};
inline constexpr GthPotentialRow kPot_H_GTH_PBE_q1 = {1,
                                                      "H",
                                                      "GTH-PBE-q1",
                                                      1,
                                                      0.20000000000000001,
                                                      kPot_H_GTH_PBE_q1_cloc,
                                                      2,
                                                      nullptr,
                                                      0};

inline constexpr double kPot_O_GTH_PBE_q6_cloc[] = {-16.6672148,
                                                    2.4873113199999999};
inline constexpr double kPot_O_GTH_PBE_q6_h0[] = {18.33745811};
inline constexpr GthChannelRow kPot_O_GTH_PBE_q6_ch[] = {
    {0, 0.22095592, 1, kPot_O_GTH_PBE_q6_h0, 1},
    {1, 0.21133246999999999, 0, nullptr, 0},
};
inline constexpr GthPotentialRow kPot_O_GTH_PBE_q6 = {8,
                                                      "O",
                                                      "GTH-PBE-q6",
                                                      6,
                                                      0.2445543,
                                                      kPot_O_GTH_PBE_q6_cloc,
                                                      2,
                                                      kPot_O_GTH_PBE_q6_ch,
                                                      2};

inline constexpr double kPot_Na_GTH_PBE_q9_cloc[] = {-8.8799239300000004,
                                                     3.94443828};
inline constexpr double kPot_Na_GTH_PBE_q9_h0[] = {38.694398820000004};
inline constexpr double kPot_Na_GTH_PBE_q9_h1[] = {-7.5332670500000001};
inline constexpr GthChannelRow kPot_Na_GTH_PBE_q9_ch[] = {
    {0, 0.14024412, 1, kPot_Na_GTH_PBE_q9_h0, 1},
    {1, 0.14800395999999999, 1, kPot_Na_GTH_PBE_q9_h1, 1},
};
inline constexpr GthPotentialRow kPot_Na_GTH_PBE_q9 = {11,
                                                       "Na",
                                                       "GTH-PBE-q9",
                                                       9,
                                                       0.17999999999999999,
                                                       kPot_Na_GTH_PBE_q9_cloc,
                                                       2,
                                                       kPot_Na_GTH_PBE_q9_ch,
                                                       2};

inline constexpr double kPot_Cl_GTH_PBE_q7_cloc[] = {-6.3920818099999996};
inline constexpr double kPot_Cl_GTH_PBE_q7_h0[] = {
    15.21898983, -4.93452321, -4.93452321, 6.3704420800000001};
inline constexpr double kPot_Cl_GTH_PBE_q7_h1[] = {4.3387752700000002};
inline constexpr GthChannelRow kPot_Cl_GTH_PBE_q7_ch[] = {
    {0, 0.33953864, 2, kPot_Cl_GTH_PBE_q7_h0, 4},
    {1, 0.37847416, 1, kPot_Cl_GTH_PBE_q7_h1, 1},
};
inline constexpr GthPotentialRow kPot_Cl_GTH_PBE_q7 = {17,
                                                       "Cl",
                                                       "GTH-PBE-q7",
                                                       7,
                                                       0.40999999999999998,
                                                       kPot_Cl_GTH_PBE_q7_cloc,
                                                       1,
                                                       kPot_Cl_GTH_PBE_q7_ch,
                                                       2};

inline constexpr double kPot_Zn_GTH_PBE_q12_h0[] = {
    11.53004133,         -8.7918981499999997, 3.14508644,
    -8.7918981499999997, 16.465775180000001,  -8.1205782699999993,
    3.14508644,          -8.1205782699999993, 6.4455091800000002};
inline constexpr double kPot_Zn_GTH_PBE_q12_h1[] = {
    2.5971951199999999, -0.59426274999999995, -0.59426274999999995,
    0.70314116999999998};
inline constexpr double kPot_Zn_GTH_PBE_q12_h2[] = {-14.466957949999999};
inline constexpr GthChannelRow kPot_Zn_GTH_PBE_q12_ch[] = {
    {0, 0.40031644, 3, kPot_Zn_GTH_PBE_q12_h0, 9},
    {1, 0.54318233000000005, 2, kPot_Zn_GTH_PBE_q12_h1, 4},
    {2, 0.25095885000000001, 1, kPot_Zn_GTH_PBE_q12_h2, 1},
};
inline constexpr GthPotentialRow kPot_Zn_GTH_PBE_q12 = {
    30,      "Zn", "GTH-PBE-q12",          12, 0.51000000000000001,
    nullptr, 0,    kPot_Zn_GTH_PBE_q12_ch, 3};

inline constexpr double kPot_H_GTH_PADE_q1_cloc[] = {-4.1802368000000003,
                                                     0.72507482000000001};
inline constexpr GthPotentialRow kPot_H_GTH_PADE_q1 = {1,
                                                       "H",
                                                       "GTH-PADE-q1",
                                                       1,
                                                       0.20000000000000001,
                                                       kPot_H_GTH_PADE_q1_cloc,
                                                       2,
                                                       nullptr,
                                                       0};

inline constexpr double kPot_Ar_GTH_PADE_q8_cloc[] = {-7.0999999999999996};
inline constexpr double kPot_Ar_GTH_PADE_q8_h0[] = {
    10.249486989999999, -2.16984522, -2.16984522, 5.6025162699999997};
inline constexpr double kPot_Ar_GTH_PADE_q8_h1[] = {4.9788010099999997};
inline constexpr GthChannelRow kPot_Ar_GTH_PADE_q8_ch[] = {
    {0, 0.31738081000000001, 2, kPot_Ar_GTH_PADE_q8_h0, 4},
    {1, 0.35161921000000002, 1, kPot_Ar_GTH_PADE_q8_h1, 1},
};
inline constexpr GthPotentialRow kPot_Ar_GTH_PADE_q8 = {
    18,
    "Ar",
    "GTH-PADE-q8",
    8,
    0.40000000000000002,
    kPot_Ar_GTH_PADE_q8_cloc,
    1,
    kPot_Ar_GTH_PADE_q8_ch,
    2};

inline constexpr double kBas_H_DZVP_MOLOPT_SR_GTH_s0_exps[] = {
    10.068468228533, 2.6802228680889999, 0.79150153912200005, 0.239116150487,
    0.082193184441000003};
inline constexpr int kBas_H_DZVP_MOLOPT_SR_GTH_s0_nshell[] = {2, 1};
inline constexpr double kBas_H_DZVP_MOLOPT_SR_GTH_s0_coef[] = {
    -0.033917444900000003, 0.059193775499999997, 0.0099051343999999996,
    -0.1222022121,         0.84331832890000002,  0.1224495665,
    -0.44381886120000003,  -1.1557071155,        0.47718324090000003,
    -0.45318218659999998,  0.049479621199999997, 0.54791967819999998,
    -0.13161286150000001,  0.52270873799999995,  0.86903185400000005};
inline constexpr GaussianBasisSetRow kBas_H_DZVP_MOLOPT_SR_GTH_sets[] = {
    {2, 0, 1, 3, kBas_H_DZVP_MOLOPT_SR_GTH_s0_exps, 5,
     kBas_H_DZVP_MOLOPT_SR_GTH_s0_nshell, 2, kBas_H_DZVP_MOLOPT_SR_GTH_s0_coef,
     15},
};
inline constexpr GaussianBasisRow kBas_H_DZVP_MOLOPT_SR_GTH = {
    1, "H", "DZVP-MOLOPT-SR-GTH", kBas_H_DZVP_MOLOPT_SR_GTH_sets, 1};

inline constexpr double kBas_O_DZVP_MOLOPT_SR_GTH_s0_exps[] = {
    10.389228018317, 3.8496210720050001, 1.3884011887409999,
    0.49695504365499998, 0.16249161504000001};
inline constexpr int kBas_O_DZVP_MOLOPT_SR_GTH_s0_nshell[] = {2, 2, 1};
inline constexpr double kBas_O_DZVP_MOLOPT_SR_GTH_s0_coef[] = {
    0.12624072289999999,   0.069215797900000001,  -0.061302037199999999,
    -0.026862701100000001, 0.029845227500000002,  0.13993370429999999,
    0.1156345389,          -0.19008751169999999,  -0.0062830209999999997,
    0.060939733900000001,  -0.43434823169999998,  -0.3228397194,
    -0.3777269828,         -0.22483918780000001,  0.73232158010000004,
    -0.85279179090000001,  -0.095944016600000001, -0.45426608600000001,
    0.38032465859999998,   0.89356491839999996,   -0.24235153779999999,
    1.1028303487,          -0.25738898300000002,  1.0541029199,
    0.15295418869999999};
inline constexpr GaussianBasisSetRow kBas_O_DZVP_MOLOPT_SR_GTH_sets[] = {
    {2, 0, 2, 5, kBas_O_DZVP_MOLOPT_SR_GTH_s0_exps, 5,
     kBas_O_DZVP_MOLOPT_SR_GTH_s0_nshell, 3, kBas_O_DZVP_MOLOPT_SR_GTH_s0_coef,
     25},
};
inline constexpr GaussianBasisRow kBas_O_DZVP_MOLOPT_SR_GTH = {
    8, "O", "DZVP-MOLOPT-SR-GTH", kBas_O_DZVP_MOLOPT_SR_GTH_sets, 1};

inline constexpr double kBas_Na_DZVP_MOLOPT_SR_GTH_s0_exps[] = {
    23.518800761960001, 11.135656103275,     4.6478138202460002,
    1.866708259982,     0.73468369719600002, 0.27567299585999999,
    0.049895108245};
inline constexpr int kBas_Na_DZVP_MOLOPT_SR_GTH_s0_nshell[] = {3, 2};
inline constexpr double kBas_Na_DZVP_MOLOPT_SR_GTH_s0_coef[] = {
    0.037798268053000003,    0.0050404706180000002,  0.037779070657000001,
    -0.041235150854000001,   -0.0054108386709999998, 0.18041527201599999,
    0.031210048356000001,    -0.119614429319,        -0.102704065658,
    -0.017609338317000001,   -0.084235489885000001,  -0.021214584886999999,
    0.224302495702,          -0.27650715637399997,   -0.043803008652999999,
    -0.50200723946799997,    -0.088006426679999999,  -0.18377981321199999,
    -0.40508798559999998,    -0.067844235623000002,  -0.48745471299400001,
    -0.24156264358000001,    1.6416498675019999,     -0.34566976259799997,
    -0.085057974012000007,   -0.088909855777999994,  0.046069503611999997,
    -2.2815648219679998,     -0.128294724774,        0.15952777071900001,
    -0.00024562485300000001, 1.0086942924,           0.80166099098099997,
    -0.002074313963,         0.98214942456800003};
inline constexpr GaussianBasisSetRow kBas_Na_DZVP_MOLOPT_SR_GTH_sets[] = {
    {2, 0, 1, 5, kBas_Na_DZVP_MOLOPT_SR_GTH_s0_exps, 7,
     kBas_Na_DZVP_MOLOPT_SR_GTH_s0_nshell, 2,
     kBas_Na_DZVP_MOLOPT_SR_GTH_s0_coef, 35},
};
inline constexpr GaussianBasisRow kBas_Na_DZVP_MOLOPT_SR_GTH = {
    11, "Na", "DZVP-MOLOPT-SR-GTH", kBas_Na_DZVP_MOLOPT_SR_GTH_sets, 1};

inline constexpr double kBas_Cl_DZVP_MOLOPT_SR_GTH_s0_exps[] = {
    2.3790781176009999, 1.4707184650130001, 0.46121347405699997,
    0.150014717415};
inline constexpr int kBas_Cl_DZVP_MOLOPT_SR_GTH_s0_nshell[] = {2, 2, 1};
inline constexpr double kBas_Cl_DZVP_MOLOPT_SR_GTH_s0_coef[] = {
    0.34906497910000001,  0.15313820619999999,  0.1164622704,
    0.16006789190000001,  0.074898027899999997, 0.023265903500000001,
    0.17005150990000001,  -0.1939453937,        -0.2127812258,
    0.30385656999999999,  -1.0154926304,        -0.70251626730000005,
    -0.46352968239999998, -0.43902323720000003, 0.76687843680000001,
    -0.40259999149999998, 0.68143733660000005,  -0.2998347147,
    1.0366341794,         0.15494739869999999};
inline constexpr GaussianBasisSetRow kBas_Cl_DZVP_MOLOPT_SR_GTH_sets[] = {
    {2, 0, 2, 5, kBas_Cl_DZVP_MOLOPT_SR_GTH_s0_exps, 4,
     kBas_Cl_DZVP_MOLOPT_SR_GTH_s0_nshell, 3,
     kBas_Cl_DZVP_MOLOPT_SR_GTH_s0_coef, 20},
};
inline constexpr GaussianBasisRow kBas_Cl_DZVP_MOLOPT_SR_GTH = {
    17, "Cl", "DZVP-MOLOPT-SR-GTH", kBas_Cl_DZVP_MOLOPT_SR_GTH_sets, 1};

inline constexpr double kBas_Zn_DZVP_MOLOPT_SR_GTH_s0_exps[] = {
    6.4008127245950002,  3.1677933326900001, 1.341703542578,
    0.54541841694100002, 0.222221794449,     0.079830045424000004};
inline constexpr int kBas_Zn_DZVP_MOLOPT_SR_GTH_s0_nshell[] = {2, 2, 2, 1};
inline constexpr double kBas_Zn_DZVP_MOLOPT_SR_GTH_s0_coef[] = {
    0.021801105639,          0.042720631346000001,  0.002944311709,
    0.015629658082000002,    0.28568721250599999,   -0.088263924682999997,
    -0.00054829900300000002, -0.095613408620000007, -0.109242428655,
    -0.010450068121,         -0.063967389104000003, 0.35891394943400001,
    -0.137258994504,         0.00106369108,         0.274599856591,
    -0.024734012319,         0.109519543853,        -0.067544866326,
    0.32607646659,           -0.22154432771300001,  0.030739369071,
    0.13705627440199999,     -1.0036477073009999,   -0.060402922898999997,
    -0.14458552403800001,    0.224403593782,        0.258114817153,
    0.50450943302700002,     -0.62518419727100005,  2.351480103434,
    -0.53192097234699998,    1.5535206381319999,    0.097325359555000004,
    0.33372421576099998,     0.22247508812399999,   -0.48218481812399999,
    -1.671373693342,         -0.498668813865,       -1.253104288811,
    0.0090489265980000008,   0.85799594281000002,   0.89409063936699995};
inline constexpr GaussianBasisSetRow kBas_Zn_DZVP_MOLOPT_SR_GTH_sets[] = {
    {2, 0, 3, 7, kBas_Zn_DZVP_MOLOPT_SR_GTH_s0_exps, 6,
     kBas_Zn_DZVP_MOLOPT_SR_GTH_s0_nshell, 4,
     kBas_Zn_DZVP_MOLOPT_SR_GTH_s0_coef, 42},
};
inline constexpr GaussianBasisRow kBas_Zn_DZVP_MOLOPT_SR_GTH = {
    30, "Zn", "DZVP-MOLOPT-SR-GTH", kBas_Zn_DZVP_MOLOPT_SR_GTH_sets, 1};

inline constexpr double kBas_H_DZVP_MOLOPT_GTH_s0_exps[] = {
    11.478000339908, 3.7007585627630002,   1.446884268432, 0.71681458969599998,
    0.247918564176,  0.066918004003999998, 0.021708243634};
inline constexpr int kBas_H_DZVP_MOLOPT_GTH_s0_nshell[] = {2, 1};
inline constexpr double kBas_H_DZVP_MOLOPT_GTH_s0_coef[] = {
    0.0249162432,         -0.012512421399999999, 0.024510918199999999,
    0.079825489999999999, -0.056449071099999998, 0.058140794099999997,
    0.12886267530000001,  0.0112426847,          0.44470949850000002,
    0.37944889459999998,  -0.4185875483,         0.6462079731,
    0.3245524326,         0.59036321670000003,   0.80338501819999997,
    0.037148121399999998, 0.43870313300000002,   0.89297120870000002,
    -0.0011251955,        -0.0596931713,         0.1201013165};
inline constexpr GaussianBasisSetRow kBas_H_DZVP_MOLOPT_GTH_sets[] = {
    {2, 0, 1, 3, kBas_H_DZVP_MOLOPT_GTH_s0_exps, 7,
     kBas_H_DZVP_MOLOPT_GTH_s0_nshell, 2, kBas_H_DZVP_MOLOPT_GTH_s0_coef, 21},
};
inline constexpr GaussianBasisRow kBas_H_DZVP_MOLOPT_GTH = {
    1, "H", "DZVP-MOLOPT-GTH", kBas_H_DZVP_MOLOPT_GTH_sets, 1};

inline constexpr double kBas_O_DZVP_MOLOPT_GTH_s0_exps[] = {
    12.015954705512,     5.1081502873850004,  2.0483980398739998,
    0.83238157558199999, 0.35231624645499998, 0.14297733088,
    0.046760918300000003};
inline constexpr int kBas_O_DZVP_MOLOPT_GTH_s0_nshell[] = {2, 2, 1};
inline constexpr double kBas_O_DZVP_MOLOPT_GTH_s0_coef[] = {
    -0.060190841199999998,
    0.065738617900000004,
    0.0365436388,
    -0.034210557400000001,
    0.0148070544,
    -0.1295979233,
    0.1108859022,
    0.1209276487,
    -0.12061977090000001,
    0.068186159299999993,
    0.1181758894,
    -0.053732406400000002,
    0.25109367030000002,
    -0.21371946459999999,
    0.2905764992,
    0.46296448499999998,
    -0.57267066619999996,
    0.35263991030000003,
    -0.47367485840000001,
    1.0633441895,
    0.45035378259999997,
    0.18676000670000001,
    0.29470864520000001,
    0.4848483764,
    0.3076561142,
    0.092715833600000006,
    0.38720145859999999,
    0.17303986930000001,
    0.71746591969999995,
    0.3183468344,
    -0.00025594579999999999,
    0.0038258495999999999,
    0.0097261106000000007,
    0.032498979400000003,
    -0.0057717365999999997};
inline constexpr GaussianBasisSetRow kBas_O_DZVP_MOLOPT_GTH_sets[] = {
    {2, 0, 2, 5, kBas_O_DZVP_MOLOPT_GTH_s0_exps, 7,
     kBas_O_DZVP_MOLOPT_GTH_s0_nshell, 3, kBas_O_DZVP_MOLOPT_GTH_s0_coef, 35},
};
inline constexpr GaussianBasisRow kBas_O_DZVP_MOLOPT_GTH = {
    8, "O", "DZVP-MOLOPT-GTH", kBas_O_DZVP_MOLOPT_GTH_sets, 1};

inline constexpr double kBas_H_DZV_GTH_PADE_s0_exps[] = {
    8.3744350009000001, 1.8058681459999999, 0.48525310319999998, 0.1658235797};
inline constexpr int kBas_H_DZV_GTH_PADE_s0_nshell[] = {2};
inline constexpr double kBas_H_DZV_GTH_PADE_s0_coef[] = {
    -0.023894373199999999, 0, -0.13979432589999999, 0,
    -0.25309708739999998,  0, -0.6955307423,        1};
inline constexpr GaussianBasisSetRow kBas_H_DZV_GTH_PADE_sets[] = {
    {1, 0, 0, 2, kBas_H_DZV_GTH_PADE_s0_exps, 4, kBas_H_DZV_GTH_PADE_s0_nshell,
     1, kBas_H_DZV_GTH_PADE_s0_coef, 8},
};
inline constexpr GaussianBasisRow kBas_H_DZV_GTH_PADE = {
    1, "H", "DZV-GTH-PADE", kBas_H_DZV_GTH_PADE_sets, 1};

inline constexpr double kBas_Ar_DZVP_GTH_PADE_s0_exps[] = {
    2.643629099, 1.6948961534, 0.52311083869999997, 0.1663306775};
inline constexpr int kBas_Ar_DZVP_GTH_PADE_s0_nshell[] = {2, 2};
inline constexpr double kBas_Ar_DZVP_GTH_PADE_s0_coef[] = {
    0.3737616862,         0, 0.1735201807,         0, -0.099656118299999999, 0,
    -0.31009287330000002, 0, -0.83151956589999998, 0, -0.58075637840000005,  0,
    -0.33399304210000003, 1, -0.38793300180000001, 1};
inline constexpr double kBas_Ar_DZVP_GTH_PADE_s1_exps[] = {0.84999999999999998};
inline constexpr int kBas_Ar_DZVP_GTH_PADE_s1_nshell[] = {1};
inline constexpr double kBas_Ar_DZVP_GTH_PADE_s1_coef[] = {1};
inline constexpr GaussianBasisSetRow kBas_Ar_DZVP_GTH_PADE_sets[] = {
    {3, 0, 1, 4, kBas_Ar_DZVP_GTH_PADE_s0_exps, 4,
     kBas_Ar_DZVP_GTH_PADE_s0_nshell, 2, kBas_Ar_DZVP_GTH_PADE_s0_coef, 16},
    {3, 2, 2, 1, kBas_Ar_DZVP_GTH_PADE_s1_exps, 1,
     kBas_Ar_DZVP_GTH_PADE_s1_nshell, 1, kBas_Ar_DZVP_GTH_PADE_s1_coef, 1},
};
inline constexpr GaussianBasisRow kBas_Ar_DZVP_GTH_PADE = {
    18, "Ar", "DZVP-GTH-PADE", kBas_Ar_DZVP_GTH_PADE_sets, 2};

inline constexpr const GthPotentialRow* kGthPotentials[] = {
    &kPot_H_GTH_PBE_q1,   &kPot_O_GTH_PBE_q6,   &kPot_Na_GTH_PBE_q9,
    &kPot_Cl_GTH_PBE_q7,  &kPot_Zn_GTH_PBE_q12, &kPot_H_GTH_PADE_q1,
    &kPot_Ar_GTH_PADE_q8,
};

inline constexpr const GaussianBasisRow* kGaussianBases[] = {
    &kBas_H_DZVP_MOLOPT_SR_GTH,  &kBas_O_DZVP_MOLOPT_SR_GTH,
    &kBas_Na_DZVP_MOLOPT_SR_GTH, &kBas_Cl_DZVP_MOLOPT_SR_GTH,
    &kBas_Zn_DZVP_MOLOPT_SR_GTH, &kBas_H_DZVP_MOLOPT_GTH,
    &kBas_O_DZVP_MOLOPT_GTH,     &kBas_H_DZV_GTH_PADE,
    &kBas_Ar_DZVP_GTH_PADE,
};

} // namespace basis_data
} // namespace gpw
} // namespace cpu
} // namespace ATV
