/// @file cpu/scf/rys_roots_iterative.cpp
/// @brief Iterative Rys-quadrature root finder for nroots 6 and 7.
///
/// This file is a namespaced port of libcint's iterative Rys root
/// finder — libcint dispatches nroots 6/7 to
///   segment_solve(CINTrys_jacobi, CINTrys_schmidt, breakpoint = 11):
/// the Jacobi-polynomial / Wheeler-recursion scheme for x <= 11, and the
/// modified-Gram-Schmidt scheme for x > 11. Both branches are ported.
///
/// Ported routines and their libcint origins:
///   Jacobi branch (x <= 11):
///     - flocke_jacobi_moments / Miller recursion ← src/rys_wheeler.c
///     - wheeler_recursion                        ← src/rys_wheeler.c
///     - rys_wheeler_partial                      ← src/rys_wheeler.c
///     - tridiagonal eigensolver (replaces _CINTdiagonalize, which in
///       libcint wraps LAPACK dstemr) — standard implicit-shift QL with
///       eigenvectors (EISPACK tql2).
///     - JACOBI_ALPHA / JACOBI_BETA / JACOBI_RN_PART2 / JACOBI_SN tables
///                                                ← src/rys_wheeler.c
///   Schmidt branch (x > 11):
///     - gamma_inc_like / TURNOVER_POINT          ← src/fmt.c
///     - R_dsmit (modified Gram-Schmidt)          ← src/rys_roots.c
///     - _rdk_rys_roots                           ← src/rys_roots.c
///     - R_dnode / _qr_step / _hessenberg_qr / _CINT_polynomial_roots
///                                                ← src/find_roots.c
///
/// The closed-form rys_roots() dispatcher already strips off the
/// SMALLX (x <= 3e-7) and LARGEX (x >= 35+5N) regimes analytically, so
/// this function only ever sees the intermediate range — the
/// naive_jacobi_moments small-x branch of libcint is therefore not
/// needed and not ported.
///
/// Upstream attribution:
///   libcint: https://github.com/sunqm/libcint (BSD 2-clause)
///   Copyright (c) 2012 Qiming Sun.
///   find_roots.c is itself modified from mpmath's
///   matrices.eigen.eig (restricted to real eigenvalues).
///
/// BSD 2-Clause License — reproduced from the libcint distribution:
///   Redistribution and use in source and binary forms, with or without
///   modification, are permitted provided that the conditions of the
///   BSD 2-Clause License are met. THE SOFTWARE IS PROVIDED "AS IS".
///
/// @see rys_roots_iterative.h for the public declaration and algorithm
///      overview.
/// @see rys_roots.cpp for the closed-form nroots-1..5 dispatcher that
///      routes nroots 6/7 here.

#include "rys_roots_iterative.h"

#include <cfloat>
#include <cmath>
#include <cstdio>

namespace ATV {
namespace cpu {
namespace scf {
namespace native {

namespace {

// ── Shared constants ─────────────────────────────────────────────────
// SQRTPIE4 = sqrt(pi)/4 ... actually 0.5*sqrt(pi)/sqrt(4) — the libcint
// F_0 erf-form prefactor (see fmt.c). Reproduced byte-for-byte.
constexpr double kSqrtPiE4 =
    .8862269254527580136490837416705725913987747280611935641069038949264;
constexpr double kSmlFloat64 = DBL_EPSILON * .5;
// libcint segment_solve breakpoint for nroots 6/7: Jacobi at/below,
// Schmidt above.
constexpr double kJacobiBreakpoint = 11.0;
// Miller-algorithm extra orders for double-precision Jacobi moments
// (libcint FLOCKE_EXTRA_ORDER_FOR_DP).
constexpr int kFlockeExtraOrder = 20;

// nroots <= 7. The Jacobi scheme works with 2*nroots moments and a
// 2*nroots-long alpha/beta; the Wheeler / Schmidt working arrays scale
// the same way. Generous fixed upper bounds keep everything on the
// stack.
constexpr int kNR = kIterativeRysMaxRoots;     // 7
constexpr int kN2 = 2 * kNR;                   // 14 — moments dim
constexpr int kMaxMoments = 2 * (kNR + 1) + 2; // 18 — Schmidt fmt
constexpr int kMaxN1 = kNR + 1;                // 8

// ════════════════════════════════════════════════════════════════════
//  Jacobi / Wheeler branch (x <= 11)
// ════════════════════════════════════════════════════════════════════

// Recurrence coefficients of the shifted Jacobi polynomials on [0,1]
// (libcint src/rys_wheeler.c). 48 entries cover all orders the
// nroots-6/7 Wheeler recursion ever indexes (max index 2*nroots-2).
const double kJacobiAlpha[] = {
    0.333333333333333333333333333333333333,
    0.523809523809523809523809523809523810,
    0.506493506493506493506493506493506494,
    0.503030303030303030303030303030303030,
    0.501754385964912280701754385964912281,
    0.501144164759725400457665903890160183,
    0.500805152979066022544283413848631240,
    0.500597371565113500597371565113500597,
    0.500460829493087557603686635944700461,
    0.500366300366300366300366300366300366,
    0.500298151460942158616577221228384019,
    0.500247402276100940128649183572488867,
    0.500208594075928243637880684188569045,
    0.500178253119429590017825311942959002,
    0.500154083204930662557781201848998459,
    0.500134517083669626042507398439601829,
    0.500118455342335939350864723999052357,
    0.500105108261509354635274332562539416,
    0.500093896713615023474178403755868545,
    0.500084388185654008438818565400843882,
    0.500076254384627116059173402470642062,
    0.500069242487190139869824124082537045,
    0.500063155235569028672476948339017305,
    0.500057836899942163100057836899942163,
    0.500053163211057947900053163211057948,
    0.500049034029616553888398548592723350,
    0.500045367933944288177116414118501043,
    0.500042098172939294434621537425275743,
    0.500039169604386995691343517430473952,
    0.500036536353671903544026306174643771,
    0.500034160005465600874496139919382387,
    0.500032008194097689008386146853594520,
    0.500030053495221494259782412694596382,
    0.500028272547356516822165677127509189,
    0.500026645350386357580602184918731681,
    0.500025154701413694219449615133068370,
    0.500023785738071452357166642880928595,
    0.500022525566517997927647880344190656,
    0.500021362956633198034607989745780816,
    0.500020288090890647190099411645364171,
    0.500019292356368406837211096963383108,
    0.500018368171632195731236912677712061,
    0.500017508841965192422173197464719683,
    0.500016708437761069340016708437761069,
    0.500015961691939345570630486831604150,
    0.500015263913056751228745001068473914,
    0.500014610911428654919493878028111394,
    0.500013998936080857854803034969342330,
};

const double kJacobiBeta[] = {
    0,
    0.0888888888888888888888888888888888889,
    0.0653061224489795918367346938775510204,
    0.0635727908455181182453909726636999364,
    0.0630668677727501256913021618903971845,
    0.0628506250145487557904048045811122233,
    0.0627383202808533621388063732109100729,
    0.0626725320467338347287261718934771297,
    0.0626306844685688562397450409219758676,
    0.0626024168881311738454595597452740310,
    0.0625824265599124886179295184923702747,
    0.0625677689984038834439182682794918809,
    0.0625567021738527914561025858962869892,
    0.0625481417132150677511108031817980614,
    0.0625413838666524411762292054790023554,
    0.0625359556958903491628039209027981060,
    0.0625315298227085892971919513683760366,
    0.0625278737843751839315875074783323933,
    0.0625248188203958038426770641549301682,
    0.0625222400126509715550811441222400127,
    0.0625200432923554218218534834461769248,
    0.0625181567280224133007342386318247239,
    0.0625165245548401282337764651568207433,
    0.0625151030019811111861504383082116693,
    0.0625138573245990196466785257298989339,
    0.0625127596587629188827452983765522236,
    0.0625117874484888667690678686386608579,
    0.0625109222768887769076079260615228261,
    0.0625101489869806127933216619694767892,
    0.0625094550129262476820902302274736489,
    0.0625088298660401298247880731141415094,
    0.0625082647359430659414435305409292754,
    0.0625077521782950542645271337054302016,
    0.0625072858682741100226250759358014808,
    0.0625068604044429658749318401286927631,
    0.0625064711515671281343432808324685505,
    0.0625061141137874382619366175795073383,
    0.0625057858316275578773529320825966201,
    0.0625054832978509385509978861892195425,
    0.0625052038883250828781839823557093059,
    0.0625049453049100895427815298756695634,
    0.0625047055280393577559120272350957998,
    0.0625044827771571428430901562023013271,
    0.0625042754775595923421139265505510973,
    0.0625040822324815102880576155219607192,
    0.0625039017995013815508333080556297147,
    0.0625037330705176781129155364679885059,
    0.0625035750546917580575963715315712799,
};

// Recurrence parts for the Miller algorithm in flocke_jacobi_moments.
// Need indices up to 2*nroots-1+kFlockeExtraOrder = 13+20 = 33 for
// nroots = 7; 88 entries is ample.
const double kJacobiRnPart2[] = {
    0.142857142857142857142857142857142857,
    0.0649350649350649350649350649350649351,
    0.0424242424242424242424242424242424242,
    0.0315789473684210526315789473684210526,
    0.0251716247139588100686498855835240275,
    0.0209339774557165861513687600644122383,
    0.0179211469534050179211469534050179211,
    0.0156682027649769585253456221198156682,
    0.0139194139194139194139194139194139194,
    0.0125223613595706618962432915921288014,
    0.0113805047006432459178624443344878773,
    0.0104297037964121818940342094284522320,
    0.00962566844919786096256684491978609626,
    0.00893682588597842835130970724191063174,
    0.00834005918751681463545870325531342480,
    0.00781805259417199715707178393745557925,
    0.00735757830565482446920327937775909186,
    0.00694835680751173708920187793427230047,
    0.00658227848101265822784810126582278481,
    0.00625285953942351685221900259264907732,
    0.00595485389835202880487467109818584684,
    0.00568397120121258052292535051155740811,
    0.00543666859456333140543666859456333141,
    0.00520999468367889420520999468367889421,
    0.00500147102088849661665195645778170050,
    0.00480900099809454677433989656111060702,
    0.00463079902332238780836911678033173360,
    0.00446533490011750881316098707403055229,
    0.00431128973328461819510412860796492510,
    0.00416752066680330668852907016465122634,
    0.00403303245630881505665450355290954484,
    0.00390695437879425377171365029752960269,
    0.00378852134577325417020073508623126944,
    0.00367705835331734612310151878497202238,
    0.00357196760074457916184534889570860794,
    0.00347271775843204414632986061557490129,
    0.00337883497769968914718205162859845925,
    0.00328989532151249732963042085024567400,
    0.00320551836072225603570703996753905457,
    0.00312536173168190762819770806806343327,
    0.00304911649094449138532750450020204989,
    0.00297650313408271176944356900234618482,
    0.00290726817042606516290726817042606516,
    0.00284118116520351157222665602553870710,
    0.00277803217632872363159019446225234301,
    0.00271762952572981502586131322871920750,
    0.00265979785536299241257664417504269676,
    0.00260437642636595516176668009128742113,
    0.00255121762659451101662156938538848087,
    0.00250018565735079337574572369235339258,
    0.00245115537469360557816329930272958759,
    0.00240401126450992513222061954804588227,
    0.00235864653367133252507439656122561446,
    0.00231496230221939046405436975682276734,
    0.00227286688371521592235395294551262363,
    0.00223227514272732660358349301673218625,
    0.00219310791998016667620191849266739135,
    0.00215529151699364465321912130422768721,
    0.00211875723315231905991275705510549274,
    0.00208344094908483564922430566317130706,
    0.00204928275103713700204928275103713700,
    0.00201622659160927141635885607368098456,
    0.00198421998281384266854152019373486446,
    0.00195321371791960027254144901203724733,
    0.00192316161897910947340605135281941366,
    0.00189402030731547542757864456501616326,
    0.00186574899456859737136697209668725901,
    0.00183830929218383092921838309292183831,
    0.00181166503747148908439231019876181167,
    0.00178578213457958129108248793647175045,
    0.00176062840890902599081518326541165462,
    0.00173617347366404444604092579953781865,
    0.00171238860737375502358902673423029879,
    0.00168924664134686242276514936794966272,
    0.00166672185613210150446482775367837787,
    0.00164478988615473925242687135163027703,
    0.00162342763178566565769766540632822565,
    0.00160261317817587914050936558975144184,
    0.00158232572025675473951336020301537543,
    0.00156254549336646059182623717693644031,
    0.00154325370901619943003758722223800191,
    0.00152443249535741012777516006541201253,
    0.00150606484195337511836587455471885287,
    0.00148813454849645577422621406243120680,
    0.00147062617714595319796692381124384014,
    0.00145352500819183169356668150998563279,
    0.00143681699877665295532730691232132154,
    0.00142048874443240640423739015288311063,
};

const double kJacobiSn[] = {
    0.133333333333333333333333333333333333,
    0.0816326530612244897959183673469387755,
    0.0741682559864378046196228014409832592,
    0.0709502262443438914027149321266968326,
    0.0691356875160036313694452850392234456,
    0.0679665136375911423170402376451525790,
    0.0671491414786433943522066127430112104,
    0.0665451022478544097547291059795993593,
    0.0660803289374717946146517575089003660,
    0.0657115478879081130488259944169887885,
    0.0654117584983313326913690986558324210,
    0.0651632314310966577667735269752989471,
    0.0649538394714156472799996802272518330,
    0.0647750047190328855039516771032524395,
    0.0646204875524200274682307182662247096,
    0.0644856401296682327127291998486377877,
    0.0643669288956803364001636106394598167,
    0.0642616193431845761716403159370115618,
    0.0641675621182470497538990689675621182,
    0.0640830443746643073673998205323313479,
    0.0640066842691658040936088633611538840,
    0.0639373546583592220572713848194757602,
    0.0638741269802850483858493608801293143,
    0.0638162293521948325559843283492718283,
    0.0637630148519381772604002043440832681,
    0.0637139372071136526684730199586351051,
    0.0636685319486830135170080728404399155,
    0.0636264016474624094503452630760745891,
    0.0635872042372870450559193721279473325,
    0.0635506436971407986552012076660438679,
    0.0635164625542647282953377810335249089,
    0.0634844358060809144874103701695775485,
    0.0634543659571873541138769710257378668,
    0.0634260789398024212554455436599970685,
    0.0633994207394466585362624705586466726,
    0.0633742545875900415711301817125560514,
    0.0633504586131360383892090527864154933,
    0.0633279238675595035319320689022355891,
    0.0633065526561241224022632641807825021,
    0.0632862571212214656620662989991154329,
    0.0632669580344788621187890031769872120,
    0.0632485837625994897816983723475668191,
    0.0632310693784614480670222280220691334,
    0.0632143558942142547231491793347102729,
    0.0631983895972736191236203448118033782,
    0.0631831214734580876576211401252492506,
    0.0631685067042097554837410137819071446,
    0.0631545042270307306418800612740449121,
    0.0631410763500519566499298347892141201,
    0.0631281884131127460009319915165060210,
    0.0631158084889316866582675155969963302,
    0.0631039071189426329431813657385193769,
    0.0630924570791929844168441760812401272,
    0.0630814331723869622592583712817914834,
    0.0630708120427293346371691454965353148,
    0.0630605720107052623408921603962253696,
    0.0630506929253359486693139231697881124,
    0.0630411560317907882669526183723252281,
    0.0630319438525254492239910205987482423,
    0.0630230400803605451462935505127575629,
    0.0630144294821244268441141487100133182,
    0.0630060978116620368437903104284681857,
    0.0629980317311645809853106304095146108,
    0.0629902187399059982519710116900037282,
    0.0629826471095851879522211025629876426,
    0.0629753058255704593910133559737027347,
    0.0629681845334270223004317456018676823,
    0.0629612734901814788868752070834879740,
    0.0629545635198408417968641138972612553,
    0.0629480459727389605947825152324343429,
    0.0629417126883315527433774503501116480,
    0.0629355559611032858765030867631851152,
    0.0629295685092873800533889095865830818,
    0.0629237434461307009797940084537341416,
    0.0629180742534659027450976347026882854,
    0.0629125547573773684532637358197768559,
    0.0629071791057699320890454947020834937,
    0.0629019417476690252459380084028850472,
    0.0628968374140983050099351803916073865,
    0.0628918611003962662746221735204718309,
    0.0628870080498470665701343133259403169,
    0.0628822737385130048545800855496113614,
    0.0628776538611669803523093451739050315,
    0.0628731443182329721334066277492113016,
    0.0628687412036512628117090270546644992,
    0.0628644407935929008720508221776268525,
    0.0628602395359548608299450549157781435,
    0.0628561340405736106069856843147540300,
};

// Flocke's Miller-algorithm recipe (JCP 131, 064107) for the shifted
// Jacobi moments mus[0..n-1] at parameter t (libcint
// flocke_jacobi_moments, src/rys_wheeler.c). t must be > SMALLX_LIMIT —
// guaranteed because rys_roots() handles the small-x regime analytically.
void flocke_jacobi_moments(int n, double t, double* mus) {
    const double t_inv = .5 / t;
    double mu1 = 1.; // arbitrary non-zero seed (Miller normalization)
    double mu2 = 0.;
    double mu0 = 0.;
    int i;
    for (i = n - 1 + kFlockeExtraOrder; i >= n; i--) {
        double rn = (2 * i + 3) * t_inv + kJacobiRnPart2[i];
        mu0 = (mu2 - rn * mu1) / kJacobiSn[i];
        mu2 = mu1;
        mu1 = mu0;
    }
    for (; i >= 0; i--) {
        double rn = (2 * i + 3) * t_inv + kJacobiRnPart2[i];
        mu0 = (mu2 - rn * mu1) / kJacobiSn[i];
        mus[i] = mu0;
        mu2 = mu1;
        mu1 = mu0;
    }

    const double tt = std::sqrt(t);
    const double norm = kSqrtPiE4 * std::erf(tt) / tt / mu0; // fmt[0]/mu0
    for (i = 0; i < n; i++)
        mus[i] *= norm;
}

// Wheeler recursion (libcint src/rys_wheeler.c) — convert the modified
// moments into the three-term recurrence coefficients (a, b) of the
// orthogonal polynomials associated with the discrete Rys weight.
//
// Uses three rotating buffers (s0 / sm / sk). libcint reuses the
// caller's `moments` array as the initial s0 and overwrites it; here we
// keep `moments` const and stage a writable copy into one of the three
// scratch buffers, which is then rotated identically.
void wheeler_recursion(int n, const double* alpha, const double* beta,
                       const double* moments, double* a, double* b) {
    double buf[kN2 * 2 * 3];
    double a0 = alpha[0] + moments[1] / moments[0];
    double b0 = 0;
    a[0] = a0;
    b[0] = b0;

    double* sm = buf;             // moment row m-1
    double* s0 = buf + n * 2;     // moment row m   (writable copy of `moments`)
    double* sk = buf + n * 2 * 2; // moment row m+1
    for (int i = 0; i < n * 2; i++)
        sm[i] = 0.;
    for (int i = 0; i < n * 2; i++)
        s0[i] = moments[i];

    for (int i = 1; i < n; i++) {
        int nc = 2 * (n - i);
        for (int j = 0; j < nc; j++) {
            sk[j] = (s0[2 + j] - (a0 - alpha[i + j]) * s0[1 + j]
                     - b0 * sm[2 + j] + beta[i + j] * s0[j]);
        }
        double a1 = alpha[i] - s0[1] / s0[0] + sk[1] / sk[0];
        double b1 = sk[0] / s0[0];
        a[i] = a1;
        b[i] = b1;
        a0 = a1;
        b0 = b1;
        double* swap = sm;
        sm = s0;
        s0 = sk;
        sk = swap;
    }
}

// Symmetric tridiagonal eigensolver (implicit-shift QL with
// accumulation of eigenvectors — the classic EISPACK tql2). Replaces
// libcint's _CINTdiagonalize (a LAPACK dstemr wrapper). For an
// n x n symmetric tridiagonal matrix with diagonal d[0..n-1] and
// sub/super-diagonal e[0..n-2], on return d holds the eigenvalues and
// z (row-major n x n, identity on entry) holds the eigenvectors in
// columns — z[i*n + k] is component i of eigenvector k.
int tql2(int n, double* d, double* e, double* z) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            z[i * n + k] = (i == k) ? 1.0 : 0.0;

    if (n == 1)
        return 0;

    double ee[kNR];
    for (int i = 1; i < n; i++)
        ee[i - 1] = e[i - 1];
    ee[n - 1] = 0.0;

    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(ee[m]) <= DBL_EPSILON * dd)
                    break;
            }
            if (m != l) {
                if (++iter > 50) {
                    std::fprintf(
                        stderr,
                        "ATV::rys_roots_iterative: tql2 no convergence\n");
                    return 1;
                }
                double g = (d[l + 1] - d[l]) / (2.0 * ee[l]);
                double r = std::hypot(g, 1.0);
                double sign_r = (g >= 0.0) ? std::fabs(r) : -std::fabs(r);
                g = d[m] - d[l] + ee[l] / (g + sign_r);
                double s = 1.0, c = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; i--) {
                    double f = s * ee[i];
                    double bb = c * ee[i];
                    r = std::hypot(f, g);
                    ee[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        ee[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * bb;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - bb;
                    for (int k = 0; k < n; k++) {
                        f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }
                if (r == 0.0 && m - 1 >= l)
                    continue;
                d[l] -= p;
                ee[l] = g;
                ee[m] = 0.0;
            }
        } while (m != l);
    }

    // Sort eigenvalues ascending (selection sort), carrying the
    // eigenvector columns. libcint's _CINTdiagonalize (LAPACK dstemr,
    // RANGE='A') returns eigenvalues in ascending order; downstream code
    // relies on that ordering.
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        double p = d[i];
        for (int j = i + 1; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int row = 0; row < n; row++) {
                double t = z[row * n + i];
                z[row * n + i] = z[row * n + k];
                z[row * n + k] = t;
            }
        }
    }
    return 0;
}

// Final step of the Jacobi/Wheeler scheme: build the symmetric
// tridiagonal Jacobi matrix from (a, b), diagonalize it, and read off
// roots (eigenvalues, mapped through r/(1-r)) and weights (first
// eigenvector component squared, times the zeroth moment).
int rys_wheeler_partial(int n, const double* alpha, const double* beta,
                        const double* moments, double* roots, double* weights) {
    double a[kNR];
    double b[kNR];
    double z[kNR * kNR];
    double mu0 = moments[0];
    int first_seen = 1;

    wheeler_recursion(n, alpha, beta, moments, a, b);

    for (int i = 1; i < n; i++) {
        if (b[i] < 1e-14) {
            if (!first_seen || b[i] < 0.) {
                std::fprintf(
                    stderr,
                    "ATV::rys_roots_iterative: wheeler singular b n=%d i=%d\n",
                    n, i);
                for (int k = 0; k < n; k++) {
                    roots[k] = 0;
                    weights[k] = 0;
                }
                return i;
            }
            first_seen = 0;
        }
        b[i] = std::sqrt(b[i]);
    }

    // Symmetric tridiagonal: diagonal a[0..n-1], off-diagonal b[1..n-1].
    double diag[kNR];
    double offd[kNR];
    for (int i = 0; i < n; i++)
        diag[i] = a[i];
    for (int i = 0; i < n - 1; i++)
        offd[i] = b[i + 1];

    int error = tql2(n, diag, offd, z);

    for (int i = 0; i < n; i++) {
        double r = diag[i];
        roots[i] = r / (1 - r);
        double v0 = z[0 * n + i]; // first component of eigenvector i
        weights[i] = v0 * v0 * mu0;
    }
    return error;
}

int rys_jacobi(int nroots, double x, double* roots, double* weights) {
    double moments[kN2 * 2];
    // flocke_jacobi_moments fills 2*nroots moments.
    flocke_jacobi_moments(nroots * 2, x, moments);
    return rys_wheeler_partial(nroots, kJacobiAlpha, kJacobiBeta, moments,
                               roots, weights);
}

// ════════════════════════════════════════════════════════════════════
//  Schmidt branch (x > 11)
// ════════════════════════════════════════════════════════════════════

// TURNOVER_POINT[m] (libcint src/fmt.c) — T threshold below which
// gamma_inc_like uses the power series and above which the erf-form.
const double kTurnoverPoint[] = {
    0.,
    0.,
    0.866025403784,
    1.295010032056,
    1.705493613097,
    2.106432965305,
    2.501471934009,
    2.892473348218,
    3.280525047072,
    3.666320693281,
    4.05033123037,
    4.432891808508,
    4.814249856864,
    5.194593501454,
    5.574069276051,
    5.952793645111,
    6.330860773135,
    6.708347923415,
    7.08531930745,
    7.461828891625,
    7.837922483937,
    8.213639312398,
    8.589013237349,
    8.964073695432,
    9.338846443746,
    9.713354153046,
    10.08761688545,
    10.46165248270,
    10.83547688448,
    11.20910439128,
    11.58254788331,
    11.95581900374,
    12.32892831326,
    12.70188542111,
    13.07469909673,
    13.44737736550,
    13.81992759110,
    14.19235654675,
    14.56467047710,
    14.93687515212,
};

void fmt1_gamma_inc_like(double* f, double t, int m) {
    double b = m + 0.5;
    double e = .5 * std::exp(-t);
    double x = e;
    double s = e;
    double tol = kSmlFloat64 * e;
    for (double bi = b + 1.; x > tol; bi += 1.) {
        x *= t / bi;
        s += x;
    }
    f[m] = s / b;
    for (int i = m; i > 0; i--) {
        b -= 1.;
        f[i - 1] = (e + t * f[i]) / b;
    }
}

void gamma_inc_like(double* f, double t, int m) {
    if (t < kTurnoverPoint[m]) {
        fmt1_gamma_inc_like(f, t, m);
    } else {
        double tt = std::sqrt(t);
        f[0] = kSqrtPiE4 / tt * std::erf(tt);
        if (m > 0) {
            double e = std::exp(-t);
            double b = .5 / t;
            for (int i = 1; i <= m; i++)
                f[i] = b * ((2 * i - 1) * f[i - 1] - e);
        }
    }
}

// Horner evaluation of a degree-`order` polynomial, a[0] = constant term.
inline double poly_value1(const double* a, int order, double x) {
    double p = a[order];
    for (int i = 1; i <= order; i++)
        p = p * x + a[order - i];
    return p;
}

// Newton/bisection root polish fallback (libcint src/find_roots.c).
int R_dnode(const double* a, double* roots, int order) {
    const double accrt = 1e-15;
    double x1init = 0.0;
    double p1init = a[0];
    for (int m = 0; m < order; ++m) {
        double x0 = x1init;
        double p0 = p1init;
        x1init = roots[m];
        p1init = poly_value1(a, order, x1init);

        if (p1init == 0)
            continue;
        if (p0 * p1init > 0) {
            std::fprintf(
                stderr,
                "ATV::rys_roots_iterative: root %d not bracketed (order %d)\n",
                m, order);
            return 1;
        }
        double x1, p1;
        if (x0 <= x1init) {
            x1 = x1init;
            p1 = p1init;
        } else {
            x1 = x0;
            p1 = p0;
            x0 = x1init;
            p0 = p1init;
        }
        double xi;
        if (p1 == 0) {
            roots[m] = x1;
            continue;
        } else if (p0 == 0) {
            roots[m] = x0;
            continue;
        } else {
            xi = x0 + (x0 - x1) / (p1 - p0) * p0;
        }

        int n = 0;
        while (std::fabs(x1 - x0) > x1 * accrt) {
            if (++n > 200) {
                std::fprintf(
                    stderr,
                    "ATV::rys_roots_iterative: R_dnode no convergence\n");
                return 1;
            }
            double pi = poly_value1(a, order, xi);
            if (pi == 0)
                break;
            else if (p0 * pi <= 0) {
                x1 = xi;
                p1 = pi;
                xi = x0 * .25 + xi * .75;
            } else {
                x0 = xi;
                p0 = pi;
                xi = xi * .75 + x1 * .25;
            }
            pi = poly_value1(a, order, xi);
            if (pi == 0)
                break;
            else if (p0 * pi <= 0) {
                x1 = xi;
                p1 = pi;
            } else {
                x0 = xi;
                p0 = pi;
            }
            xi = x0 + (x0 - x1) / (p1 - p0) * p0;
        }
        roots[m] = xi;
    }
    return 0;
}

void qr_step(double* A, int nroots, int n0, int n1, double shift) {
    int m1 = n0 + 1;
    double c = A[n0 * nroots + n0] - shift;
    double s = A[m1 * nroots + n0];
    double v = std::sqrt(c * c + s * s);

    if (v == 0) {
        v = 1;
        c = 1;
        s = 0;
    }
    v = 1. / v;
    c *= v;
    s *= v;

    for (int k = n0; k < nroots; k++) {
        double x = A[n0 * nroots + k];
        double y = A[m1 * nroots + k];
        A[n0 * nroots + k] = c * x + s * y;
        A[m1 * nroots + k] = c * y - s * x;
    }

    int m3 = (n1 < n0 + 3) ? n1 : (n0 + 3);
    for (int k = 0; k < m3; k++) {
        double x = A[k * nroots + n0];
        double y = A[k * nroots + m1];
        A[k * nroots + n0] = c * x + s * y;
        A[k * nroots + m1] = c * y - s * x;
    }

    for (int j = n0; j < n1 - 2; j++) {
        int j1 = j + 1;
        int j2 = j + 2;
        c = A[j1 * nroots + j];
        s = A[j2 * nroots + j];
        v = std::sqrt(c * c + s * s);
        A[j1 * nroots + j] = v;
        A[j2 * nroots + j] = 0;

        if (v == 0) {
            v = 1;
            c = 1;
            s = 0;
        }
        v = 1. / v;
        c *= v;
        s *= v;

        for (int k = j1; k < nroots; k++) {
            double x = A[j1 * nroots + k];
            double y = A[j2 * nroots + k];
            A[j1 * nroots + k] = c * x + s * y;
            A[j2 * nroots + k] = c * y - s * x;
        }
        m3 = (n1 < j + 4) ? n1 : (j + 4);
        for (int k = 0; k < m3; k++) {
            double x = A[k * nroots + j1];
            double y = A[k * nroots + j2];
            A[k * nroots + j1] = c * x + s * y;
            A[k * nroots + j2] = c * y - s * x;
        }
    }
}

int hessenberg_qr(double* A, int nroots) {
    const double eps = 1e-15;
    const int maxits = 30;
    int n0 = 0;
    int n1 = nroots;
    int its = 0;
    for (int ic = 0; ic < nroots * maxits; ic++) {
        int k = n0;
        while (k + 1 < n1) {
            double s = std::fabs(A[k * nroots + k])
                       + std::fabs(A[(k + 1) * nroots + k + 1]);
            if (std::fabs(A[(k + 1) * nroots + k]) < eps * s)
                break;
            k += 1;
        }

        int k1 = k + 1;
        if (k1 < n1) {
            A[k1 * nroots + k] = 0;
            n0 = k1;
            its = 0;
            if (n0 + 1 >= n1) {
                n0 = 0;
                n1 = k1;
                if (n1 < 2)
                    return 0;
            }
        } else {
            int mm1 = n1 - 1;
            int mm2 = n1 - 2;
            double a11 = A[mm1 * nroots + mm1];
            double a22 = A[mm2 * nroots + mm2];
            double shift;
            double t = a11 + a22;
            double s = (a11 - a22) * (a11 - a22);
            s += 4 * A[mm1 * nroots + mm2] * A[mm2 * nroots + mm1];
            if (s > 0) {
                s = std::sqrt(s);
                double a = (t + s) * .5;
                double bb = (t - s) * .5;
                shift = (std::fabs(a11 - a) > std::fabs(a11 - bb)) ? bb : a;
            } else {
                if (n1 == 2) {
                    std::fprintf(stderr, "ATV::rys_roots_iterative: "
                                         "hessenberg_qr no real roots\n");
                    return 1;
                }
                shift = t * .5;
            }
            its += 1;
            qr_step(A, nroots, n0, n1, shift);
            if (its > maxits) {
                std::fprintf(
                    stderr,
                    "ATV::rys_roots_iterative: hessenberg_qr no convergence\n");
                return 1;
            }
        }
    }
    std::fprintf(stderr, "ATV::rys_roots_iterative: hessenberg_qr failed\n");
    return 1;
}

int cint_polynomial_roots(double* roots, const double* cs, int nroots) {
    int nroots1 = nroots + 1;
    if (nroots == 1) {
        roots[0] = -cs[2] / cs[3];
        return 0;
    } else if (nroots == 2) {
        double dum = std::sqrt(cs[2 * 3 + 1] * cs[2 * 3 + 1]
                               - 4 * cs[2 * 3 + 0] * cs[2 * 3 + 2]);
        roots[0] = (-cs[2 * 3 + 1] - dum) / cs[2 * 3 + 2] / 2;
        roots[1] = (-cs[2 * 3 + 1] + dum) / cs[2 * 3 + 2] / 2;
        return 0;
    }

    double A[kMaxN1 * kMaxN1];
    double fac = -1. / cs[nroots * nroots1 + nroots];
    for (int i = 0; i < nroots; i++)
        A[nroots - 1 - i] = cs[nroots * nroots1 + i] * fac;
    for (int i = nroots; i < nroots * nroots; i++)
        A[i] = 0;
    for (int i = 0; i < nroots - 1; i++)
        A[(i + 1) * nroots + i] = 1.;

    int err = hessenberg_qr(A, nroots);
    if (err == 0) {
        for (int i = 0; i < nroots; i++)
            roots[nroots - 1 - i] = A[i * nroots + i];
    } else {
        double dum = std::sqrt(cs[2 * nroots1 + 1] * cs[2 * nroots1 + 1]
                               - 4 * cs[2 * nroots1 + 0] * cs[2 * nroots1 + 2]);
        roots[0] = .5 * (-cs[2 * nroots1 + 1] - dum) / cs[2 * nroots1 + 2];
        roots[1] = .5 * (-cs[2 * nroots1 + 1] + dum) / cs[2 * nroots1 + 2];
        for (int i = 2; i < nroots; i++)
            roots[i] = 1;
        for (int k = 2; k < nroots; ++k) {
            int order = k + 1;
            const double* a = cs + order * nroots1;
            err = R_dnode(a, roots, order);
            if (err)
                break;
        }
    }
    return err;
}

// Modified Gram-Schmidt orthogonalization (libcint src/rys_roots.c).
int R_dsmit(double* cs, const double* fmt_ints, int n) {
    double v[kMaxN1];

    double fac = -fmt_ints[1] / fmt_ints[0];
    double tmp = fmt_ints[2] + fac * fmt_ints[1];
    if (tmp <= 0) {
        std::fprintf(stderr,
                     "ATV::rys_roots_iterative: R_dsmit negative sqrt (j=1)\n");
        for (int k = 0; k < n; k++)
            cs[1 + k * n] = 0;
        return 1;
    }
    tmp = 1 / std::sqrt(tmp);
    cs[0 + 0 * n] = 1 / std::sqrt(fmt_ints[0]);
    cs[0 + 1 * n] = fac * tmp;
    cs[1 + 1 * n] = tmp;

    for (int j = 2; j < n; ++j) {
        for (int k = 0; k < j; ++k)
            v[k] = 0;
        fac = fmt_ints[j + j];
        for (int k = 0; k < j; ++k) {
            double dot = 0;
            for (int i = 0; i <= k; ++i)
                dot += cs[i + k * n] * fmt_ints[i + j];
            for (int i = 0; i <= k; ++i)
                v[i] -= dot * cs[i + k * n];
            fac -= dot * dot;
        }

        if (fac <= 0) {
            for (int k = j; k < n; k++)
                cs[j + k * n] = 0;
            if (fac == 0)
                return 0;
            std::fprintf(
                stderr,
                "ATV::rys_roots_iterative: R_dsmit negative sqrt (j=%d)\n", j);
            return j;
        }
        fac = 1 / std::sqrt(fac);
        cs[j + j * n] = fac;
        for (int k = 0; k < j; ++k)
            cs[k + j * n] = fac * v[k];
    }
    return 0;
}

int rdk_rys_roots(int nroots, const double* fmt_ints, double* roots,
                  double* weights) {
    int nroots1 = nroots + 1;
    double rt[kMaxN1 + kMaxN1 * kMaxN1];
    double* cs = rt + nroots1;

    if (fmt_ints[0] == 0) {
        for (int k = 0; k < nroots; ++k) {
            roots[k] = 0;
            weights[k] = 0;
        }
        return 0;
    }
    if (nroots == 1) {
        roots[0] = fmt_ints[1] / (fmt_ints[0] - fmt_ints[1]);
        weights[0] = fmt_ints[0];
        return 0;
    }

    int error = R_dsmit(cs, fmt_ints, nroots1);
    if (error)
        return 1;
    error = cint_polynomial_roots(rt, cs, nroots);
    if (error)
        return error;

    for (int k = 0; k < nroots; ++k) {
        double root = rt[k];
        if (root == 1) {
            roots[k] = 0;
            weights[k] = 0;
            continue;
        }

        double dum = 1 / fmt_ints[0];
        for (int j = 1; j < nroots; ++j) {
            const double* a = cs + j * nroots1;
            double poly = poly_value1(a, j, root);
            dum += poly * poly;
        }
        roots[k] = root / (1 - root);
        weights[k] = 1 / dum;
    }
    return 0;
}

int rys_schmidt(int nroots, double x, double* roots, double* weights) {
    double fmt_ints[kMaxMoments];
    gamma_inc_like(fmt_ints, x, nroots * 2);
    return rdk_rys_roots(nroots, fmt_ints, roots, weights);
}

} // anonymous namespace

int rys_roots_iterative(int nroots, double x, double* u, double* w) {
    // libcint segment_solve for nroots 6/7: Jacobi at/below x = 11,
    // Schmidt above; on failure, fall through to the other scheme.
    int err;
    if (x <= kJacobiBreakpoint) {
        err = rys_jacobi(nroots, x, u, w);
        if (err)
            err = rys_schmidt(nroots, x, u, w);
    } else {
        err = rys_schmidt(nroots, x, u, w);
        if (err)
            err = rys_jacobi(nroots, x, u, w);
    }
    return err;
}

} // namespace native
} // namespace scf
} // namespace cpu
} // namespace ATV
