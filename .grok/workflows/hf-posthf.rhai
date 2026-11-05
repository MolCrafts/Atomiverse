// hf-posthf — Gaussian HF and post-HF, one slug per wave, MD-shaped front.
// Tests are the changed gtest cases plus that slug's regression case.
// A reported green whose command is a full suite is rejected here.

let meta = #{
    name: "hf-posthf",
    description: "HF and post-HF in the MD shape; test only the changed cases and their regression",
    when_to_use: "Continue Gaussian HF and post-HF. Architecture follows cpu::md::MD / cuda::md::MD. Never run the full ctest suite.",
    phases: [
        #{ title: "Census", detail: "one HF or post-HF slug" },
        #{ title: "Implement", detail: "MD-shaped front, PySCF source" },
        #{ title: "Test", detail: "changed gtests only" },
        #{ title: "Validate", detail: "one regression case" },
        #{ title: "Simplify", detail: "re-run the same commands" },
        #{ title: "Manual", detail: "roadmap row" },
        #{ title: "Land", detail: "merge to dev, drop the tree" },
    ],
};

let mode = "exhaust";
let max_waves = 4;
let target_branch = "dev";
let slug_arg = "";
if args != () {
    if args.mode != () { mode = args.mode; }
    if args.max_waves != () { max_waves = args.max_waves; }
    if args.target_branch != () { target_branch = args.target_branch; }
    if args.slug != () { slug_arg = args.slug; }
}
if mode != "exhaust" && mode != "burst" { mode = "exhaust"; }
if max_waves < 1 { max_waves = 1; }
if max_waves > 8 { max_waves = 8; }

let census_schema = #{
    "type": "object",
    "required": ["slug", "kind", "summary"],
    "properties": #{
        "slug": #{ "type": "string" },
        "kind": #{ "type": "string" },
        "domain": #{ "type": "string" },
        "pyscf_files": #{ "type": "string" },
        "classic_systems": #{ "type": "string" },
        "regression_case": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "gaps_remaining": #{ "type": "integer" },
    },
};
let stage_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "worktree_path": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "head_sha": #{ "type": "string" },
        "build_command": #{ "type": "string" },
        "test_commands": #{ "type": "string" },
    },
};

fn bad_cmd(cmd, needle) {
    cmd.contains(needle)
}

fn tests_scoped(cmd) {
    if cmd == () { return false; }
    if cmd == "" { return false; }
    if bad_cmd(cmd, "-L regression") { return false; }
    if bad_cmd(cmd, "-LE regression") { return false; }
    if bad_cmd(cmd, "-L cuda") { return false; }
    if bad_cmd(cmd, "--gtest_filter=*") { return false; }
    if bad_cmd(cmd, "--gtest_filter='*'") { return false; }
    if bad_cmd(cmd, "-R ''") { return false; }
    if bad_cmd(cmd, "-R '.*'") { return false; }
    if bad_cmd(cmd, "-R .*") { return false; }
    if bad_cmd(cmd, "test_cpu_(scf") { return false; }
    let has_filter = cmd.contains("-R ") || cmd.contains("-R'") || cmd.contains("--gtest_filter=") || cmd.contains("--case ");
    if !has_filter { return false; }
    if cmd.contains("cmake --build") && !cmd.contains("--target") { return false; }
    if cmd.contains("gpu_job.sh") {
        if !cmd.contains("ATV_CTEST_RE=") { return false; }
        if !cmd.contains("ATV_TARGETS=") { return false; }
    }
    true
}

fn build_scoped(cmd) {
    if cmd == () { return false; }
    if cmd == "" { return false; }
    if cmd.contains("ctest") { return false; }
    if cmd.contains("atv_regressions --") { return false; }
    if cmd.contains("cmake --build") && !cmd.contains("--target") { return false; }
    if cmd.contains("gpu_job.sh") { return false; }
    true
}

fn stage_ok(r) {
    if r == () || r.success != true || r.output == () || r.output.status == () {
        return false;
    }
    let s = r.output.status;
    s == "green" || s == "pass" || s == "ok" || s == "merged" || s == "success"
}

fn accept_build(r) {
    if !stage_ok(r) { return false; }
    let cmd = "";
    if r.output.build_command != () { cmd = r.output.build_command; }
    if !build_scoped(cmd) {
        log("rejected unscoped build: " + cmd);
        return false;
    }
    true
}

fn accept_tested(r) {
    if !stage_ok(r) { return false; }
    let cmd = "";
    if r.output.test_commands != () { cmd = r.output.test_commands; }
    if !tests_scoped(cmd) {
        log("rejected unscoped tests: " + cmd);
        return false;
    }
    true
}

if !tests_scoped("cmake --build build --target atv_tests atv_regressions && ctest --test-dir build --output-on-failure -R '^(Mp2Energy|scf/mp2)$'") {
    complete(#{ status: "blocked", summary: "scope self-test failed" });
}
if tests_scoped("ctest --test-dir build --output-on-failure") {
    complete(#{ status: "blocked", summary: "scope self-test accepted a full suite" });
}
if tests_scoped("ctest --test-dir build --output-on-failure -L cuda -LE regression") {
    complete(#{ status: "blocked", summary: "scope self-test accepted the cuda label" });
}

let law = "";
law += "REPO: Atomiverse. Active branch is dev. Never commit to master. Never stage frequencies.tsv or any file you did not change for this slug. ";
law += "Read CLAUDE.md, .claude/specs/es-unification.md, include/atomiverse/cpu/dynamics/md.h, and include/atomiverse/cuda/dynamics/md.cuh before editing. ";
law += "ARCHITECTURE, copy the MD preset and do not restyle MD. cpu::md::MD is Driver<md::Phase>. cuda::md::MD is cuda::Driver<md::Phase>. Same phase enum, same Frame entry, infrastructure mounted with on(phase).add in the constructor. There is no DynamicsDriver category. An evaluator is a separate type the preset mounts; it writes the Frame and does not own a second time loop. CUDA matches that front and keeps its engines internal. ";
law += "Apply that shape to this slug only. A mean-field method that iterates is Driver<scf::Phase> on CPU and, when a device path exists, on CUDA, plus the HF tag. Entries are run(Frame&), run(const Frame&), and forces(Frame&). Periodic stress(Frame&) only when the method already has stress. Parameters are constructor arguments, const after construction. Residuals go to ConvergenceMetrics. A Criterion decides. The method does not assign converged. ";
law += "Outer MD or Minimize mounts the method with on(phase).add on the parent Runtime and accumulates energy and forces. No *Evaluator, no *Config, no private Runtime, no with_gradient flag. ";
law += "Post-HF stays out of MeanFieldMethod. It holds the reference method the way an MD component holds no second MD. run(Frame&) uses that reference, then overwrites kEnergy with the correlated total in eV. forces(Frame&) overwrites kAtom_f* in eV/Å. Hartree stays on the returned result object. If this slug's post-HF method iterates (amplitudes, active-space orbitals), that iteration is a Driver with typed stages and a Criterion. Do not add a free for-loop that declares convergence. ";
law += "KRHF (CPU and CUDA), KUHF, ComplexGHF, and DHF are already Driver<scf::Phase> plus the HF tag on dev. Do not reopen krhf-driver, kuhf-driver, complex-ghf-driver, or hf-dhf-driver. Their nuclear-force throws stay until a slug whose job is forces. Do not insert a category base. Inheritance depth stays ≤ 2. ";
law += "CUDA presents the same class and the same entries. A missing kernel throws and names the kernel. No host ERI fill on a path the device claims. ";
law += "GOLDEN SOURCE: read the named PySCF files on disk and transplant the algorithm. Do not re-derive it. Pin numbers come from tools/pyscf_oracle run offline, never from ATV's own measurement. ";
law += "OUT OF SCOPE: GPW, plane-wave DFT, semiempirical, DFTB, xTB, classical MD, electrodes. Do not edit those trees. Do not pick es-kind-model, es-ghost-atoms, es-bind-dialects, es-cuda-pw-driver, es-cuda-gpw-frame-io, es-red-regressions. Do not fold ADC, MP2, CC, CAS, Hessian, or solvent back into MeanFieldMethod. ";
law += "TESTS, this is absolute. Run only the gtest cases whose source files this slug added or edited, plus the one regression case this slug owns. ";
law += "Unit tests live in atv_tests (CUDA in atv_tests_cuda). Regressions are one ctest per case directory, named domain/case, for example scf/mp2. Run that case with ctest -R '^scf/mp2$' or atv_regressions --case scf/mp2. A case may also take --section. ";
law += "Build with cmake --build <dir> --target and only these targets: atomiverse, atv_tests, atv_tests_cuda, atv_tests_mpi, atv_regressions, or the single dedicated unit executable the diff touches. ";
law += "Forbidden: ctest with no -R; ctest -L regression; ctest -L cuda; ctest -LE regression; a gtest filter of *; a regex of .* or empty; the domain filter test_cpu_(scf|linalg)_; cmake --build with no --target; ninja with no target; bare ctest of the whole build directory. ";
law += "gpu_job.sh with an empty ATV_CTEST_RE runs every CUDA unit test. That is forbidden. Export ATV_TARGETS to the one binary and ATV_CTEST_RE to the anchored regex. Never #SBATCH --exclusive. Never request a full node. Login nodes have no CUDA toolkit. A CPU run does not make a device path green. ";
law += "CPU configure, from the worktree, in bash -lc: module load buildenv-gcc/2026.03-mpich; export OPENBLAS_ROOT=$OPENBLASDIR; export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu; export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH; cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_HF=ON -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF -DBLA_VENDOR=OpenBLAS. ";
law += "tests/ stays single-function. An SCF or correlation solve goes to regressions/scf/<case>/ as an out-of-library consumer. Update the matching row in docs/roadmap/hartree-fock.md or docs/roadmap/post-hf.md. Do not write wave ids, slugs, or job ids into docs/. ";

let pyscf_root = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts";
let wave = 0;
let landed = [];
let last_slug = "";

while wave < max_waves {
    wave += 1;
    phase("Census");
    let census_prompt = law;
    census_prompt += "You are GapCensus for Gaussian HF and post-HF, wave " + wave.to_string() + ". ";
    census_prompt += "Use grep and read_file. Read docs/roadmap/hartree-fock.md, docs/roadmap/post-hf.md, .claude/specs/es-unification.md, include/atomiverse/cpu/scf/mean_field_method.h, include/atomiverse/cpu/scf/post_hf.h, include/atomiverse/cpu/dynamics/md.h, include/atomiverse/cuda/scf/rhf.cuh. ";
    census_prompt += "Find PySCF under " + pyscf_root + " (pyscf/scf, pyscf/mp, pyscf/cc, pyscf/mcscf, pyscf/fci). Name the files you opened in pyscf_files. ";
    census_prompt += "Pick ONE slug. Prefer a Gaussian HF or post-HF type whose public shape is not the MD preset (loop is not Driver<scf::Phase>, CUDA front missing run(const Frame&), post-HF run() does not overwrite kEnergy in eV, a free for-loop declares convergence). ";
    census_prompt += "If that shape is already in place for the open rows, pick one ○ cell on those two roadmap pages. A foothold is not closed. ";
    census_prompt += "kind=architecture or kind=capability. domain=hf or domain=posthf. regression_case is the ctest case directory, such as scf/mp2 or scf/rhf. ";
    census_prompt += "Do not pick a GPW, PW, or semiempirical row. Do not rename a closed slug. Do not pick krhf-driver, kuhf-driver, complex-ghf-driver, or hf-dhf-driver; those Driver reshapes are on dev. gaps_remaining counts open ○ cells on those two pages plus HF/post-HF types that still miss the MD shape. ";
    if slug_arg != "" {
        census_prompt += "The operator forced slug " + slug_arg + ". Use that slug. ";
    }
    census_prompt += "Return slug, kind, domain, pyscf_files, classic_systems, regression_case, summary, gaps_remaining.";

    let census = agent(census_prompt, #{
        label: "census:" + wave.to_string(),
        capability_mode: "read-only",
        output_schema: census_schema,
    });
    if census == () || census.success != true || census.output == () || census.output.slug == () || census.output.slug == "" {
        log("census failed on wave " + wave.to_string());
        complete(#{ status: "blocked", wave: wave, landed: landed, summary: "census failed" });
    }
    let slug = census.output.slug;
    if slug == last_slug && slug_arg == "" {
        log("census repeated " + slug + "; stop");
        complete(#{ status: "stall", wave: wave, landed: landed, summary: "same slug twice" });
    }
    last_slug = slug;
    let gaps = 1;
    if census.output.gaps_remaining != () { gaps = census.output.gaps_remaining; }
    if gaps == 0 {
        complete(#{ status: "ok", wave: wave, landed: landed, summary: "no gaps" });
    }
    let kind = "capability";
    if census.output.kind != () { kind = census.output.kind; }
    let pyscf_files = "";
    if census.output.pyscf_files != () { pyscf_files = census.output.pyscf_files; }
    let classic = "H2 STO-3G, H2O 6-31G";
    if census.output.classic_systems != () { classic = census.output.classic_systems; }
    let reg_case = "scf/rhf";
    if census.output.regression_case != () && census.output.regression_case != "" {
        reg_case = census.output.regression_case;
    }
    log("wave " + wave.to_string() + " " + kind + " " + slug + " case " + reg_case);

    phase("Implement");
    let impl_prompt = law;
    impl_prompt += "Implement slug " + slug + " (kind " + kind + ") on a new branch wf/hf-" + slug + " from " + target_branch + ". ";
    impl_prompt += "Work in the isolation worktree. Read these PySCF files first and cite the functions you transplanted: " + pyscf_files + ". ";
    impl_prompt += "Match the MD preset described in the law. CPU and CUDA both change when the slug is a product path that already has a device twin. ";
    impl_prompt += "If kind=architecture, move the type onto that shape and delete the dialect. Do not add a new method in the same change. ";
    impl_prompt += "You may compile with cmake --build build --target atomiverse. Do not run ctest in this stage. Do not build a target you did not name. ";
    impl_prompt += "Commit on the worktree branch. status=green only if that targeted build succeeds. Put the exact build command in build_command. ";
    impl_prompt += "status=blocked if PySCF has no corresponding source or the slug is outside HF and post-HF. ";
    impl_prompt += "Return status, summary, worktree_path, branch, head_sha, build_command.";
    let impl = agent(impl_prompt, #{
        label: "impl:" + slug,
        capability_mode: "execute",
        isolation_worktree: true,
        output_schema: stage_schema,
    });
    if !accept_build(impl) {
        let why = "implement failed";
        if impl != () && impl.output != () && impl.output.summary != () { why = impl.output.summary; }
        log(why);
        phase("Land");
        let drop_prompt = "Remove the isolation worktree and local branch wf/hf-" + slug + " if they exist. Do not reset " + target_branch + ". Do not delete unrelated worktrees. Delete that worktree's build directory only. Return status and summary.";
        agent(drop_prompt, #{ label: "drop:" + slug, capability_mode: "execute", output_schema: stage_schema });
        if mode == "burst" {
            complete(#{ status: "blocked", slug: slug, summary: why, landed: landed });
        }
    } else {
        let tree = "";
        if impl.output.worktree_path != () { tree = impl.output.worktree_path; }
        let branch = "wf/hf-" + slug;
        if impl.output.branch != () && impl.output.branch != "" { branch = impl.output.branch; }

        phase("Test");
        let test_prompt = law;
        test_prompt += "In worktree " + tree + " branch " + branch + ", test slug " + slug + ". cd to that worktree for every command. ";
        test_prompt += "From the diff against " + target_branch + ", list the test sources you changed. Build only atv_tests or atv_tests_cuda if a .cu test changed, with cmake --build build --target. ";
        test_prompt += "Run ctest -R with an anchored regex of those gtest names only, or the binary with --gtest_filter set to those names. ";
        test_prompt += "Do not run the regression in this stage. Do not run any other test. ";
        test_prompt += "status=green only if those cases passed. Put every command you ran in test_commands. A green with a full-suite command is rejected. ";
        test_prompt += "Return status, summary, worktree_path, branch, head_sha, test_commands.";
        let tested = agent(test_prompt, #{
            label: "test:" + slug,
            capability_mode: "execute",
            output_schema: stage_schema,
        });
        let ok = accept_tested(tested);

        if ok {
            phase("Validate");
            let val_prompt = law;
            val_prompt += "In worktree " + tree + ", validate slug " + slug + " on " + classic + ". ";
            val_prompt += "The regression case is " + reg_case + ". Add or adjust that case under regressions/" + reg_case + "/ against an offline PySCF pin from tools/pyscf_oracle. ";
            val_prompt += "Build with cmake --build build --target atv_regressions. Run only that case: ctest --test-dir build --output-on-failure -R '^" + reg_case + "$' or ./build/atv_regressions --case " + reg_case + ". ";
            val_prompt += "Do not pass -L regression. Do not run a second case. ";
            val_prompt += "CPU energy or forces must match the pin within the tolerance already used for that observable. ";
            val_prompt += "If the slug touches CUDA, submit tools/gpu_job.sh with ATV_TARGETS and ATV_CTEST_RE set to this case. No --exclusive. If this machine cannot reach a GPU node, status=blocked and say cuda_not_run. Do not mark the device path green from the host. ";
            val_prompt += "Return status, summary, worktree_path, branch, head_sha, test_commands.";
            let val = agent(val_prompt, #{
                label: "validate:" + slug,
                capability_mode: "execute",
                output_schema: stage_schema,
            });
            ok = accept_tested(val);
        }

        if ok {
            phase("Simplify");
            let sim_prompt = law;
            sim_prompt += "In worktree " + tree + ", simplify only the diff of branch " + branch + " against " + target_branch + ". ";
            sim_prompt += "Delete duplicated host/device loops and pass-through wrappers. Keep the MD-shaped entries. ";
            sim_prompt += "Re-run the same scoped unit tests and the same regression case " + reg_case + ". No other tests. ";
            sim_prompt += "status=green only if those commands passed. Put them in test_commands. Return status, summary, worktree_path, branch, head_sha, test_commands.";
            let sim = agent(sim_prompt, #{
                label: "simplify:" + slug,
                capability_mode: "execute",
                output_schema: stage_schema,
            });
            ok = accept_tested(sim);
        }

        if ok {
            phase("Manual");
            let man_prompt = law;
            man_prompt += "In worktree " + tree + ", document slug " + slug + ". ";
            man_prompt += "Update docs/roadmap/hartree-fock.md or docs/roadmap/post-hf.md so the row matches the headers you just read. ";
            man_prompt += "If the user-facing call changed, add one short tutorial section with Frame in and Frame out. ";
            man_prompt += "Commit on " + branch + ". Do not run the test suite. status=green when the docs match the symbols. Return status, summary, branch, head_sha.";
            let man = agent(man_prompt, #{
                label: "manual:" + slug,
                capability_mode: "execute",
                output_schema: stage_schema,
            });
            ok = stage_ok(man);
        }

        phase("Land");
        if ok {
            let land_prompt = law;
            land_prompt += "Merge branch " + branch + " from worktree " + tree + " into " + target_branch + " with a merge commit. ";
            land_prompt += "No force-push. If the merge conflicts, status=blocked and leave the branch. ";
            land_prompt += "On success, delete that worktree and the merged local branch, and remove only that worktree's build directory. ";
            land_prompt += "Do not run ctest. Return status and summary.";
            let land = agent(land_prompt, #{
                label: "land:" + slug,
                capability_mode: "execute",
                output_schema: stage_schema,
            });
            if stage_ok(land) {
                landed.push(slug);
                log("landed " + slug);
            } else {
                log("merge failed for " + slug);
                if mode == "burst" {
                    complete(#{ status: "blocked", slug: slug, landed: landed, summary: "merge failed" });
                }
            }
        } else {
            let drop_prompt = "Tests or validation failed for " + slug + " in " + tree + ". Leave branch " + branch + ". Delete that worktree's build directory only. Do not reset " + target_branch + ". Do not run ctest. Return status and summary.";
            agent(drop_prompt, #{ label: "drop:" + slug, capability_mode: "execute", output_schema: stage_schema });
            if mode == "burst" {
                complete(#{ status: "blocked", slug: slug, landed: landed, summary: "stage failed" });
            }
        }
    }
}

complete(#{
    status: "ok",
    waves: wave,
    landed: landed,
    summary: landed.len().to_string() + " slugs landed",
});
