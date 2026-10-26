// Align Gaussian HF and post-HF with the classical MD Driver contract.
// One slice per wave, on the current checkout. No worktrees, no merge.
// Tests rebuild only dirty objects plus one unit-test translation unit.

let meta = #{
    name: "hf-md-align",
    description: "Align Gaussian HF and post-HF with the MD Driver contract, one slice at a time",
    when_to_use: "Reshape cpu/scf HF and post-HF to the MD loop shape. Compile only dirty objects. No full tests, no regressions.",
    phases: [
        #{ title: "Census", detail: "pick the next MD-alignment slice" },
        #{ title: "Implement", detail: "edit the current tree" },
        #{ title: "Test", detail: "dirty library objects plus one unit test" },
    ],
};

let max_waves = 3;
if args != () {
    if args.max_waves != () { max_waves = args.max_waves; }
}
if max_waves < 1 { max_waves = 1; }
if max_waves > 6 { max_waves = 6; }

let slice_schema = #{
    "type": "object",
    "required": ["slug", "summary", "gaps_remaining"],
    "properties": #{
        "slug": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "files": #{ "type": "string" },
        "test_file": #{ "type": "string" },
        "gaps_remaining": #{ "type": "integer" },
    },
};
let impl_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "test_file": #{ "type": "string" },
        "gtest_filter": #{ "type": "string" },
    },
};
let test_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
    },
};

let law = "";
law += "REPO: /Users/roykid/work/molcrafts/Atomiverse. Stay on the current branch and the current checkout. ";
law += "Do not commit, push, merge, stash, reset, or create a worktree. Do not revert unrelated dirty files (SCC SccDriver and NDDO NddoConverged are in this tree). ";
law += "Do not edit .claude/specs/minimizer-backend-alignment.md. ";
law += "Read CLAUDE.md (Iron laws, Architecture, 'A loop is a Driver') and .claude/specs/es-unification.md before editing. ";
law += "MD CONTRACT, copy this shape: a loop is cpu::Driver<Phase> with typed stages; a Criterion is polled after the iteration and is the only thing that sets converged; the algorithm publishes residuals and does not assign converged; ";
law += "because the poll stops the next iteration, accept-versus-continue for this iteration lives in the stage; nesting uses the parent Runtime (build_plan, step_plan, destroy_plan, swap context.loop) and accumulates; standalone uses setup_pass and teardown_pass; ";
law += "parameters are constructor arguments; Frame is the only public I/O (run/forces/stress, span columns, meta.get); one bind rule is outer.on(phase).add(node). ";
law += "MeanFieldMethod is already that SCF loop. Do not redesign it. ";
law += "SCOPE: include/atomiverse/cpu/scf, src/cpu/scf, and the one unit test under tests/cpu/scf that the slice needs. ";
law += "OUT: GPW, PW, DFTB, xTB, NDDO, classical MD, CUDA, new methods, PySCF capability gaps, regressions/, benchmarks/, a new build directory. ";
law += "Do not wrap a Davidson subspace iteration or a scalar Newton solve as Driver phases. Those stay math kernels. ";
law += "Numerics stay put: same equations, same thresholds, same throw strings. ";
law += "SLICE ORDER, take the first one that the code still violates, and no other: ";
law += "1. posthf-no-private-scf — src/cpu/scf/post_hf.cpp drive_converged still calls mf.run. Post-HF (MP2, MP3, MP4, CCSD, CC2, CISD, FCI, CASCI, CASSCF, NEVPT2, ADC, G0W0, GF2, DirectRPA, BSE, TDA, TDHF, and the other types in post_hf.h) must read a mean field the caller already converged. Refuse with the existing throw when !mf.converged(). Delete the private mf.run. Update only unit tests that do not compile, by converging the reference in the test setup when that setup is already a single-function test. Do not add an end-to-end SCF under tests/. ";
law += "2. posthf-kenergy — a post-HF entry that writes Frame energy writes the correlated total in eV to kEnergy. Hartree stays on the result object. ";
law += "3. posthf-ctor-params — frozen, os_scale, ss_scale, and active-space counts that are the method's identity become constructor arguments. Do not introduce a Config bag. ";
law += "4. ccsd-criterion — the CCSD amplitude loop in src/cpu/scf/ccsd.cpp publishes its residual; a Criterion sets the verdict. Leave the tensor contractions in the existing functions. One family only. ";
law += "5. tdhf-criterion — same verdict split for the TDA/TDHF outer loop in src/cpu/scf/tdhf.cpp. Do not retile Davidson. ";
law += "TESTS: tests/ holds unit tests only. ";
law += "BUILD DIR: /tmp/atv-architecture-pw (this repo, ATV_BUILD_HF=ON, Unix Makefiles). Do not reconfigure. Do not use /tmp/atv-agent-tb (HF is OFF) or /tmp/atv-consol-build (different source). This is the Mac workbench: /usr/bin/c++ is already in the cache. Do not module load. Do not call Arrhenius. ";
law += "COMPILE: cmake --build /tmp/atv-architecture-pw --target atomiverse -j 4. That rebuilds dirty objects only. ";
law += "Then compile and link only the one test .cpp this slice changed or added. Read CXX_DEFINES, CXX_INCLUDES, and CXX_FLAGS from /tmp/atv-architecture-pw/tests/CMakeFiles/atv_tests.dir/flags.make. ";
law += "Link that single object with libatomiverse.a plus the libraries after '-o atv_tests' in tests/CMakeFiles/atv_tests.dir/link.txt. ";
law += "Run that binary with --gtest_filter for its suite only. OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, OMPI_MCA_mpi_yield_when_idle=1. ";
law += "FORBIDDEN: ctest, cmake --build without --target, target atv_tests, target atv_regressions, -j greater than 4, a second compile at the same time, regressions. ";
law += "If the slice changes no test, the atomiverse link is the only build. ";
law += "After a landed slice, rewrite the current sentence in .claude/specs/es-unification.md and .claude/notes/es-architecture-debt.md. Do not append history. Do not close GPW or PW items. ";

let wave = 0;
let landed = [];
let stopped = "";
while wave < max_waves {
    phase("Census");
    let census_prompt = law;
    census_prompt += "TASK: Census only. Use grep and read_file on src/cpu/scf/post_hf.cpp, include/atomiverse/cpu/scf/post_hf.h, src/cpu/scf/ccsd.cpp, and src/cpu/scf/tdhf.cpp. ";
    census_prompt += "Return the first slice in SLICE ORDER that the code still violates. ";
    census_prompt += "slug is that slice id. files is the source files to edit. test_file is the one existing unit test to update, or empty. ";
    census_prompt += "gaps_remaining is how many of the five slices are still open, counting the one you picked. ";
    census_prompt += "If none of the five are open, slug is empty and gaps_remaining is 0. Do not invent a sixth slice. Do not edit files.";
    let census = agent(census_prompt, #{
        label: "census",
        capability_mode: "read-only",
        output_schema: slice_schema,
    });
    if census == () || !census.success || census.output == () {
        stopped = "census failed";
        break;
    }
    let slug = "";
    let summary = "";
    let files = "";
    let test_file = "";
    let gaps = 1;
    if census.output.slug != () { slug = census.output.slug; }
    if census.output.summary != () { summary = census.output.summary; }
    if census.output.files != () { files = census.output.files; }
    if census.output.test_file != () { test_file = census.output.test_file; }
    if census.output.gaps_remaining != () { gaps = census.output.gaps_remaining; }
    log("wave " + wave.to_string() + " slice " + slug + " gaps " + gaps.to_string());
    if slug == "" || gaps == 0 {
        stopped = "no open slice";
        break;
    }

    phase("Implement");
    let impl_prompt = law;
    impl_prompt += "TASK: Implement exactly one slice, then stop. slug=" + slug + ". files=" + files + ". test_file=" + test_file + ". ";
    impl_prompt += "Census summary: " + summary + ". ";
    impl_prompt += "Read the files before editing. Edit only what this slice needs. ";
    impl_prompt += "Do not compile. The next phase compiles. ";
    impl_prompt += "status is ok when the slice is edited, blocked when the slice would require a regression or an end-to-end SCF under tests/. ";
    impl_prompt += "Return test_file and gtest_filter for the one suite the test phase must run. Empty test_file means library-only.";
    let impl = agent(impl_prompt, #{
        label: "implement:" + slug,
        capability_mode: "read-write",
        output_schema: impl_schema,
    });
    if impl == () || !impl.success || impl.output == () {
        stopped = "implement failed for " + slug;
        break;
    }
    let impl_status = "";
    if impl.output.status != () { impl_status = impl.output.status; }
    if impl_status != "ok" {
        stopped = "implement blocked " + slug;
        if impl.output.summary != () { stopped += " " + impl.output.summary; }
        break;
    }
    if impl.output.test_file != () && impl.output.test_file != "" {
        test_file = impl.output.test_file;
    }
    let gtest_filter = "";
    if impl.output.gtest_filter != () { gtest_filter = impl.output.gtest_filter; }

    phase("Test");
    let test_prompt = law;
    test_prompt += "TASK: Compile and run only this slice. slug=" + slug + ". test_file=" + test_file + ". gtest_filter=" + gtest_filter + ". ";
    test_prompt += "Follow COMPILE exactly. Paste the gtest PASSED line or the compiler error into summary. ";
    test_prompt += "status is ok only when the atomiverse link succeeded and, if test_file is non-empty, that one filter passed. ";
    test_prompt += "status is failed when the compiler or the filter fails. Do not start a second, broader build to make it pass.";
    let tested = agent(test_prompt, #{
        label: "test:" + slug,
        capability_mode: "all",
        output_schema: test_schema,
    });
    if tested == () || !tested.success || tested.output == () {
        stopped = "test agent failed for " + slug;
        break;
    }
    let test_status = "";
    if tested.output.status != () { test_status = tested.output.status; }
    if test_status != "ok" {
        let fix_prompt = law;
        fix_prompt += "TASK: The narrow build or the single gtest filter failed for slug=" + slug + ". ";
        fix_prompt += "Failure: ";
        if tested.output.summary != () { fix_prompt += tested.output.summary; }
        fix_prompt += ". Fix the cause in the slice. Do not compile. status ok or blocked.";
        let fix = agent(fix_prompt, #{
            label: "fix:" + slug,
            capability_mode: "read-write",
            output_schema: impl_schema,
        });
        if fix == () || !fix.success || fix.output == () || fix.output.status != "ok" {
            stopped = "fix failed for " + slug;
            break;
        }
        let retest_prompt = law;
        retest_prompt += "TASK: Recompile only slug=" + slug + " the same narrow way. test_file=" + test_file + ". gtest_filter=" + gtest_filter + ". ";
        retest_prompt += "status ok only if that build and filter pass.";
        let retest = agent(retest_prompt, #{
            label: "retest:" + slug,
            capability_mode: "all",
            output_schema: test_schema,
        });
        if retest == () || !retest.success || retest.output == () || retest.output.status != "ok" {
            stopped = "retest failed for " + slug;
            break;
        }
    }
    landed.push(slug);
    wave += 1;
}

let done_list = "";
for s in landed {
    if done_list != "" { done_list += ", "; }
    done_list += s;
}
if done_list == "" { done_list = "(none)"; }
let final_summary = "landed: " + done_list + ". stop: " + stopped;
log(final_summary);
complete(#{ summary: final_summary, landed: landed });
