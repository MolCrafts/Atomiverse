// cp2k-perf-loop — CP2K-anchored optimization of the landed GPW/PW surface
//
// Rewrite of gpw-perf-loop with CP2K as the HARD anchor, not an optional
// reference: every landing is measured as T_atv/T_cp2k on a matched twin
// against a pinned real-CP2K wall (provenance per oracle-pin-provenance).
// Per wave: Baseline? → Census → Assign → Worktrees → Merge? → Clearup
// (always) → Harness?  Stops early with "cp2k_matched" when every baseline
// twin reaches target_ratio.
// Design: docs/developer/cp2k-perf-loop.md + .claude/notes/gpw-perf-architecture.md

let meta = #{
    name: "cp2k-perf-loop",
    description: "CP2K-anchored perf loop: land only measured T_atv/T_cp2k improvements on matched twins; wipe builds every wave",
    when_to_use: "Close the wall gap vs real CP2K on landed GPW/PW; no capability expansion (that is cp2k-method-loop)",
    phases: [
        #{ title: "Bootstrap", detail: "parent branch + paths" },
        #{ title: "Recover", detail: "incomplete perf tips first" },
        #{ title: "Baseline", detail: "pin real-CP2K walls + parent ATV walls per twin" },
        #{ title: "Census", detail: "re-rank hotspots by T_atv/T_cp2k contribution" },
        #{ title: "Assign", detail: "incomplete ≫ pending" },
        #{ title: "Worktrees", detail: "impl + correctness + FD-force + ratio gate" },
        #{ title: "Merge", detail: "greens → target_branch" },
        #{ title: "Clearup", detail: "worktrees + merged tips + wipe ALL builds" },
        #{ title: "Harness", detail: "queue/handoff + target-ratio check" },
    ],
};

// ── CONFIG ──────────────────────────────────────────────────────────────
let loop_id = "cp2k-perf";
let queue_path = "docs/developer/gpw-perf-queue.md";
let loop_doc = "docs/developer/cp2k-perf-loop.md";
let arch_note = ".claude/notes/gpw-perf-architecture.md";
let baseline_note = ".claude/notes/gpw-perf-baseline.md";
let provenance_note = ".claude/notes/oracle-pin-provenance.md";
// wf/perf- kept so Recover picks up gpw-perf-loop's in-flight tips seamlessly.
let branch_prefix = "wf/perf-";
let cp2k_env = "/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh";
// Fair twins (queue table): wall = SAME single twin SCF job, 1 MPI × OMP=4.
let baseline_twins = "gpw-uks-o2-gamma,gpw-kpoints-si,gpw-multigrid-kpoints,ot-minimizer-water";
let omp_fixed = 4;
let cmake_flags = "-DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS";
let test_filter = "^test_cpu_(gpw|pw|scf|xc)_|^test_xc_";
let wipe_cmd = "rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null; rm -rf /tmp/atv-* /tmp/cp2k_oracle_* 2>/dev/null; true";
let default_staff_count = 4;
let default_max_waves = 12;
let default_target_ratio = 1.0; // goal: MATCH/EXCEED CP2K (T_atv/T_cp2k ≤ 1)

// ── Args ────────────────────────────────────────────────────────────────
let mode = "exhaust";
let staff_count = default_staff_count;
let max_waves = default_max_waves;
let min_budget = 10;
let do_wall = true;
let require_wall_gain = false; // pure extraction may land without gain if no regression
let target_ratio = default_target_ratio;
let skip_implement = false;
let skip_recover = false;
let skip_baseline = false;
let max_consecutive_fail_waves = 3;
let stall_no_work = 2;
let target_branch = "dev";
let theme = "";

if args != () {
    if args.mode != () { mode = args.mode; }
    if args.staff_count != () { staff_count = args.staff_count; }
    if args.parallelism != () { staff_count = args.parallelism; }
    if args.max_waves != () { max_waves = args.max_waves; }
    if args.min_budget != () { min_budget = args.min_budget; }
    if args.do_wall != () { do_wall = args.do_wall; }
    if args.require_wall_gain != () { require_wall_gain = args.require_wall_gain; }
    if args.target_ratio != () { target_ratio = args.target_ratio; }
    if args.skip_implement != () { skip_implement = args.skip_implement; }
    if args.skip_recover != () { skip_recover = args.skip_recover; }
    if args.skip_baseline != () { skip_baseline = args.skip_baseline; }
    if args.max_consecutive_fail_waves != () { max_consecutive_fail_waves = args.max_consecutive_fail_waves; }
    if args.stall_no_work != () { stall_no_work = args.stall_no_work; }
    if args.target_branch != () { target_branch = args.target_branch; }
    if args.theme != () { theme = args.theme; }
}
if do_wall == false { require_wall_gain = false; }
if mode != "exhaust" && mode != "burst" { mode = "exhaust"; }
if mode == "burst" && (args == () || args.max_waves == ()) { max_waves = 2; }
if staff_count < 1 { staff_count = 1; }
if staff_count > 8 { staff_count = 8; }
if max_waves < 1 { max_waves = 1; }
if max_waves > 30 { max_waves = 30; }
if min_budget < 8 { min_budget = 8; }

// ── Schemas ─────────────────────────────────────────────────────────────
let item_props = #{
    "slug": #{ "type": "string" },
    "title": #{ "type": "string" },
    "phase": #{ "type": "string" },
    "files": #{ "type": "string" },
    "rationale": #{ "type": "string" },
    "twin": #{ "type": "string" },
    "resume_from": #{ "type": "string" },
    "is_resume": #{ "type": "boolean" },
};
let item_schema = #{
    "type": "object",
    "required": ["slug", "title"],
    "properties": item_props,
};
let batch_schema = #{
    "type": "object",
    "required": ["status", "items"],
    "properties": #{
        "status": #{ "type": "string" },
        "items": #{ "type": "array", "maxItems": 8, "items": item_schema },
        "summary": #{ "type": "string" },
    },
};
let recover_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "items": #{ "type": "array", "maxItems": 8, "items": item_schema },
        "incomplete_count": #{ "type": "integer" },
    },
};
let baseline_schema = #{
    "type": "object",
    "required": ["status", "summary", "builds_wiped"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "twins_pinned": #{ "type": "integer" },
        "cp2k_ran": #{ "type": "boolean" },
        "worst_ratio": #{ "type": "number" },
        "builds_wiped": #{ "type": "boolean" },
    },
};
let census_schema = #{
    "type": "object",
    "required": ["status", "pending_count", "gaps_remaining", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "pending_count": #{ "type": "integer" },
        "gaps_remaining": #{ "type": "integer" },
        "new_slugs_added": #{ "type": "integer" },
        "summary": #{ "type": "string" },
    },
};
let worker_schema = #{
    "type": "object",
    "required": ["status", "slug", "summary", "acceptance_met", "builds_wiped"],
    "properties": #{
        "status": #{ "type": "string" },
        "slug": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "acceptance_met": #{ "type": "boolean" },
        "correctness_green": #{ "type": "boolean" },
        "twin": #{ "type": "string" },
        "t_cp2k": #{ "type": "number" },
        "t_cp2k_source": #{ "type": "string" },
        "ratio_before": #{ "type": "number" },
        "ratio_after": #{ "type": "number" },
        "wall_improved": #{ "type": "boolean" },
        "wall_regression": #{ "type": "boolean" },
        "extraction_only": #{ "type": "boolean" },
        "touched_energy_path": #{ "type": "boolean" },
        "fd_force_checked": #{ "type": "boolean" },
        "bit_exact": #{ "type": "boolean" },
        "tolerance_ok": #{ "type": "boolean" },
        "builds_wiped": #{ "type": "boolean" },
        "head_sha": #{ "type": "string" },
        "commit_message": #{ "type": "string" },
        "worktree_path": #{ "type": "string" },
        "block_reason": #{ "type": "string" },
    },
};
let merge_schema = #{
    "type": "object",
    "required": ["status", "summary", "merged", "builds_wiped"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "merged": #{ "type": "array", "items": #{ "type": "string" } },
        "failed": #{ "type": "array", "items": #{ "type": "string" } },
        "dev_head": #{ "type": "string" },
        "builds_wiped": #{ "type": "boolean" },
    },
};
let clearup_schema = #{
    "type": "object",
    "required": ["status", "summary", "builds_wiped"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "worktrees_removed": #{ "type": "array", "items": #{ "type": "string" } },
        "branches_pruned": #{ "type": "array", "items": #{ "type": "string" } },
        "builds_wiped": #{ "type": "boolean" },
        "build_paths_removed": #{ "type": "array", "items": #{ "type": "string" } },
    },
};
let harness_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "done_slugs": #{ "type": "array", "items": #{ "type": "string" } },
        "pending_remaining": #{ "type": "integer" },
        "worst_ratio": #{ "type": "number" },
        "twins_at_target": #{ "type": "boolean" },
    },
};

// ── Shared law ──────────────────────────────────────────────────────────
let law = "";
law += "LOOP=cp2k-perf: OPTIMIZE the landed CP2K-aligned CPU surface (cpu/gpw + coupled pw/xc/grid helpers) ONLY. ";
law += "No new electronic methods, no capability expansion (cp2k-method-loop owns that), no CUDA. ";
law += "Binding arch: " + arch_note + " + " + loop_doc + " + CLAUDE.md iron laws. ";
law += "CP2K ANCHOR (binding): the metric is T_atv/T_cp2k on a MATCHED twin, 1 MPI x OMP=" + omp_fixed.to_string() + ", same single SCF job. ";
law += "T_cp2k must be a REAL CP2K run pinned in " + baseline_note + " with provenance (binary+version+input+output line, see " + provenance_note + "); ";
law += "CP2K build: source " + cp2k_env + " -> cp2k.psmp. Never estimate, never reuse a stale pin after twin inputs change, never self-golden from ATV. ";
law += "FD-FORCE LAW (standing lesson): any change touching the ENERGY path (collocate/integrate/Poisson/XC/V_eff/PP) must re-run the FD force oracle — energy pins cannot see a force/energy functional mismatch. ";
law += "SIMD FIRST-CLASS (2026-08-03): portable xsimd is a primary optimization lane at ANY priority (native_integrals.cpp / multigrid.cpp axis_dot idiom — raw spans, scalar tail, single source); arch intrinsics (_mm512_*/NEON) and per-arch dispatch remain FORBIDDEN (tree must build x86_64 + aarch64; CP2K's AVX2-only kernel falls back to scalar on aarch64 — xsimd beats that trade); ";
law += "NEVER set global XTENSOR_USE_XSIMD (breaks reshape_view in xc_grid_engine.h). ";
law += "WORKER RUNTIME LAW (two workers died to this): verify the worktree base FIRST — branch wf/<slug> from `" + target_branch + "` explicitly and confirm target files exist (auto-worktrees can sit at a stale snapshot). NEVER end a turn waiting on a background event (no background builds, no monitors, no 'waiting for X' sign-offs) — a worker that ends its turn is terminated. Long builds run FOREGROUND with a timeout and are re-run to completion (incremental resume). ";
law += "BIT-EXACT LAW: default gate is bit-identical vs current pins. A change that reorders FP summation (vectorized reductions etc.) may land ONLY with a stated numeric tolerance vs the CP2K pins AND a fresh FD-force check, documented in the commit + queue row. Never silently soften a tol. ";
law += "Parent `" + target_branch + "`; workers ONLY on " + branch_prefix + "<slug>; no push; no reset --hard. ";
law += "DISK LAW (shared nobackup): after EVERY cmake/build/ctest, and before exit: " + wipe_cmd + " ";
law += "Also wipe any build* under isolation worktrees you created. Never leave multi-GB builds. ";
law += "Never delete sources, regressions/, .claude/specs/, tools/*_oracle/inputs, .git. ";
law += "Correctness: do not soften tols or self-golden pins. Public API: Frame only; no *Config bags; no make_*/build_* factories. ";
law += "tests/ = unit only; e2e stays in regressions/. Profiling via ATV_GPW_PROF. ";
law += "CLEARUP same wave as Worktrees; no nested long-lived subagents. ";

log(loop_id + " staff=" + staff_count.to_string() + " target=" + target_branch
    + " max_waves=" + max_waves.to_string() + " target_ratio=" + target_ratio.to_string()
    + " wipe_builds=always");

// ── Bootstrap ───────────────────────────────────────────────────────────
phase("Bootstrap");
let boot_p = law;
boot_p += "BOOTSTRAP parent. Prefer `" + target_branch + "`. Confirm " + queue_path + ", " + loop_doc + ", " + arch_note + ", " + baseline_note + ". ";
boot_p += "Confirm the CP2K install answers: source " + cp2k_env + " && cp2k.psmp --version (record it). ";
boot_p += "If parent has leftover build* or build-*, wipe them now (" + wipe_cmd + "). No push.";
let boot = agent(boot_p, #{ label: "bootstrap", capability_mode: "execute" });
if boot != () && boot.success { log("bootstrap ok"); }

// ── Recover ─────────────────────────────────────────────────────────────
let resume_seed = [];
if skip_recover == false {
    phase("Recover");
    let rec_p = law;
    rec_p += "RECOVER incomplete on `" + target_branch + "`: queue [~], open " + branch_prefix + "* unmerged, half-done perf work. ";
    rec_p += "Normalize to ## in_progress. Pick ≤" + staff_count.to_string() + ". Do not invent capability slugs. Output recover schema.";
    let rec = agent(rec_p, #{ label: "recover", capability_mode: "all", output_schema: recover_schema });
    if rec != () && rec.success && rec.output != () {
        log("recover: " + rec.output.summary);
        if rec.output.items != () { resume_seed = rec.output.items; }
    }
}

// ── Baseline (once): pin CP2K + parent ATV walls per twin ───────────────
let last_worst_ratio = -1.0;
if skip_baseline == false && do_wall {
    phase("Baseline");
    let base_p = law;
    base_p += "BASELINE on `" + target_branch + "`. For EACH twin in [" + baseline_twins + "] ensure " + baseline_note + " holds: ";
    base_p += "(a) a pinned CP2K wall with full provenance (binary+version+input+output line) from a REAL run — if missing or the twin input changed, source " + cp2k_env + " and run cp2k.psmp NOW (1 MPI x OMP=" + omp_fixed.to_string() + "); ";
    base_p += "(b) a current parent ATV wall (scoped build: cmake -B build " + cmake_flags + "; --target atomiverse + the twin reg only; OMP_NUM_THREADS=" + omp_fixed.to_string() + "). ";
    base_p += "Record per-twin ratio T_atv/T_cp2k and worst_ratio. Commit chore(baseline) if the note changed. No push. ";
    base_p += "MANDATORY before exit: " + wipe_cmd + " set builds_wiped=true. Output baseline schema.";
    let base = agent(base_p, #{ label: "baseline", capability_mode: "all", output_schema: baseline_schema });
    if base != () && base.success && base.output != () {
        log("baseline: " + base.output.summary);
        if base.output.worst_ratio != () && base.output.worst_ratio > 0.0 {
            last_worst_ratio = base.output.worst_ratio;
        }
        if base.output.builds_wiped != true { log("baseline: WARNING builds_wiped not true"); }
    } else {
        log("baseline: failed — workers must pin their own twin walls");
    }
}

// ── Waves ───────────────────────────────────────────────────────────────
let wave = 0;
let merged_total = 0;
let consecutive_fail = 0;
let consecutive_no_work = 0;
let stop_reason = "completed";
let used_resume_seed = false;
let last_gaps_remaining = -1;

while wave < max_waves {
    let b = budget();
    if b.remaining < min_budget {
        stop_reason = "budget_low";
        break;
    }
    wave += 1;
    let wn = wave.to_string();
    log("══ " + loop_id + " wave " + wn + " ══");

    // Census — rank by CP2K-ratio contribution
    phase("Census");
    let census_p = law;
    census_p += "PERF-CENSUS wave " + wn + " on `" + target_branch + "`. ";
    census_p += "Read " + arch_note + " + " + baseline_note + " + " + queue_path + ". ";
    census_p += "Re-diff hotspots in cpu/gpw (solver, collocate, multigrid, kpoints, poisson, OT, forces) and RANK by expected contribution to the measured T_atv/T_cp2k of the worst twins — not by aesthetics. ";
    census_p += "Each pending row names its twin from [" + baseline_twins + "]. ";
    census_p += "Rewrite ## pending only (keep done/blocked/in_progress). Prefer P0 then P1 then P2. ";
    census_p += "Do NOT add capability features; do NOT queue arch-intrinsic work; portable-xsimd rows must state the bit-exact/tolerance plan up front. ";
    census_p += "Optional commit chore(queue). No push. ";
    census_p += "JSON: status=ok|empty, pending_count, gaps_remaining, new_slugs_added, summary. empty only if gaps_remaining=0.";
    let census = agent(census_p, #{
        label: "census-w" + wn,
        capability_mode: "all",
        output_schema: census_schema,
    });
    if census != () && census.success && census.output != () {
        log("census: " + census.output.summary);
        if census.output.gaps_remaining != () { last_gaps_remaining = census.output.gaps_remaining; }
    } else {
        log("census: failed");
    }

    // Assign
    phase("Assign");
    let items = [];
    if used_resume_seed == false && resume_seed.len() > 0 {
        used_resume_seed = true;
        let n = resume_seed.len();
        if n > staff_count { n = staff_count; }
        let i = 0;
        while i < n { items.push(resume_seed[i]); i += 1; }
        log("assign: resume_seed n=" + items.len().to_string());
    } else {
        let asg_p = law;
        asg_p += "ASSIGN wave " + wn + ". Read " + queue_path + ". Priority: [~] then pending (census rank order). Up to "
            + staff_count.to_string() + ". Prefer disjoint file surfaces (avoid two workers editing gpw_solver.cpp). ";
        asg_p += "Every pick carries its twin. Move picks to [~]. Output batch schema.";
        if theme != "" { asg_p += " Theme: " + theme + "."; }
        let asg = agent(asg_p, #{ label: "assign-w" + wn, capability_mode: "all", output_schema: batch_schema });
        if asg != () && asg.success && asg.output != () && asg.output.items != () {
            items = asg.output.items;
            log("assign: " + json_encode(items));
        }
    }

    if items.len() == 0 {
        consecutive_no_work += 1;
        if last_gaps_remaining == 0 {
            if consecutive_no_work >= stall_no_work { stop_reason = "queue_empty"; break; }
        } else {
            if consecutive_no_work >= stall_no_work { stop_reason = "census_stalled"; break; }
        }
        // still clear leftover builds on stall cycles
        phase("Clearup");
        let idle_c = law + "CLEARUP idle wave " + wn + ": wipe parent build*; prune stray " + branch_prefix + " if fully merged. builds_wiped required.";
        agent(idle_c, #{ label: "clearup-idle-w" + wn, capability_mode: "all", output_schema: clearup_schema });
        continue;
    }
    consecutive_no_work = 0;

    let take = items.len();
    if take > staff_count { take = staff_count; }

    // Worktrees
    phase("Worktrees");
    let jobs = [];
    let ji = 0;
    while ji < take {
        let it = items[ji];
        let slug = it.slug;
        let wp = law;
        wp += "WORKER wave " + wn + " isolation. slug=" + slug + " title=" + it.title + ". ";
        if it.twin != () { wp += "twin=" + it.twin + ". "; }
        wp += "Branch " + branch_prefix + slug + " from `" + target_branch + "`. ";
        wp += "Goal: cut T_atv/T_cp2k per queue row — algorithmic wins and shared-structure extraction first; portable xsimd only with the tolerance plan. ";
        wp += "CMake: bash -lc module load buildenv-gcc/2026.03-mpich; export OPENBLAS_ROOT=$OPENBLASDIR; ";
        wp += "export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH; ";
        wp += "export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu; ";
        wp += "cmake -B build " + cmake_flags + "; cmake --build build -j$(nproc) --target atomiverse <your test_*> <your reg_*>. ";
        wp += "SCOPED build+test only. Correctness: ctest -R '" + test_filter + "' plus any reg you claim; no tol soften. ";
        if do_wall {
            wp += "WALL (mandatory): measure your twin BEFORE and AFTER on this tree (OMP_NUM_THREADS=" + omp_fixed.to_string() + ", same job). ";
            wp += "T_cp2k: take the pinned wall from " + baseline_note + "; if your twin has no valid pin, source " + cp2k_env + " and run the REAL cp2k.psmp twin now, record provenance, and set t_cp2k_source. ";
            wp += "Report ratio_before and ratio_after as T_atv/T_cp2k (absolute). ";
        }
        if require_wall_gain {
            wp += "green REQUIRES wall_improved=true and wall_regression=false. ";
        } else {
            wp += "extraction_only ok if wall_regression=false on your twin. ";
        }
        wp += "If you touched the energy path set touched_energy_path=true and re-run the FD force oracle (fd_force_checked). ";
        wp += "If your change is not bit-identical set bit_exact=false and land only with the documented tolerance vs pins (tolerance_ok). ";
        if skip_implement { wp += "skip_implement: measure/document only. "; }
        wp += "MANDATORY before exit: " + wipe_cmd + " set builds_wiped=true. No nested subagents. Output worker schema.";
        jobs.push(#{
            prompt: wp,
            label: "wt-" + slug,
            capability_mode: "all",
            isolation_worktree: true,
            output_schema: worker_schema,
        });
        ji += 1;
    }
    let results = parallel(jobs);
    log("wave " + wn + ": workers done → clearup owed");

    let greens = [];
    let wave_slugs = [];
    let ri = 0;
    while ri < results.len() {
        let r = results[ri];
        if r != () && r.success && r.output != () {
            if r.output.slug != () { wave_slugs.push(r.output.slug); }
            // accept status green OR done (workers often emit done)
            if r.output.status == "green" || r.output.status == "done" {
                let ok = r.output.acceptance_met == true;
                if ok && r.output.builds_wiped != true { ok = false; }
                if ok && r.output.correctness_green == false { ok = false; }
                if ok && r.output.wall_regression == true { ok = false; }
                // FD-force law: energy-path change without a fresh FD check never lands
                if ok && r.output.touched_energy_path == true && r.output.fd_force_checked != true { ok = false; }
                // bit-exact law: non-bit-identical needs the documented tolerance
                if ok && r.output.bit_exact == false && r.output.tolerance_ok != true { ok = false; }
                // CP2K anchor: a measured absolute ratio OR an honest extraction_only
                if ok && do_wall {
                    let measured = r.output.ratio_after != () && r.output.ratio_after > 0.0;
                    if !measured && r.output.extraction_only != true { ok = false; }
                }
                if ok && require_wall_gain {
                    if r.output.wall_improved != true { ok = false; }
                }
                if ok { greens.push(r.output); }
            }
        }
        ri += 1;
    }
    if wave_slugs.len() == 0 {
        let si = 0;
        while si < take {
            if items[si].slug != () { wave_slugs.push(items[si].slug); }
            si += 1;
        }
    }

    // Merge
    let merged = [];
    if greens.len() > 0 {
        phase("Merge");
        let mp = law;
        mp += "MERGE wave " + wn + " on parent. Candidates:\n" + json_encode(greens) + "\n";
        mp += "merge --no-ff " + branch_prefix + "<slug> into `" + target_branch + "`. No push. ";
        mp += "Conflict hotspots: gpw_solver, collocate, multigrid, kpoints, poisson. ";
        mp += "Merge gate: scoped rebuild + ctest -R '" + test_filter + "'. ";
        mp += "Record each candidate's ratio_after in the queue done row (commit message carries T_atv/T_cp2k). ";
        mp += "MANDATORY: if parent rebuilt, " + wipe_cmd + " builds_wiped=true. Output merge schema.";
        let mres = agent(mp, #{ label: "merge-w" + wn, capability_mode: "all", output_schema: merge_schema });
        if mres != () && mres.success && mres.output != () && mres.output.merged != () {
            merged = mres.output.merged;
            log("merge: " + mres.output.summary);
            if mres.output.builds_wiped != true {
                log("merge: WARNING builds_wiped not true — clearup must finish wipe");
            }
        }
        if merged.len() == 0 {
            consecutive_fail += 1;
            log("merge: empty (fail streak " + consecutive_fail.to_string() + ")");
        } else {
            consecutive_fail = 0;
            merged_total += merged.len();
        }
    } else {
        consecutive_fail += 1;
        log("wave " + wn + ": no greens (fail streak " + consecutive_fail.to_string() + ")");
    }

    // Clearup ALWAYS
    phase("Clearup");
    let cp = law;
    cp += "CLEARUP wave " + wn + ". Staffed: " + json_encode(wave_slugs) + ". Merged: " + json_encode(merged) + ". ";
    cp += "1) git worktree list → under EACH this-wave isolation tree: " + wipe_cmd + " then git worktree remove --force. ";
    cp += "2) git branch -d " + branch_prefix + "<slug> only if fully merged into `" + target_branch + "` (never -D incomplete). ";
    cp += "3) parent and any leftover: find -name build -o -name 'build-*' under repo worktrees → wipe. ";
    cp += "builds_wiped=true REQUIRED. No feature work. Output clearup schema.";
    let cres = agent(cp, #{ label: "clearup-w" + wn, capability_mode: "all", output_schema: clearup_schema });
    if cres != () && cres.success && cres.output != () {
        log("clearup: " + cres.output.summary);
        if cres.output.builds_wiped != true {
            log("clearup: FAILED builds_wiped — disk risk");
        }
    } else {
        log("clearup: FAILED");
    }

    if greens.len() == 0 || merged.len() == 0 {
        if consecutive_fail >= max_consecutive_fail_waves {
            if greens.len() == 0 { stop_reason = "fail_streak"; } else { stop_reason = "merge_empty_streak"; }
            break;
        }
        continue;
    }

    // Harness (+ target-ratio check)
    phase("Harness");
    let hp = law;
    hp += "HARNESS wave " + wn + " after " + json_encode(merged) + ". ";
    hp += "Mark done [x] on " + queue_path + " with each slug's measured ratio_after; keep other [~]; update " + baseline_note + " parent ATV walls for twins whose path merged this wave (scoped rebuild allowed — wipe after); update " + loop_doc + " run log. ";
    if target_ratio > 0.0 {
        hp += "TARGET CHECK: twins_at_target=true ONLY if every twin in [" + baseline_twins + "] has a fresh post-merge measured T_atv/T_cp2k ≤ " + target_ratio.to_string() + " against its pinned CP2K wall — never assert from stale numbers. Report worst_ratio. ";
    }
    hp += "Never claim queue exhausted without census empty. Clearup already ran. ";
    hp += "If you rebuilt parent, wipe build*. Commit chore(harness). No push. Output harness schema.";
    let hres = agent(hp, #{ label: "harness-w" + wn, capability_mode: "all", output_schema: harness_schema });
    if hres != () && hres.success && hres.output != () {
        log("harness: " + hres.output.summary);
        if hres.output.worst_ratio != () && hres.output.worst_ratio > 0.0 {
            last_worst_ratio = hres.output.worst_ratio;
        }
        if target_ratio > 0.0 && hres.output.twins_at_target == true {
            stop_reason = "cp2k_matched";
            break;
        }
    }
}

// Final disk sweep
phase("Clearup");
let final_c = law + "FINAL CLEARUP: wipe any remaining build* under main repo and leftover " + branch_prefix + " worktrees. builds_wiped required.";
agent(final_c, #{ label: "clearup-final", capability_mode: "all", output_schema: clearup_schema });

let b_end = budget();
let report = loop_id + " finished: " + stop_reason
    + " waves=" + wave.to_string()
    + " merged_total=" + merged_total.to_string()
    + " last_worst_ratio=" + last_worst_ratio.to_string()
    + " last_gaps_remaining=" + last_gaps_remaining.to_string()
    + " payroll=" + b_end.remaining.to_string() + "/" + b_end.total.to_string();
log(report);
let report_path = write_scratch_file(loop_id + "-loop-report.md", report);
complete(#{
    status: stop_reason,
    waves: wave,
    merged_total: merged_total,
    last_worst_ratio: last_worst_ratio,
    last_gaps_remaining: last_gaps_remaining,
    staff_count: staff_count,
    target_branch: target_branch,
    target_ratio: target_ratio,
    report: report_path,
    queue: queue_path,
    pattern: "cp2k-perf-loop",
    wipe_builds: true,
});
