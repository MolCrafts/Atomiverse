// metalwalls-method-loop — instance of oracle-capability-loop-template
// Customize CONFIG + extra_law only; machinery matches template.
// Design: docs/developer/oracle-capability-loop.md
// Skill:  .agents/skills/create-oracle-loop/SKILL.md
//
// Per wave:  GapCensus → Assign → Worktrees → Merge? → Clearup → Harness? → Tutorial?
// Clearup always (greens or reds). Stop queue_empty only if gaps_remaining=0 + stall.

let meta = #{
    name: "metalwalls-method-loop",
    description: "Per-wave gap-census vs MetalWalls/map; MD-only worktrees; LAMMPS-first oracle",
    when_to_use: "Ship classical electrochemical MD continuously: implement wave then re-scan MetalWalls gaps",
    phases: [
        #{ title: "Bootstrap", detail: "parent branch + paths" },
        #{ title: "Recover", detail: "incomplete first" },
        #{ title: "GapCensus", detail: "oracle/map vs ATV every wave" },
        #{ title: "Assign", detail: "incomplete ≫ pending" },
        #{ title: "Worktrees", detail: "spec + pin + impl + dual gate" },
        #{ title: "Merge", detail: "greens → target_branch" },
        #{ title: "Clearup", detail: "drop worktrees + merged tips + builds" },
        #{ title: "Harness", detail: "queue/handoff/specs" },
        #{ title: "Tutorial", detail: "optional methods/<slug>.md" },
    ],
};

// ── CONFIG ──────────────────────────────────────────────────────────────
let loop_id = "metalwalls";
let oracle_name = "LAMMPS|MetalWalls";
let queue_path = ".claude/campaigns/metalwalls-method-queue.md";
let handoff_path = ".claude/campaigns/metalwalls-method-handoff.md";
let loop_doc = ".claude/campaigns/metalwalls-method-loop.md";
let map_path = ".claude/notes/metalwalls-capability-map.md";
let oracle_readme = "tools/metalwalls_oracle/README.md";
let tutorial_root = "docs/tutorial/methods/";
let specs_root = ".claude/specs/";
let specs_index = ".claude/specs/INDEX.md";
let regression_hint = "regressions/{md,forcefield}/<slug>/";
let cmake_flags = "-DCMAKE_BUILD_TYPE=Release -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS";
// ctest -R regex bounding THIS loop's domain suite. Merge gate only.
// Workers build/run ONLY the test binaries they touched + reg_<slug>.
let test_filter = "^test_cpu_(md|mc|pair|fix|observer|ewald|pme|pppm|slab|gaussian_charge|bond|dihedral|improper|eam|sw|pim|charging|classical)";
let scope_in = "classical electrochemical MD + MetalWalls map E1-E12 + LAMMPS-overlap classical MD";
let scope_out = "SCF/post-HF (pyscf); GPW/PW (cp2k); CUDA electrode product; PLUMED; full 2D-PBC product; cross-lang product ports";
let gap_sources = map_path + "; docs/roadmap/force-fields.md; docs/roadmap/molecular-dynamics.md; LAMMPS/MW overlap features";
let extra_law = "";
extra_law += "ORACLE PRIORITY: if feature also exists in LAMMPS, offline pin + fair bench MUST use LAMMPS; MetalWalls only for MW-only physics. ";
extra_law += "REFERENCE.md must state oracle: lammps|metalwalls|literature and oracle_version. ";
extra_law += "MD-ONLY build (cmake_flags); FORBIDDEN: ATV_BUILD_HF/PW/GPW/CUDA=ON. ";
extra_law += "Arch: electrodes/const-pot = Components on cpu::md::MD; never new Driver category; never charge solve inside KSpace*. ";
extra_law += "Temps: /tmp/atv-* /tmp/lammps_oracle_* /tmp/mw_oracle_*. Map: " + map_path + ". ";
let default_staff_count = 10;
let default_do_perf = true;
let default_require_perf = true;
let default_write_tutorials = true;
let default_skip_recover = false;
// ────────────────────────────────────────────────────────────────────────

// ── Args ────────────────────────────────────────────────────────────────
let mode = "exhaust";
let staff_count = default_staff_count;
let max_waves = 20;
let min_budget = 10;
let do_perf = default_do_perf;
let max_ratio = 10.0;
let require_perf = default_require_perf;
let write_tutorials = default_write_tutorials;
let theme = "";
let skip_implement = false;
let skip_recover = default_skip_recover;
let max_consecutive_fail_waves = 3;
let stall_no_work = 2;
let target_branch = "dev";

if args != () {
    if args.mode != () { mode = args.mode; }
    if args.staff_count != () { staff_count = args.staff_count; }
    if args.parallelism != () { staff_count = args.parallelism; }
    if args.max_waves != () { max_waves = args.max_waves; }
    if args.max_rounds != () { max_waves = args.max_rounds; }
    if args.min_budget != () { min_budget = args.min_budget; }
    if args.do_perf != () { do_perf = args.do_perf; }
    if args.max_ratio != () { max_ratio = args.max_ratio; }
    if args.require_perf != () { require_perf = args.require_perf; }
    if args.write_tutorials != () { write_tutorials = args.write_tutorials; }
    if args.theme != () { theme = args.theme; }
    if args.skip_implement != () { skip_implement = args.skip_implement; }
    if args.skip_recover != () { skip_recover = args.skip_recover; }
    if args.max_consecutive_fail_waves != () { max_consecutive_fail_waves = args.max_consecutive_fail_waves; }
    if args.stall_no_work != () { stall_no_work = args.stall_no_work; }
    if args.target_branch != () { target_branch = args.target_branch; }
}
if do_perf == false { require_perf = false; }
if mode != "exhaust" && mode != "burst" { mode = "exhaust"; }
if mode == "burst" && (args == () || (args.max_waves == () && args.max_rounds == ())) { max_waves = 2; }
if staff_count < 1 { staff_count = 1; }
if staff_count > 20 { staff_count = 20; }
if max_waves < 1 { max_waves = 1; }
if max_waves > 50 { max_waves = 50; }
if min_budget < 8 { min_budget = 8; }

// ── Schemas ─────────────────────────────────────────────────────────────
let item_props = #{
    "slug": #{ "type": "string" },
    "method": #{ "type": "string" },
    "chapter": #{ "type": "string" },
    "geom": #{ "type": "string" },
    "observable": #{ "type": "string" },
    "tolerance": #{ "type": "string" },
    "rationale": #{ "type": "string" },
    "oracle": #{ "type": "string" },
    "spec_path": #{ "type": "string" },
    "resume_from": #{ "type": "string" },
    "is_resume": #{ "type": "boolean" },
};
let item_schema = #{
    "type": "object",
    "required": ["slug", "method"],
    "properties": item_props,
};

let census_schema = #{
    "type": "object",
    "required": ["status", "pending_count", "gaps_remaining", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "pending_count": #{ "type": "integer" },
        "gaps_remaining": #{ "type": "integer" },
        "new_slugs_added": #{ "type": "integer" },
        "roadmap_updated": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
    },
};
let recover_schema = #{
    "type": "object",
    "required": ["status", "items", "incomplete_count", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "incomplete_count": #{ "type": "integer" },
        "summary": #{ "type": "string" },
        "items": #{ "type": "array", "maxItems": 20, "items": item_schema },
    },
};
let batch_schema = #{
    "type": "object",
    "required": ["status", "items"],
    "properties": #{
        "status": #{ "type": "string" },
        "items": #{ "type": "array", "maxItems": 20, "items": item_schema },
        "incomplete_remaining": #{ "type": "integer" },
        "new_assigned": #{ "type": "integer" },
    },
};
let worker_schema = #{
    "type": "object",
    "required": ["status", "slug", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "slug": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "worktree_path": #{ "type": "string" },
        "branch": #{ "type": "string" },
        "head_sha": #{ "type": "string" },
        "spec_path": #{ "type": "string" },
        "acceptance_met": #{ "type": "boolean" },
        "regression_path": #{ "type": "string" },
        "oracle": #{ "type": "string" },
        "ratio_atv_over_oracle": #{ "type": "number" },
        "within_10x": #{ "type": "boolean" },
        "perf_exempt": #{ "type": "boolean" },
        "perf_exempt_reason": #{ "type": "string" },
        "commit_message": #{ "type": "string" },
        "block_reason": #{ "type": "string" },
        "md_only_build": #{ "type": "boolean" },
    },
};
let merge_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "merged": #{ "type": "array", "items": #{ "type": "string" } },
        "failed": #{ "type": "array", "items": #{ "type": "string" } },
        "dev_head": #{ "type": "string" },
    },
};
let clearup_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "worktrees_removed": #{ "type": "array", "items": #{ "type": "string" } },
        "branches_pruned": #{ "type": "array", "items": #{ "type": "string" } },
        "builds_wiped": #{ "type": "integer" },
    },
};
let harness_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "queue_remaining": #{ "type": "integer" },
        "incomplete_remaining": #{ "type": "integer" },
        "roadmap_updated": #{ "type": "boolean" },
        "specs_closed": #{ "type": "array", "items": #{ "type": "string" } },
    },
};
let tutorial_schema = #{
    "type": "object",
    "required": ["status", "slug", "path", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "slug": #{ "type": "string" },
        "path": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "has_accuracy_table": #{ "type": "boolean" },
        "has_speed_table": #{ "type": "boolean" },
        "ratio": #{ "type": "number" },
    },
};
let index_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "index_updated": #{ "type": "boolean" },
        "pages": #{ "type": "integer" },
    },
};

// ── Shared law (injected into every agent prompt) ───────────────────────
let law = "";
law += "Atomiverse unattended. C++17 ATV::; Frame I/O; no public *Config; no make_*/build_* factories. ";
law += "Loop=" + loop_id + " oracle=" + oracle_name + ". IN: " + scope_in + ". OUT: " + scope_out + ". ";
law += "Offline pins only (" + oracle_readme + "); never live oracle in ctest. ";
law += "Spec-first: " + specs_root + "<slug>/ before code; green ⇒ acceptance_met=true. ";
law += "Build env: bash -lc 'module load buildenv-gcc/2026.03-mpich; export OPENBLAS_ROOT=\"$OPENBLASDIR\"; ";
law += "export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu; export LD_LIBRARY_PATH=\"$LIBRARY_PATH:$LD_LIBRARY_PATH\"; ...'. ";
law += "Configure once per tree: cmake -B build " + cmake_flags + ". ";
law += "SCOPED BUILD (binding): never bare `cmake --build build` — that compiles every test binary. ";
law += "Build only what this slug develops: cmake --build build -j$(nproc) --target atomiverse <touched test_*> reg_<slug>. ";
law += "SCOPED TEST (binding): ctest --test-dir build -R '^(<touched test_*>)$' + regression via --target run_reg_<slug>. ";
law += "Loop-domain suite ctest -R '" + test_filter + "' runs only at the merge gate on `" + target_branch + "`; full unfiltered ctest never. ";
law += "Parent `" + target_branch + "`; workers only on wf/<slug>; no push; no reset --hard; worktrees on nobackup. ";
law += "GapCensus every wave: diff " + gap_sources + " + tree vs `" + target_branch + "`; foothold ≠ capability closed; refill " + queue_path + ". ";
law += "DISK: after each build+test wipe build build-* CMakeCache.txt CMakeFiles Testing .cache + temps; never delete source/regressions/specs/.git. ";
law += "CLEARUP (same wave as Worktrees): parallel return ⇒ workers dead; no nested long-lived subagents; ";
law += "remove isolation worktrees; git branch -d fully-merged wf/* only; wipe leftover builds. Design: docs/developer/oracle-capability-loop.md + " + loop_doc + ". ";
if extra_law != "" { law += extra_law; }

log(loop_id + " staff=" + staff_count.to_string() + " target=" + target_branch
    + " max_waves=" + max_waves.to_string());

// ── Bootstrap ───────────────────────────────────────────────────────────
phase("Bootstrap");
let boot = agent(
    law + "BOOTSTRAP parent. Prefer `" + target_branch + "`. Confirm "
        + queue_path + ", " + loop_doc + ", " + handoff_path + ", " + oracle_readme + ". No push.",
    #{ label: "bootstrap", capability_mode: "execute" },
);
if boot != () && boot.success { log("bootstrap ok"); }

// ── Recover ─────────────────────────────────────────────────────────────
let resume_seed = [];
if skip_recover == false {
    phase("Recover");
    let rec = agent(
        law + "RECOVER incomplete on `" + target_branch + "`: queue [~], open specs, unmerged wf/<slug>, half regressions. "
            + "Normalize to ## in_progress. Pick ≤" + staff_count.to_string() + ". Do not invent methods. Output recover schema.",
        #{ label: "recover", capability_mode: "all", output_schema: recover_schema },
    );
    if rec != () && rec.success && rec.output != () {
        log("recover: " + rec.output.summary);
        if rec.output.items != () { resume_seed = rec.output.items; }
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

    // GapCensus
    phase("GapCensus");
    let census_p = law;
    census_p += "GAP-CENSUS wave " + wn + " on `" + target_branch + "`. ";
    census_p += "Diff " + oracle_name + " + " + map_path + " + " + gap_sources + " vs ATV on branch. ";
    census_p += "Rewrite " + queue_path + " ## pending (keep done/blocked/in_progress). Update roadmap ○/✓. ";
    census_p += "scope_out never pending. Optional commit chore(queue). No push. ";
    census_p += "JSON: status=ok|empty, pending_count, gaps_remaining, new_slugs_added, roadmap_updated, summary. empty only if gaps_remaining=0.";
    let census = agent(census_p, #{
        label: "gap-census-w" + wn,
        capability_mode: "all",
        output_schema: census_schema,
    });
    if census != () && census.success && census.output != () {
        log("gap-census: " + census.output.summary);
        if census.output.gaps_remaining != () { last_gaps_remaining = census.output.gaps_remaining; }
    } else {
        log("gap-census: failed");
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
        let asg_p = law + "ASSIGN wave " + wn + ". Read " + queue_path + ". Priority: [~] then open specs then pending. Up to "
            + staff_count.to_string() + ". Move picks to [~]. Output batch schema.";
        if theme != "" { asg_p += " Theme for NEW only: " + theme + "."; }
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
            if consecutive_no_work >= stall_no_work { stop_reason = "gap_census_stalled"; break; }
        }
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
        wp += "WORKER wave " + wn + " isolation. slug=" + slug + " method=" + it.method + ". Branch wf/" + slug + ". ";
        wp += "Spec → offline " + oracle_name + " pin → " + regression_hint + " REFERENCE.md → impl → dual gate → commit. ";
        wp += "Oracle: LAMMPS-first else MetalWalls; set oracle field in JSON; md_only_build=true required for green. ";
        wp += "SCOPED build+test only: --target atomiverse <your test_*> reg_" + slug + "; ctest -R your tests + run_reg_" + slug + ". ";
        if skip_implement { wp += "skip_implement: verify only. "; }
        if do_perf {
            wp += "Perf: T_atv/T_oracle ≤ " + max_ratio.to_string() + " or perf_exempt. ";
            if require_perf { wp += "green needs within_10x or perf_exempt. "; }
        }
        wp += "Exit clean: wipe build*; no nested subagents. Output worker schema.";
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
            if r.output.status == "green" {
                let ok = r.output.acceptance_met == true;
                if ok && require_perf {
                    if r.output.within_10x != true && r.output.perf_exempt != true { ok = false; }
                }
                // MetalWalls: green requires md_only_build=true
                if ok && r.output.md_only_build != true { ok = false; }
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

    // Merge (optional)
    let merged = [];
    if greens.len() > 0 {
        phase("Merge");
        let mp = law;
        mp += "MERGE wave " + wn + " on parent. Candidates:\n" + json_encode(greens) + "\n";
        mp += "merge --no-ff into `" + target_branch + "`. No push. Leave worktree remove to Clearup. ";
        mp += "Conflict hotspots: cpu/dynamics, potentials, longrange, tools/metalwalls_oracle, " + specs_root + ", queue/roadmap. ";
        mp += "Merge gate: scoped rebuild (atomiverse + merged slugs' test_*/reg_*) + ctest -R '" + test_filter + "'. ";
        mp += "If parent rebuilt: wipe build*. Output merge schema.";
        let mres = agent(mp, #{ label: "merge-w" + wn, capability_mode: "all", output_schema: merge_schema });
        if mres != () && mres.success && mres.output != () && mres.output.merged != () {
            merged = mres.output.merged;
            log("merge: " + mres.output.summary);
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

    // Clearup (always)
    phase("Clearup");
    let cp = law;
    cp += "CLEARUP wave " + wn + ". Staffed: " + json_encode(wave_slugs) + ". Merged: " + json_encode(merged) + ". ";
    cp += "1) worktree list → wipe build* under this-wave trees → git worktree remove. ";
    cp += "2) git branch -d wf/<slug> only if fully merged into `" + target_branch + "` (never -D incomplete). ";
    cp += "3) parent wipe leftover build*. No new feature work. Output clearup schema.";
    let cres = agent(cp, #{ label: "clearup-w" + wn, capability_mode: "all", output_schema: clearup_schema });
    if cres != () && cres.success && cres.output != () {
        log("clearup: " + cres.output.summary);
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

    // Harness
    phase("Harness");
    let hp = law;
    hp += "HARNESS wave " + wn + " after " + json_encode(merged) + ". ";
    hp += "Mark done [x]; keep other [~]; close specs; update " + handoff_path + " + " + loop_doc + " run log. ";
    hp += "Roadmap ○→✓ only if complete on branch. Never claim queue exhausted (GapCensus owns discovery). ";
    hp += "Clearup already ran. Commit chore(harness). No push. Output harness schema.";
    let hres = agent(hp, #{ label: "harness-w" + wn, capability_mode: "all", output_schema: harness_schema });
    if hres != () && hres.success && hres.output != () { log("harness: " + hres.output.summary); }

    // Tutorial (optional)
    if write_tutorials {
        phase("Tutorial");
        let tjobs = [];
        let ti = 0;
        while ti < merged.len() {
            let slug = merged[ti];
            tjobs.push(#{
                prompt: law + "TUTORIAL " + tutorial_root + slug + ".md technical report; hard-coded pins; oracle="
                    + oracle_name + ". Single-shot. Output tutorial schema.",
                label: "tut-" + slug,
                capability_mode: "all",
                output_schema: tutorial_schema,
            });
            ti += 1;
        }
        parallel(tjobs);
        agent(
            law + "INDEX methods README for " + json_encode(merged) + ". Single-shot. Output index schema.",
            #{ label: "index-w" + wn, capability_mode: "all", output_schema: index_schema },
        );
    }
}

let b_end = budget();
let report = loop_id + " finished: " + stop_reason
    + " waves=" + wave.to_string()
    + " merged_total=" + merged_total.to_string()
    + " last_gaps_remaining=" + last_gaps_remaining.to_string()
    + " payroll=" + b_end.remaining.to_string() + "/" + b_end.total.to_string();
log(report);
let report_path = write_scratch_file(loop_id + "-loop-report.md", report);
complete(#{
    status: stop_reason,
    waves: wave,
    merged_total: merged_total,
    last_gaps_remaining: last_gaps_remaining,
    staff_count: staff_count,
    target_branch: target_branch,
    report: report_path,
    queue: queue_path,
    pattern: "oracle-capability-loop",
});
