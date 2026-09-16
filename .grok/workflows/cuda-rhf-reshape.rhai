// cuda-rhf-reshape — Route B: mature RHF line, multi-wave unattended loop
//
// Each wave: GapCensus → Plan → Implement → Verify → (advance milestone | stop)
// Discovers remaining debt/gaps, implements, re-checks until acceptance green
// or max_waves / agent budget exhausted.
//
// Unattended overnight: unattended=true (default) — NO await_user gates.
// Oracle = offline PySCF pins; never ATV self-golden.
// CUDA must use GPU paradigms — not host loops in .cu wrappers.

let meta = #{
    name: "cuda-rhf-reshape",
    description: "Unattended multi-wave: CPU/oracle audit + CUDA RHF reshape + parity/perf until green",
    when_to_use: "Overnight reshape of CUDA SCF to concrete RHF while clearing debt; loop until shape/force/aimd green",
    phases: [
        #{ title: "Bootstrap", detail: "lock line + first inventory" },
        #{ title: "Wave", detail: "census → plan → impl → verify (loop)" },
        #{ title: "Report", detail: "final scratch report" },
    ],
};

// ── Defaults ────────────────────────────────────────────────────────────
let method = "RHF";
let system_slug = "h2o_rhf_sto3g";
let basis = "STO3G";
let milestone = "shape";
// shape | force | aimd — starting milestone; can auto-advance
let skip_implement = false;
let do_perf = true;
let require_perf = false;
let max_ratio = 10.0;
let target_branch = "dev";
let isolation = true;
let dry_run = false;
let unattended = true;
// no human gates when true
let max_waves = 12;
let mode = "exhaust";
// exhaust | burst (burst caps waves lower unless set)
let auto_advance = true;
// after shape green → force → aimd until done or budget
let min_budget = 12;
// stop starting a new wave if remaining agents below this
let stall_fail_waves = 3;
// consecutive waves with no acceptance progress → stop

if args != () {
    if args.method != () { method = args.method; }
    if args.system_slug != () { system_slug = args.system_slug; }
    if args.system != () { system_slug = args.system; }
    if args.basis != () { basis = args.basis; }
    if args.milestone != () { milestone = args.milestone; }
    if args.skip_implement != () { skip_implement = args.skip_implement; }
    if args.do_perf != () { do_perf = args.do_perf; }
    if args.require_perf != () { require_perf = args.require_perf; }
    if args.max_ratio != () { max_ratio = args.max_ratio; }
    if args.target_branch != () { target_branch = args.target_branch; }
    if args.isolation != () { isolation = args.isolation; }
    if args.dry_run != () { dry_run = args.dry_run; }
    if args.unattended != () { unattended = args.unattended; }
    if args.max_waves != () { max_waves = args.max_waves; }
    if args.mode != () { mode = args.mode; }
    if args.auto_advance != () { auto_advance = args.auto_advance; }
    if args.min_budget != () { min_budget = args.min_budget; }
    if args.stall_fail_waves != () { stall_fail_waves = args.stall_fail_waves; }
}
if dry_run { skip_implement = true; }
if do_perf == false { require_perf = false; }
if milestone != "shape" && milestone != "force" && milestone != "aimd" {
    milestone = "shape";
}
if mode != "exhaust" && mode != "burst" { mode = "exhaust"; }
if mode == "burst" && (args == () || args.max_waves == ()) { max_waves = 3; }
if max_waves < 1 { max_waves = 1; }
if max_waves > 30 { max_waves = 30; }
if min_budget < 8 { min_budget = 8; }
if stall_fail_waves < 2 { stall_fail_waves = 2; }

let cpu_rhf_h = "include/atomiverse/cpu/scf/rhf.h";
let cuda_scf_root = "include/atomiverse/cuda/scf/";
let reg_path = "regressions/scf/" + system_slug + "/";
let oracle_readme = "tools/pyscf_oracle/README.md";
let pin_tool = "tools/pyscf_oracle/pin_scf.py";
let fair_bench = "tools/pyscf_oracle/fair_bench.py";
let arch_note = ".claude/notes/architecture.md";
let claude_md = "CLAUDE.md";
let report_name = "cuda-rhf-reshape-report.md";

let cmake_cpu = "-DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS";
let cmake_cuda = "-DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS";

// ── Schemas ─────────────────────────────────────────────────────────────
let inventory_schema = #{
    "type": "object",
    "required": ["cpu_public_api", "cuda_public_api", "debt_items", "mature_line", "oracle_baseline", "gaps_remaining", "acceptance_met"],
    "properties": #{
        "cpu_public_api": #{ "type": "string" },
        "cuda_public_api": #{ "type": "string" },
        "debt_items": #{
            "type": "array", "maxItems": 24,
            "items": #{
                "type": "object",
                "required": ["id", "side", "severity", "summary"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "side": #{ "type": "string" },
                    "severity": #{ "type": "string" },
                    "summary": #{ "type": "string" },
                    "blocks_isomorphism": #{ "type": "boolean" },
                    "open": #{ "type": "boolean" },
                },
            },
        },
        "mature_line": #{ "type": "string" },
        "oracle_baseline": #{ "type": "string" },
        "existing_cuda_kernels": #{ "type": "string" },
        "gaps_remaining": #{ "type": "integer" },
        "acceptance_met": #{ "type": "boolean" },
        "suggested_next_tasks": #{ "type": "array", "items": #{ "type": "string" } },
        "notes": #{ "type": "string" },
    },
};

let arch_audit_schema = #{
    "type": "object",
    "required": ["verdict", "findings", "target_shape"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "findings": #{
            "type": "array", "maxItems": 16,
            "items": #{
                "type": "object",
                "required": ["path", "issue", "severity"],
                "properties": #{
                    "path": #{ "type": "string" },
                    "issue": #{ "type": "string" },
                    "severity": #{ "type": "string" },
                    "iron_law": #{ "type": "string" },
                },
            },
        },
        "target_shape": #{ "type": "string" },
        "blocking_for_cuda_port": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

let oracle_audit_schema = #{
    "type": "object",
    "required": ["status", "pin_source", "observables"],
    "properties": #{
        "status": #{ "type": "string" },
        "pin_source": #{ "type": "string" },
        "pyscf_version": #{ "type": "string" },
        "observables": #{
            "type": "array", "maxItems": 12,
            "items": #{
                "type": "object",
                "required": ["name", "oracle_value", "atv_value", "tolerance", "pass"],
                "properties": #{
                    "name": #{ "type": "string" },
                    "oracle_value": #{ "type": "string" },
                    "atv_value": #{ "type": "string" },
                    "tolerance": #{ "type": "string" },
                    "pass": #{ "type": "boolean" },
                    "unit": #{ "type": "string" },
                },
            },
        },
        "provenance_ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "rerun_commands": #{ "type": "string" },
    },
};

let perf_audit_schema = #{
    "type": "object",
    "required": ["status", "summary"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "hotspots": #{
            "type": "array", "maxItems": 12,
            "items": #{
                "type": "object",
                "required": ["name", "why_hot", "cuda_fit"],
                "properties": #{
                    "name": #{ "type": "string" },
                    "why_hot": #{ "type": "string" },
                    "cuda_fit": #{ "type": "string" },
                    "evidence": #{ "type": "string" },
                },
            },
        },
        "cpu_wall_s": #{ "type": "string" },
        "bench_commands": #{ "type": "string" },
        "caveats": #{ "type": "string" },
    },
};

let plan_schema = #{
    "type": "object",
    "required": ["milestone", "spec_slug", "tasks", "non_goals", "acceptance", "cuda_paradigm_rules", "wave_goal"],
    "properties": #{
        "milestone": #{ "type": "string" },
        "spec_slug": #{ "type": "string" },
        "wave_goal": #{ "type": "string" },
        "tasks": #{
            "type": "array", "maxItems": 12,
            "items": #{
                "type": "object",
                "required": ["id", "side", "title", "detail"],
                "properties": #{
                    "id": #{ "type": "string" },
                    "side": #{ "type": "string" },
                    "title": #{ "type": "string" },
                    "detail": #{ "type": "string" },
                },
            },
        },
        "non_goals": #{ "type": "array", "items": #{ "type": "string" } },
        "acceptance": #{ "type": "array", "items": #{ "type": "string" } },
        "cuda_paradigm_rules": #{ "type": "array", "items": #{ "type": "string" } },
        "files_touch": #{ "type": "array", "items": #{ "type": "string" } },
        "risk": #{ "type": "string" },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["status", "summary", "changed_paths", "debt_closed", "debt_remaining"],
    "properties": #{
        "status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "changed_paths": #{ "type": "array", "items": #{ "type": "string" } },
        "debt_closed": #{ "type": "array", "items": #{ "type": "string" } },
        "debt_remaining": #{ "type": "array", "items": #{ "type": "string" } },
        "build_commands": #{ "type": "string" },
        "tests_run": #{ "type": "string" },
        "branch_or_worktree": #{ "type": "string" },
        "blockers": #{ "type": "string" },
        "progress_made": #{ "type": "boolean" },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["parity_status", "perf_status", "summary", "pass_acceptance"],
    "properties": #{
        "parity_status": #{ "type": "string" },
        "perf_status": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "energy_cpu": #{ "type": "string" },
        "energy_cuda": #{ "type": "string" },
        "energy_oracle": #{ "type": "string" },
        "force_max_abs_err": #{ "type": "string" },
        "wall_cpu_s": #{ "type": "string" },
        "wall_cuda_s": #{ "type": "string" },
        "speedup": #{ "type": "string" },
        "speedup_notes": #{ "type": "string" },
        "commands": #{ "type": "string" },
        "pass_acceptance": #{ "type": "boolean" },
        "milestone_complete": #{ "type": "boolean" },
        "open_gaps": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

// ── Law ─────────────────────────────────────────────────────────────────
let law = "";
law += "IRON LAWS: Frame-only public I/O; no user Context; no public *Config; no make_*/build_*; ";
law += "concrete RHF not SCFSolver bag; public units eV/Å/fs. ";
law += "Oracle: " + pin_tool + " / PySCF offline pins — NEVER seed from ATV. ";
law += "Scope: Gaussian RHF dual-backend reshape only — NOT classical MD, GPW/PW, post-HF CUDA, electrodes. ";
law += "CPU cmake: " + cmake_cpu + ". CUDA: " + cmake_cuda + " only on GPU (GH200) node. ";
law += "Env: bash -lc + module load per AGENTS.md; RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu on x86; ";
law += "LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH. Scoped build targets only. ";
law += "Unattended: do not wait for human; make progress every wave; leave buildable state. ";

let cuda_paradigm = "";
cuda_paradigm += "CUDA PARADIGM: (1) device-resident P/F/C across SCF iters; ";
cuda_paradigm += "(2) LaunchContext streams; (3) coalesced ERI/Fock, smem tiles; ";
cuda_paradigm += "(4) cublas/cusolver for dense linalg; (5) host integral bag only as named one-shot bridge; ";
cuda_paradigm += "(6) public cuda::scf::RHF + Evaluate — demote SCFSolver to internal; ";
cuda_paradigm += "(7) shape=energy; force/aimd only on those milestones. ";

// Milestone acceptance hints for census
fn milestone_hint(ms) {
    if ms == "force" {
        return "FORCE: analytic forces on Frame; vs PySCF/FD; public RHF has Force capability.";
    }
    if ms == "aimd" {
        return "AIMD: cuda::md::MD + RHF evaluator short run; no AimdDriver; energy finite.";
    }
    return "SHAPE: public cuda::scf::RHF members+run(frame)/Evaluate; no public integral bag ctor; SCFSolver internal; kEnergy eV; H2O STO-3G E vs pin; CPU reg still green.";
}

log("cuda-rhf-reshape UNATTENDED loop: method=" + method + " system=" + system_slug
    + " start_milestone=" + milestone + " max_waves=" + max_waves.to_string()
    + " auto_advance=" + auto_advance.to_string() + " dry_run=" + dry_run.to_string());

// ════════════════════════════════════════════════════════════════════════
phase("Bootstrap");
// ════════════════════════════════════════════════════════════════════════

let wave = 0;
let consecutive_no_progress = 0;
let last_inv = ();
let last_arch = ();
let last_oracle = ();
let last_perf = ();
let last_plan = ();
let last_impl = ();
let last_verify = ();
let wave_log = "";
let milestones_done = [];
let overall = "running";
let stop_reason = "";

// Initial deep audit once (wave 0 prep) — parallel arch/oracle/perf
let boot_prompt = "";
boot_prompt += "Bootstrap inventory for cuda-rhf-reshape. READ-ONLY. Read real headers.\n";
boot_prompt += "Milestone focus: " + milestone + " — " + milestone_hint(milestone) + "\n";
boot_prompt += "Map CPU " + cpu_rhf_h + " vs " + cuda_scf_root + "; reg " + reg_path + "; debt list.\n";
boot_prompt += "Set gaps_remaining = count of open blocker/high debts for THIS milestone.\n";
boot_prompt += "acceptance_met=true only if public isomorphic RHF energy path already exists and e2e would pass.\n";
boot_prompt += "suggested_next_tasks: short list for implementers.\n";
boot_prompt += "Return full inventory schema. " + law;

let inv0 = agent(boot_prompt, #{
    label: "bootstrap-inventory",
    capability_mode: "read-only",
    agent_type: "explore",
    output_schema: inventory_schema,
});
if inv0 == () || inv0.success == false {
    if unattended {
        complete(#{ summary: "bootstrap failed", overall: "blocked", stop_reason: "bootstrap_inventory_failed" });
    }
    pause("infra", "Bootstrap inventory failed.");
}
last_inv = inv0.output;

let inv_json = json_encode(last_inv);

let arch_prompt0 = "";
arch_prompt0 += "Architecture audit (read-only) for milestone " + milestone + ".\n";
arch_prompt0 += "Inventory:\n" + inv_json + "\n";
arch_prompt0 += "CLAUDE.md + " + arch_note + ". findings with path:issue. " + law;
let oracle_prompt0 = "";
oracle_prompt0 += "PySCF oracle audit for " + system_slug + ". Execute if build exists.\n";
oracle_prompt0 += "Read " + reg_path + " REFERENCE + reg_*.cpp; " + oracle_readme + " if any.\n";
oracle_prompt0 += "Provenance must not be ATV self-golden. " + law;
let perf_prompt0 = "";
perf_prompt0 += "CPU RHF perf baseline (read-only). Hotspots + cuda_fit. No fake CUDA speedups. " + law;

let audit0 = parallel([
    #{ prompt: arch_prompt0, label: "arch-w0", capability_mode: "read-only", agent_type: "mol:architect", output_schema: arch_audit_schema },
    #{ prompt: oracle_prompt0, label: "oracle-w0", capability_mode: "execute", agent_type: "general-purpose", output_schema: oracle_audit_schema },
    #{ prompt: perf_prompt0, label: "perf-w0", capability_mode: "read-only", agent_type: "av-hpc", output_schema: perf_audit_schema },
]);
if audit0.len() > 0 && audit0[0] != () && audit0[0].success { last_arch = audit0[0].output; }
if audit0.len() > 1 && audit0[1] != () && audit0[1].success { last_oracle = audit0[1].output; }
if audit0.len() > 2 && audit0[2] != () && audit0[2].success { last_perf = audit0[2].output; }

// ════════════════════════════════════════════════════════════════════════
phase("Wave");
// ════════════════════════════════════════════════════════════════════════

while wave < max_waves {
    wave += 1;
    let b = budget();
    log("=== WAVE " + wave.to_string() + "/" + max_waves.to_string()
        + " milestone=" + milestone
        + " budget_remaining=" + b.remaining.to_string() + " ===");

    if b.remaining < min_budget {
        stop_reason = "budget_exhausted";
        log("Stop: remaining budget " + b.remaining.to_string() + " < min_budget " + min_budget.to_string());
        break;
    }

    // ── GapCensus (every wave) ──────────────────────────────────────────
    let census_prompt = "";
    census_prompt += "GapCensus wave " + wave.to_string() + " for cuda-rhf-reshape. READ actual tree (grep/read_file).\n";
    census_prompt += "Current milestone: " + milestone + "\n" + milestone_hint(milestone) + "\n";
    census_prompt += "Prior inventory (may be stale):\n" + json_encode(last_inv) + "\n";
    if last_verify != () {
        census_prompt += "Prior verify:\n" + json_encode(last_verify) + "\n";
    }
    if last_impl != () {
        census_prompt += "Prior impl:\n" + json_encode(last_impl) + "\n";
    }
    census_prompt += "Re-diff CPU vs CUDA for this milestone. List OPEN debts only as open:true.\n";
    census_prompt += "gaps_remaining = number of open items blocking acceptance_met.\n";
    census_prompt += "acceptance_met=true ONLY if milestone acceptance is already satisfied in tree.\n";
    census_prompt += "suggested_next_tasks: 1-5 concrete implementer tasks for THIS wave (smallest increment).\n";
    census_prompt += "Foothold ≠ closed: partial RHF shell without energy e2e is still open.\n" + law;

    let census_r = agent(census_prompt, #{
        label: "census-w" + wave.to_string(),
        capability_mode: "read-only",
        agent_type: "explore",
        output_schema: inventory_schema,
    });
    if census_r != () && census_r.success && census_r.output != () {
        last_inv = census_r.output;
    }

    if last_inv != () && last_inv.acceptance_met == true {
        log("Census: acceptance_met already true for " + milestone);
        milestones_done.push(milestone);
        if auto_advance {
            if milestone == "shape" {
                milestone = "force";
                log("Auto-advance → force");
                consecutive_no_progress = 0;
                wave_log += "w" + wave.to_string() + ":shape_already_green→force\n";
                continue;
            }
            if milestone == "force" {
                milestone = "aimd";
                log("Auto-advance → aimd");
                consecutive_no_progress = 0;
                wave_log += "w" + wave.to_string() + ":force_already_green→aimd\n";
                continue;
            }
            // aimd green
            overall = "pass";
            stop_reason = "all_milestones_green";
            break;
        } else {
            overall = "pass";
            stop_reason = "milestone_green";
            break;
        }
    }

    if last_inv != () && last_inv.gaps_remaining != () && last_inv.gaps_remaining == 0 {
        // no gaps but acceptance not met — still plan/verify
        log("Census gaps_remaining=0 but acceptance_met false — still planning");
    }

    // ── Plan (this wave only — small batch) ─────────────────────────────
    let plan_prompt = "";
    plan_prompt += "Plan WAVE " + wave.to_string() + " only (not the whole multi-month roadmap).\n";
    plan_prompt += "Milestone: " + milestone + "\n" + milestone_hint(milestone) + "\n";
    plan_prompt += "Census:\n" + json_encode(last_inv) + "\n";
    if last_arch != () { plan_prompt += "Arch:\n" + json_encode(last_arch) + "\n"; }
    if last_oracle != () { plan_prompt += "Oracle:\n" + json_encode(last_oracle) + "\n"; }
    if last_verify != () { plan_prompt += "Last verify fails:\n" + json_encode(last_verify) + "\n"; }
    plan_prompt += "wave_goal: one sentence what this wave must land.\n";
    plan_prompt += "tasks: MAX 5 high-leverage tasks closable in one implement agent session.\n";
    plan_prompt += "Prefer finishing acceptance over perfection. " + law + " " + cuda_paradigm;

    let plan_r = agent(plan_prompt, #{
        label: "plan-w" + wave.to_string(),
        capability_mode: "read-only",
        agent_type: "plan",
        output_schema: plan_schema,
    });
    if plan_r == () || plan_r.success == false || plan_r.output == () {
        consecutive_no_progress += 1;
        wave_log += "w" + wave.to_string() + ":plan_failed\n";
        log("Plan failed wave " + wave.to_string());
        if consecutive_no_progress >= stall_fail_waves {
            stop_reason = "stall_plan_failures";
            break;
        }
        continue;
    }
    last_plan = plan_r.output;
    let plan_json = json_encode(last_plan);

    // ── Implement ───────────────────────────────────────────────────────
    let impl_out = #{
        status: "skipped",
        summary: "skip_implement/dry_run",
        changed_paths: [],
        debt_closed: [],
        debt_remaining: [],
        build_commands: "",
        tests_run: "",
        branch_or_worktree: "",
        blockers: "",
        progress_made: false,
    };

    if skip_implement == false {
        let impl_prompt = "";
        impl_prompt += "IMPLEMENT wave " + wave.to_string() + " milestone=" + milestone + " UNATTENDED.\n";
        impl_prompt += "You WRITE code. Execute the plan tasks fully if possible.\n";
        impl_prompt += "Plan:\n" + plan_json + "\n";
        impl_prompt += "Census:\n" + json_encode(last_inv) + "\n";
        impl_prompt += "Rules:\n";
        impl_prompt += "- Minimal diff; iron laws; " + cuda_paradigm + "\n";
        impl_prompt += "- Land working intermediate state; compile what you touch.\n";
        impl_prompt += "- CPU: only blocking isomorphism / oracle bugs on this line.\n";
        impl_prompt += "- CUDA: reshape toward cuda::scf::RHF; demote bag; Frame-only user path.\n";
        impl_prompt += "- Tests: unit under tests/; e2e regressions/scf/*_cuda when ready.\n";
        impl_prompt += "- GPU node: use " + cmake_cuda + ". Login x86: still land host API + skip CUDA runtime tests cleanly.\n";
        impl_prompt += "- Do NOT push. Do NOT wait for user. progress_made=true if any acceptance-moving change landed.\n";
        impl_prompt += "- If blocked (no GPU for force kernels etc.), land maximum host-side shape work; status=partial.\n";
        impl_prompt += "Return impl schema. " + law;

        let impl_r = agent(impl_prompt, #{
            label: "impl-w" + wave.to_string(),
            capability_mode: "all",
            agent_type: "general-purpose",
            output_schema: impl_schema,
            isolation_worktree: isolation,
        });
        if impl_r != () && impl_r.success && impl_r.output != () {
            impl_out = impl_r.output;
            // isolation worktree: instruct merge? parent cannot auto-merge isolation.
            // Prefer isolation=false for unattended overnight so edits land on parent workspace.
        } else {
            impl_out = #{
                status: "blocked",
                summary: "implement agent failed",
                changed_paths: [],
                debt_closed: [],
                debt_remaining: [],
                build_commands: "",
                tests_run: "",
                branch_or_worktree: "",
                blockers: "agent failure",
                progress_made: false,
            };
        }
    }
    last_impl = impl_out;

    // ── Verify ──────────────────────────────────────────────────────────
    let verify_prompt = "";
    verify_prompt += "VERIFY wave " + wave.to_string() + " milestone=" + milestone + ".\n";
    verify_prompt += "Fail closed. Read tree + run scoped builds/tests when possible.\n";
    verify_prompt += "Plan acceptance:\n" + plan_json + "\n";
    verify_prompt += "Impl:\n" + json_encode(impl_out) + "\n";
    if last_oracle != () { verify_prompt += "Oracle baseline:\n" + json_encode(last_oracle) + "\n"; }
    verify_prompt += "Set pass_acceptance / milestone_complete from EVIDENCE.\n";
    verify_prompt += "open_gaps: remaining blockers. Measure CUDA speedup only with real walls; else n/a.\n";
    verify_prompt += "do_perf=" + do_perf.to_string() + ". " + law;

    let verify_r = agent(verify_prompt, #{
        label: "verify-w" + wave.to_string(),
        capability_mode: "execute",
        agent_type: "general-purpose",
        output_schema: verify_schema,
    });
    if verify_r != () && verify_r.success && verify_r.output != () {
        last_verify = verify_r.output;
    } else {
        last_verify = #{
            parity_status: "unknown",
            perf_status: "unknown",
            summary: "verify agent failed",
            pass_acceptance: false,
            milestone_complete: false,
            open_gaps: ["verify_agent_failed"],
        };
    }

    let progressed = false;
    if last_impl != () && last_impl.progress_made == true { progressed = true; }
    if last_verify != () && last_verify.pass_acceptance == true { progressed = true; }

    if last_verify != () && (last_verify.pass_acceptance == true || last_verify.milestone_complete == true) {
        wave_log += "w" + wave.to_string() + ":" + milestone + ":PASS\n";
        log("Wave " + wave.to_string() + " PASS milestone " + milestone);
        milestones_done.push(milestone);
        consecutive_no_progress = 0;
        if auto_advance {
            if milestone == "shape" {
                milestone = "force";
                continue;
            }
            if milestone == "force" {
                milestone = "aimd";
                continue;
            }
            overall = "pass";
            stop_reason = "all_milestones_green";
            break;
        }
        overall = "pass";
        stop_reason = "milestone_green";
        break;
    }

    if progressed {
        consecutive_no_progress = 0;
        wave_log += "w" + wave.to_string() + ":" + milestone + ":progress\n";
    } else {
        consecutive_no_progress += 1;
        wave_log += "w" + wave.to_string() + ":" + milestone + ":no_progress\n";
        log("No progress wave " + wave.to_string() + " streak=" + consecutive_no_progress.to_string());
        if consecutive_no_progress >= stall_fail_waves {
            stop_reason = "stall_no_progress";
            overall = "stalled";
            break;
        }
    }

    // optional mid-run light re-oracle if energy claims changed
}

if overall == "running" {
    if stop_reason == "" { stop_reason = "max_waves"; }
    if last_verify != () && last_verify.pass_acceptance == true {
        overall = "pass";
    } else if last_impl != () && (last_impl.status == "ok" || last_impl.status == "partial") {
        overall = "partial";
    } else {
        overall = "incomplete";
    }
}

// ════════════════════════════════════════════════════════════════════════
phase("Report");
// ════════════════════════════════════════════════════════════════════════

let report = "";
report += "# cuda-rhf-reshape overnight report\n\n";
report += "- method: `" + method + "` system: `" + system_slug + "` basis: `" + basis + "`\n";
report += "- final_milestone_cursor: `" + milestone + "`\n";
report += "- waves_run: " + wave.to_string() + " / max " + max_waves.to_string() + "\n";
report += "- overall: **" + overall + "**\n";
report += "- stop_reason: " + stop_reason + "\n";
report += "- unattended: " + unattended.to_string() + " isolation: " + isolation.to_string() + "\n";
report += "- auto_advance: " + auto_advance.to_string() + "\n\n";

report += "## Wave log\n\n```\n" + wave_log + "```\n\n";

report += "## Milestones completed this run\n\n";
if milestones_done.len() == 0 {
    report += "_(none fully closed)_\n";
} else {
    for m in milestones_done {
        report += "- `" + m + "`\n";
    }
}

report += "\n## Last census\n\n";
if last_inv != () {
    report += "- gaps_remaining: " + last_inv.gaps_remaining.to_string() + "\n";
    report += "- acceptance_met: " + last_inv.acceptance_met.to_string() + "\n";
    report += "- CPU API: " + last_inv.cpu_public_api + "\n";
    report += "- CUDA API: " + last_inv.cuda_public_api + "\n";
    report += "- oracle_baseline: " + last_inv.oracle_baseline + "\n";
    if last_inv.debt_items != () {
        report += "\n### Debt\n\n";
        for d in last_inv.debt_items {
            report += "- **" + d.id + "** (" + d.side + "/" + d.severity + "): " + d.summary + "\n";
        }
    }
    if last_inv.suggested_next_tasks != () {
        report += "\n### Suggested next\n\n";
        for t in last_inv.suggested_next_tasks {
            report += "- " + t + "\n";
        }
    }
}

report += "\n## Last architecture audit (wave0)\n\n";
if last_arch != () {
    report += "- verdict: " + last_arch.verdict + "\n";
    report += "- target: " + last_arch.target_shape + "\n";
}

report += "\n## Oracle\n\n";
if last_oracle != () {
    report += "- status: " + last_oracle.status + "\n";
    report += "- pin: " + last_oracle.pin_source + "\n";
    if last_oracle.observables != () {
        report += "\n| name | oracle | atv | tol | pass |\n|---|---|---|---|---|\n";
        for o in last_oracle.observables {
            let p = "no";
            if o.pass { p = "yes"; }
            report += "| " + o.name + " | " + o.oracle_value + " | " + o.atv_value + " | "
                + o.tolerance + " | " + p + " |\n";
        }
    }
}

report += "\n## CPU perf baseline\n\n";
if last_perf != () {
    report += last_perf.summary + "\n";
    if last_perf.hotspots != () {
        for h in last_perf.hotspots {
            report += "- **" + h.name + "**: " + h.why_hot + " → " + h.cuda_fit + "\n";
        }
    }
}

report += "\n## Last plan\n\n";
if last_plan != () {
    report += "- spec: `" + last_plan.spec_slug + "` wave_goal: " + last_plan.wave_goal + "\n";
    if last_plan.tasks != () {
        for t in last_plan.tasks {
            report += "- [" + t.side + "] " + t.title + ": " + t.detail + "\n";
        }
    }
}

report += "\n## Last implement\n\n";
if last_impl != () {
    report += "- status: **" + last_impl.status + "**\n";
    report += "- " + last_impl.summary + "\n";
    if last_impl.changed_paths != () {
        for p in last_impl.changed_paths {
            report += "- `" + p + "`\n";
        }
    }
    if last_impl.blockers != () && last_impl.blockers != "" {
        report += "- blockers: " + last_impl.blockers + "\n";
    }
}

report += "\n## Last verify (parity + speedup)\n\n";
if last_verify != () {
    report += "- parity: **" + last_verify.parity_status + "** perf: **" + last_verify.perf_status + "**\n";
    report += "- " + last_verify.summary + "\n";
    if last_verify.energy_oracle != () { report += "- E_oracle: " + last_verify.energy_oracle + "\n"; }
    if last_verify.energy_cpu != () { report += "- E_cpu: " + last_verify.energy_cpu + "\n"; }
    if last_verify.energy_cuda != () { report += "- E_cuda: " + last_verify.energy_cuda + "\n"; }
    if last_verify.wall_cpu_s != () { report += "- wall_cpu: " + last_verify.wall_cpu_s + "\n"; }
    if last_verify.wall_cuda_s != () { report += "- wall_cuda: " + last_verify.wall_cuda_s + "\n"; }
    if last_verify.speedup != () { report += "- **speedup: " + last_verify.speedup + "**\n"; }
    if last_verify.pass_acceptance != () {
        report += "- pass_acceptance: " + last_verify.pass_acceptance.to_string() + "\n";
    }
    if last_verify.open_gaps != () {
        report += "\nOpen gaps:\n";
        for g in last_verify.open_gaps {
            report += "- " + g + "\n";
        }
    }
    if last_verify.commands != () {
        report += "\n```bash\n" + last_verify.commands + "\n```\n";
    }
}

report += "\n---\n*Unattended multi-wave `cuda-rhf-reshape`. Check git status / diffs in the morning.*\n";

let report_path = write_scratch_file(report_name, report);
log("Final report: " + report_path + " overall=" + overall + " stop=" + stop_reason);

complete(#{
    summary: "cuda-rhf-reshape " + overall + " (" + stop_reason + ") waves=" + wave.to_string(),
    overall: overall,
    stop_reason: stop_reason,
    waves: wave,
    milestone_cursor: milestone,
    milestones_done: milestones_done,
    report_path: report_path,
    method: method,
    system_slug: system_slug,
});
