<!-- mol:note:topic:oracle-pin-provenance -->
## Oracle pin provenance (binding)

Agents chasing a green invent pins: numpy rewrites of ATV, self-golden
`expected := what_we_got()`, literature windows labelled `oracle: cp2k`,
cross-representation "twins" (GTH-GPW energies standing in for NC-UPF PW).
Those never fail and never measure the implementation. (Uninstalled-engine
pins have disagreed with the real engine by 6–16 % once it was run.)

### All oracle loops (pyscf / cp2k / metalwalls / tb)

1. **A pin is a number the other code printed.** Provenance = binary path +
   version/sha + input file + output line. Re-runnable.
2. **`tools/*_oracle/` only parses oracle output** — never reimplements the ATV
   algorithm, never seeds from ATV's own measurement.
3. **`oracle:` names the code that produced the number.** A formula from a
   paper is `oracle: literature`, and only if it is from the paper, not our
   header.
4. **No external number ⇒** `oracle: literature` with a real citation, or the
   slug is **not landable**. Never invent `oracle: <engine>`.
5. **Self-golden is forbidden.** If our implementation were wrong, would this
   number change? If not, it is not a pin.

### CP2K method loop — strict twin

| Gate | Requirement |
| --- | --- |
| Install | Use a real CP2K binary on the pin host. "Module empty" / "not on PATH" is an action item (`tools/cp2k_oracle/` + source build), not a skip. |
| Algorithm match | CP2K's algorithm for the method class ATV claims — same representation family (GPW↔GPW GTH/MOLOPT …), same spin/k/XC class, comparable cutoff/basis/PP — with ATV exercised on that matched path. |
| Correctness | Numbers CP2K printed; the regression asserts ATV vs them within a stated tol. |
| Perf | Fair wall on the same twin job: \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) in `REFERENCE.md` (plus `within_10x` / ratio gate when the loop requires it). |
| Literature escape | Only when the quantity is not a CP2K algorithm, or a published number is the scientific standard. Never a substitute for "CP2K not installed". |

**Forbidden as CP2K pins:**

- GTH-GPW (any Quickstep GTH) numbers anchoring NC-UPF / pure PW ATV, or the
  reverse — unequal Hamiltonians are not twins.
- Cross-PP, cross-basis-class or cross-method "physics windows" labelled
  `oracle: cp2k`.
- `perf_exempt` because the binary is missing or the twin was not run.
- `.inp` sketches never executed; "offline pins" without output lines.
- ATV self-FD of the total energy as the sole correctness gate under
  `oracle: cp2k` (FD may be an extra consistency check).

If ATV's production path is representation-unequal to CP2K: (a) land a matched
ATV path CP2K can twin, (b) pin to literature, or (c) leave the row blocked —
never a mismatched window.

ctest never launches CP2K; pins are frozen offline, but they must come from a
real CP2K run (or real literature).

### Related

- `tools/cp2k_oracle/README.md`, `tools/metalwalls_oracle/`
- `docs/developer/oracle-capability-loop.md` — where it disagrees (optional
  speed, mismatch windows), this note and docs/developer/project-contract.md win.
