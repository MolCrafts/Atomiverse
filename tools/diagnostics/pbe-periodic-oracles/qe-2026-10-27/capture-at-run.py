#!/usr/bin/env python3
"""Run independent QE; retain original inputs/output, parse printed E/F/stress.

No Atomiverse calculation participates in producing a reference value.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
FUNCTIONALS = (("pbesol", "PBEsol"), ("revpbe", "revPBE"), ("rpbe", "RPBE"))
BOUNDS = {"energy_ha": 1e-5, "each_force_ha_bohr": 1e-4, "each_stress_ev_ang3": 2e-3}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text, nat):
    def number(pattern):
        matches = re.findall(pattern, text, re.MULTILINE)
        return float(matches[-1].replace("D", "E")) if matches else None
    engines = re.findall(r"Program PWSCF[^\n]+", text)
    energies = re.findall(r"^!\s+total energy\s*=\s*([-+\d.EeDd]+)", text, re.MULTILINE)
    blocks = re.findall(r"Forces acting on atoms[^\n]*\n(.*?)(?:\n\s*The |Total force|Computing stress)", text, re.DOTALL)
    forces = []
    if blocks:
        forces = [[float(v.replace("D", "E")) for v in row] for row in re.findall(
            r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)", blocks[-1])]
    stresses = re.findall(r"total\s+stress\s+\(Ry/bohr\*\*3\).*\n([^\n]+)\n([^\n]+)\n([^\n]+)", text)
    tensor = [[float(v) for v in row.split()[:3]] for row in stresses[-1]] if stresses else None
    xc = re.findall(r"Exchange-correlation\s*=\s*([^\n]+)", text)
    result = {"engine": engines[-1].strip() if engines else None,
              "converged": "convergence has been achieved" in text and "JOB DONE." in text and "convergence NOT achieved" not in text,
              "energy_ry": float(energies[-1]) if energies else None,
              "forces_ry_bohr": forces, "stress_tensor_ry_bohr3": tensor,
              "printed_xc": xc[-1].strip() if xc else None,
              "entropy_energy_ry": number(r"smearing contrib\.\s*\(-TS\)\s*=\s*([-+\d.EeDd]+)"),
              "magnetization_e": number(r"total magnetization\s*=\s*([-+\d.EeDd]+)"),
              "printed_process_wall": next((line.strip() for line in reversed(text.splitlines()) if re.match(r"\s*PWSCF\s*:", line)), None)}
    if energies:
        result["energy_ha"] = result["energy_ry"] / 2
    result["forces_ha_bohr"] = [[v/2 for v in row] for row in forces]
    if tensor:
        result["stress_voigt_ha_bohr3"] = [tensor[i][j]/2 for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]
    numbers = ([result["energy_ry"]] if result["energy_ry"] is not None else []) + [v for row in forces for v in row] + ([v for row in tensor for v in row] if tensor else [])
    result["complete_printed_quantities"] = len(forces)==nat and tensor is not None and len(numbers)==1+3*nat+9 and all(math.isfinite(v) for v in numbers)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--allocation", action="store_true",
                   help="Run each independent MPI executable in a fresh step of the existing CPU allocation")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    # An MPI executable inherits the allocation's PMI bootstrap descriptor.
    # Python normally closes it before exec; retain only the declared handles.
    mpi_fds = []
    for key in ("PMI_FD", "PMI2_FD"):
        value = os.environ.get(key, "")
        if value.isdigit():
            descriptor = int(value)
            os.fstat(descriptor)
            mpi_fds.append(descriptor)
    published = Path("/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.2-gmpflf-2026.03/share/cp2k/data/GTH_POTENTIALS")
    report = {"state":"capturing", "scope":"Actual independent QE outputs for PBEsol/revPBE/RPBE; no native pass implied",
              "captured_utc":datetime.now(timezone.utc).isoformat(), "binary":str(binary),"binary_sha256":sha(binary),
              "capture_script_sha256":sha(__file__),"predeclared_bounds":BOUNDS,
              "published_potential_source":str(published), "published_potential_source_sha256":sha(published) if published.exists() else None,
              "xc_definition_source":"https://github.com/QEF/q-e/blob/qe-7.6/Modules/funct.f90",
              "stress_convention":"QE compressive-positive Cartesian tensor; Voigt xx,yy,zz,xy,xz,yz",
              "environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","SLURM_JOB_ID","SLURM_STEP_ID","SLURM_JOB_NODELIST"]},
              "cases":[]}
    si_template=ROOT/"tools/diagnostics/nc-extension/si2.in"
    h_template=ROOT/"tools/qe_oracle/inputs/pw-spin-smear-stress/h3.in"
    for system,nat,template,potential in [
        ("si2-rks",2,si_template,ROOT/"tools/diagnostics/nc-extension/Si_GTH_PBE_q4.gth"),
        ("h3-uks",3,h_template,ROOT/"tools/qe_oracle/inputs/pw-spin-smear-stress/H_GTH_PBE_q1.gth")]:
        for short,xc in FUNCTIONALS:
            identity=system+"-"+short; work=args.out/identity; work.mkdir()
            text=template.read_text().replace("input_dft='PBE'",f"input_dft='{short}'")
            if nat==2:text=text.replace("tstress=.false.","tstress=.true.")
            else:text=text.replace("degauss=0.12","degauss=0.02")
            (work/"input.in").write_text(text);shutil.copyfile(potential,work/potential.name)
            command=[str(binary),"-in","input.in"]
            if args.allocation:
                # A PMI connection belongs to one MPI executable lifetime.
                # Reuse the allocation, with a separate bootstrap per case.
                script = """module load QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$1"
shift
python3 -c 'import json,os,socket;print(json.dumps({"hostname":socket.gethostname(),"environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","SLURM_JOB_ID","SLURM_STEP_ID","SLURM_JOB_NODELIST"]}},indent=2))' > execution.json
exec "$@"
"""
                command=[str(ROOT/"tools/diagnostics/gpw-driver/run-allocation.sh"),"cpu","--cpus","4","--",
                         "bash","-lc",script,"qe-oracle",str(work.resolve()),str(binary),"-in","input.in"]
            item={"id":identity,"system":system,"xc":xc,"input_dft":short,"nat":nat,
                  "command":command,"input":str((work/"input.in").relative_to(args.out)),
                  "input_sha256":sha(work/"input.in"),"original_template":str(template.relative_to(ROOT)),"original_template_sha256":sha(template),
                  "potential":potential.name,"potential_sha256":sha(potential),"output":identity+"/qe.out"}
            start=time.monotonic()
            try:
                with (work/"qe.out").open("w") as stream:
                    result=subprocess.run(command,cwd=work,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout,pass_fds=tuple(mpi_fds))
                item.update(parse((work/"qe.out").read_text(),nat));item["exit_code"]=result.returncode
                item["output_sha256"]=sha(work/"qe.out")
                if (work/"execution.json").exists():
                    item["execution"]=json.loads((work/"execution.json").read_text())
                selected=(item["printed_xc"] or "").lower()
                item["accepted_external_output"]=result.returncode==0 and item["converged"] and item["complete_printed_quantities"] and short in selected
            except (OSError, ValueError, subprocess.SubprocessError) as e:
                item.update(accepted_external_output=False,error=str(e))
            item["elapsed_wall_seconds"]=time.monotonic()-start
            report["cases"].append(item)
            (args.out/"reference.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            print(("PASS" if item["accepted_external_output"] else "FAIL")+" QE_CAPTURE "+identity+" E_Ha="+str(item.get("energy_ha")),flush=True)
    report["state"]="captured" if all(c["accepted_external_output"] for c in report["cases"]) else "failed"
    (args.out/"reference.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return 0 if report["state"]=="captured" else 1


if __name__=="__main__":
    raise SystemExit(main())
