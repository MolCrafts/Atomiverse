#!/usr/bin/env python3
"""Capture independent CP2K GPW E/F/numerical-stress outputs; no native values."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
DATA=Path('/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.2-gmpflf-2026.03/share/cp2k/data')
MODELS=[('pbesol','PBEsol','GGA_X_PBE_SOL','GGA_C_PBE_SOL'),('revpbe','revPBE','GGA_X_PBE_R','GGA_C_PBE'),('rpbe','RPBE','GGA_X_RPBE','GGA_C_PBE')]
BOUNDS={'energy_ha':5e-6,'each_force_ha_bohr':5e-5,'each_supported_diagonal_stress_ha_bohr3':5e-5}
TEMPLATE='''&GLOBAL
 PROJECT h2_variant
 RUN_TYPE ENERGY_FORCE
 PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
 METHOD Quickstep
 STRESS_TENSOR NUMERICAL
 &DFT
  BASIS_SET_FILE_NAME {basis}
  POTENTIAL_FILE_NAME {potential}
  &MGRID
   CUTOFF 80
   NGRIDS 1
  &END MGRID
  &QS
   METHOD GPW
   CORE_PPL GRID
   EPS_DEFAULT 1.0E-10
   EPS_PGF_ORB 1.0E-10
   EPS_PPNL 1.0E-10
   EPS_FILTER_MATRIX 0.0
  &END QS
  &SCF
   SCF_GUESS ATOMIC
   EPS_SCF 1.0E-10
   EPS_DIIS 0.0
   MAX_SCF 200
   &DIAGONALIZATION
    ALGORITHM STANDARD
   &END DIAGONALIZATION
   &MIXING
    METHOD BROYDEN_MIXING
    ALPHA 0.5
    NBROYDEN 6
   &END MIXING
  &END SCF
  &XC
   &XC_FUNCTIONAL
    &{exchange}
    &END {exchange}
    &{correlation}
    &END {correlation}
   &END XC_FUNCTIONAL
   &XC_GRID
    XC_DERIV PW
    XC_SMOOTH_RHO NONE
   &END XC_GRID
  &END XC
  &POISSON
   PERIODIC XYZ
  &END POISSON
 &END DFT
 &SUBSYS
  &CELL
   ABC [bohr] 12 11 10
   PERIODIC XYZ
  &END CELL
  &COORD
   UNIT bohr
   H 5.45 5.15 4.4
   H 6.55 5.65 4.8
  &END COORD
  &KIND H
   BASIS_SET DZV-GTH-PADE
   POTENTIAL GTH-PBE-q1
  &END KIND
 &END SUBSYS
 &PRINT
  &FORCES
  &END FORCES
  &STRESS_TENSOR
  &END STRESS_TENSOR
 &END PRINT
&END FORCE_EVAL
'''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse(text):
 energy=re.findall(r'ENERGY\| Total FORCE_EVAL \([^)]*\) energy \[(?:a\.u\.|hartree)\]:\s*([-+\d.EeDd]+)',text)
 blocks=re.findall(r'ATOMIC FORCES in \[a\.u\.\].*?\n(.*?)(?:SUM OF ATOMIC FORCES|\n\s*STRESS)',text,re.S)
 forces=[]
 if blocks:
  forces=[[float(x) for x in row] for row in re.findall(r'^\s*\d+\s+\d+\s+H\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)',blocks[-1],re.M)]
 stressblocks=re.findall(r'STRESS\| (?:Analytical|Numerical) stress tensor \[GPa\].*?\n(.*?)(?:STRESS\|\s*1/3|STRESS\|\s*Determinant)',text,re.S)
 tensor=None
 if stressblocks:
  rows=re.findall(r'^\s*STRESS\|\s*[xyz]\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)',stressblocks[-1],re.M)
  if len(rows)==3:tensor=[[float(v) for v in row] for row in rows]
 version=re.findall(r'CP2K\| version string:\s*([^\n]+)',text)
 converged=bool(re.search(r'SCF run converged in\s+\d+\s+steps',text)) and 'PROGRAM ENDED AT' in text and not re.search(r'SCF run NOT converged|SCF run did not converge|ABORT',text)
 result={'engine':version[-1].strip() if version else None,'converged':converged,'energy_ha':float(energy[-1]) if energy else None,'forces_ha_bohr':forces,'stress_tensor_gpa':tensor,
         'printed_xc':[line.strip() for line in text.splitlines() if ('XC_FUNCTIONAL|' in line or 'FUNCTIONAL|' in line or 'LIBXC' in line)]}
 numbers=([result['energy_ha']] if result['energy_ha'] is not None else [])+[v for row in forces for v in row]+([v for row in tensor for v in row] if tensor else [])
 result['complete_printed_quantities']=len(forces)==2 and tensor is not None and len(numbers)==16 and all(math.isfinite(v) for v in numbers)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=float,default=300);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 binary=DATA.parents[2]/'bin/cp2k.psmp'
 report={'state':'capturing','scope':'Actual CP2K GPW H2 RKS PBE-family E/fullF/CP2K numerical stress; native public GPW supports diagonal stress only',
         'captured_utc':datetime.now(timezone.utc).isoformat(),'binary':str(binary),'binary_sha256':sha(binary),'script_sha256':sha(__file__),'predeclared_bounds':BOUNDS,
         'basis_data':str(DATA/'BASIS_SET'),'basis_data_sha256':sha(DATA/'BASIS_SET'),'potential_data':str(DATA/'GTH_POTENTIALS'),'potential_data_sha256':sha(DATA/'GTH_POTENTIALS'),
         'original_public_fixture':'tools/diagnostics/pbe-family/gpw.cpp','original_public_fixture_sha256':sha(ROOT/'tools/diagnostics/pbe-family/gpw.cpp'),
         'stress_convention':'CP2K numerical physical tensor in GPa, compressive positive; native comparison restricted to xx/yy/zz',
         'cases':[]}
 for key,xc,exchange,correlation in MODELS:
  identity='gpw-h2-rks-'+key;work=a.out/identity;work.mkdir()
  (work/'input.inp').write_text(TEMPLATE.format(basis=DATA/'BASIS_SET',potential=DATA/'GTH_POTENTIALS',exchange=exchange,correlation=correlation))
  script='''module load CP2K/2026.2-gcc-2026.03-mpich-eb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$1"
shift
python3 -c 'import json,os,socket;print(json.dumps({"hostname":socket.gethostname(),"environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","SLURM_JOB_ID","SLURM_STEP_ID","SLURM_JOB_NODELIST"]}},indent=2))' > execution.json
exec "$@"
'''
  command=[str(ROOT/'tools/diagnostics/gpw-driver/run-allocation.sh'),'cpu','--cpus','4','--','bash','-lc',script,'cp2k-oracle',str(work.resolve()),str(binary),'-i','input.inp']
  item={'id':identity,'xc':xc,'xc_sections':[exchange,correlation],'input':identity+'/input.inp','input_sha256':sha(work/'input.inp'),'output':identity+'/cp2k.out','command':command}
  start=time.monotonic()
  try:
   with (work/'cp2k.out').open('w') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
   item.update(parse((work/'cp2k.out').read_text()));item['exit_code']=r.returncode;item['output_sha256']=sha(work/'cp2k.out')
   if (work/'execution.json').exists():item['execution']=json.loads((work/'execution.json').read_text())
   item['accepted_external_output']=r.returncode==0 and item['converged'] and item['complete_printed_quantities']
  except (OSError,ValueError,subprocess.SubprocessError) as e:item.update(accepted_external_output=False,error=str(e))
  item['elapsed_wall_seconds']=time.monotonic()-start;report['cases'].append(item)
  (a.out/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print(('PASS' if item['accepted_external_output'] else 'FAIL')+' CP2K_CAPTURE '+identity+' E_Ha='+str(item.get('energy_ha')),flush=True)
 report['state']='captured' if all(c['accepted_external_output'] for c in report['cases']) else 'failed'
 (a.out/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return 0 if report['state']=='captured' else 1
if __name__=='__main__':raise SystemExit(main())
