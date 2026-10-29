#!/usr/bin/env python3
"""Actual CP2K Gamma UKS H3 four-model capture; no native scientific values."""
import argparse,hashlib,importlib.util,json,math,re,shutil,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path.cwd()
SOURCE=ROOT/'tools/diagnostics/pbe-periodic-oracles/capture_cp2k.py'
spec=importlib.util.spec_from_file_location('cp2k_capture',SOURCE);capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
DATA=capture.DATA
MODELS=[('pbe','PBE','GGA_X_PBE','GGA_C_PBE')]+capture.MODELS
POSITIONS_A=[(1.2,1.1,1.),(2.1,1.6,1.3),(1.6,2.25,1.8)]
POSITIONS_BOHR=[[x*1.8897261245650618 for x in p] for p in POSITIONS_A]
BOUNDS={'energy_ha':5e-6,'each_force_ha_bohr':5e-5,'each_supported_diagonal_stress_ha_bohr3':5e-5}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse(raw):
 r=capture.parse(raw)
 r['forces_ha_bohr']=[[float(x) for x in row] for row in re.findall(r'^\s*FORCES\|\s*\d+\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+[-+\d.EeDd]+',raw,re.M)][-3:]
 r['scf_converged_runs']=len(re.findall(r'SCF run converged in\s+\d+\s+steps',raw))
 r['printed_gaussian_core_overlap_ha']=[float(x) for x in re.findall(r'Overlap energy of the core charge distribution:\s*([-+\d.EeDd]+)',raw)]
 r['printed_electron_lines']=[line.strip() for line in raw.splitlines() if 'Number of electrons' in line or 'Spin ' in line or 'Multiplicity' in line]
 grid=re.findall(r'PW_GRID\|\s+Bounds\s+[123]\s+[-\d]+\s+[-\d]+\s+Points:\s+(\d+)',raw)
 r['initial_grid_points']=[int(x) for x in grid[:3]]
 values=([r['energy_ha']] if r['energy_ha'] is not None else [])+[v for f in r['forces_ha_bohr'] for v in f]+([v for row in r['stress_tensor_gpa'] for v in row] if r['stress_tensor_gpa'] else [])
 r['complete_printed_quantities']=len(values)==19 and len(r['forces_ha_bohr'])==3 and all(math.isfinite(v) for v in values)
 return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=float,default=600);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 binary=DATA.parents[2]/'bin/cp2k.psmp'
 for name in ['BASIS_SET','GTH_POTENTIALS']:shutil.copyfile(DATA/name,a.out/name)
 shutil.copyfile(__file__,a.out/'capture-at-run.py')
 report={'state':'capturing','scope':'Actual independent CP2K orthorhombic periodic Gamma UKS H3; integer Na2 Nb1; one 80Ry grid; four PBE models; no native qualification implied',
 'captured_utc':datetime.now(timezone.utc).isoformat(),'binary':str(binary),'binary_sha256':sha(binary),'script_sha256':sha(__file__),'shared_capture_source':str(SOURCE.relative_to(ROOT)),'shared_capture_source_sha256':sha(SOURCE),
 'predeclared_bounds':BOUNDS,'basis_data':str(DATA/'BASIS_SET'),'basis_data_sha256':sha(DATA/'BASIS_SET'),'basis_data_snapshot':'BASIS_SET','potential_data':str(DATA/'GTH_POTENTIALS'),'potential_data_sha256':sha(DATA/'GTH_POTENTIALS'),'potential_data_snapshot':'GTH_POTENTIALS',
 'fixture':{'atoms':'H3','cell_bohr':[12.,11.,10.],'positions_angstrom':POSITIONS_A,'positions_bohr':POSITIONS_BOHR,'angstrom_in_bohr':1.8897261245650618,'basis':'DZV-GTH-PADE','potential':'GTH-PBE-q1','cutoff_ry':80,'ngrids':1,'expected_grid_points':[36,32,30],'n_alpha':2,'n_beta':1,'multiplicity':2,'local_pp_on_grid':True,'periodic':'XYZ','kpoints':'Gamma','smearing':'none','scf_eps':1e-10,'scf_max':200},
 'stress_convention':'CP2K own NUMERICAL full-SCF compressive-positive tensor [bar] converted to GPa; native scope restricted xx/yy/zz; 20 converged SCF expected per case','cases':[]}
 coords='\n'.join('   H '+' '.join(format(v,'.17g') for v in row) for row in POSITIONS_BOHR)
 template=capture.TEMPLATE.replace(' &DFT\n',' &DFT\n  UKS\n  MULTIPLICITY 2\n').replace('   H 5.45 5.15 4.4\n   H 6.55 5.65 4.8',coords)
 for key,xc,exchange,correlation in MODELS:
  identity='gpw-h3-uks-'+key;work=a.out/identity;work.mkdir();(work/'input.inp').write_text(template.format(basis='../BASIS_SET',potential='../GTH_POTENTIALS',exchange=exchange,correlation=correlation))
  script='''module load CP2K/2026.2-gcc-2026.03-mpich-eb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$1"
shift
python3 -c 'import json,os,socket;print(json.dumps({"hostname":socket.gethostname(),"environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","SLURM_JOB_ID","SLURM_STEP_ID","SLURM_JOB_NODELIST"]}},indent=2))' > execution.json
exec "$@"
'''
  command=[str(ROOT/'tools/diagnostics/gpw-driver/run-allocation.sh'),'cpu','--cpus','4','--','bash','-lc',script,'cp2k-uks-oracle',str(work.resolve()),str(binary),'-i','input.inp']
  item={'id':identity,'xc':xc,'xc_sections':[exchange,correlation],'input':identity+'/input.inp','input_sha256':sha(work/'input.inp'),'output':identity+'/cp2k.out','command':command};start=time.monotonic()
  try:
   with (work/'cp2k.out').open('w') as log:run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
   item.update(parse((work/'cp2k.out').read_text()));item['exit_code']=run.returncode;item['output_sha256']=sha(work/'cp2k.out')
   if (work/'execution.json').exists():item['execution']=json.loads((work/'execution.json').read_text())
   item['accepted_external_output']=run.returncode==0 and item['converged'] and item['complete_printed_quantities'] and item['scf_converged_runs']==20 and item['initial_grid_points']==[36,32,30]
  except (OSError,ValueError,subprocess.SubprocessError) as e:item.update(accepted_external_output=False,error=str(e))
  item['elapsed_wall_seconds']=time.monotonic()-start;report['cases'].append(item);(a.out/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(('PASS' if item['accepted_external_output'] else 'FAIL')+' CP2K_UKS_CAPTURE '+identity+' E_Ha='+str(item.get('energy_ha')),flush=True)
 report['state']='captured' if all(c['accepted_external_output'] for c in report['cases']) else 'failed';(a.out/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return 0 if report['state']=='captured' else 1
if __name__=='__main__':raise SystemExit(main())
