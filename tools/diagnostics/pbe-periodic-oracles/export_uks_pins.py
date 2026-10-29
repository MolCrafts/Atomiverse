#!/usr/bin/env python3
"""Export four Gamma UKS pins exclusively from verified real CP2K stdout."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path)
p.add_argument('header',type=Path)
a=p.parse_args()
d=json.loads(a.reference.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert d['state']=='captured' and len(d['cases'])==4
assert {c['id'] for c in d['cases']}=={'gpw-h3-uks-'+x for x in ['pbe','pbesol','revpbe','rpbe']}
assert d['predeclared_bounds']=={'energy_ha':5e-6,'each_force_ha_bohr':5e-5,'each_supported_diagonal_stress_ha_bohr3':5e-5}
assert d['fixture']['n_alpha']==2 and d['fixture']['n_beta']==1
for key in ['basis_data','potential_data']:
 assert sha(a.reference.parent/d[key+'_snapshot'])==d[key+'_sha256']
assert sha(a.reference.parent/'capture-at-run.py')==d['script_sha256']
lines=['// Generated exclusively from actual independent CP2K Gamma UKS stdout.','#pragma once','#include <array>','namespace pbe_periodic_cp2k_uks {',
 'struct Pin { const char* id; int functional; double energy_ha; std::array<std::array<double,3>,3> forces_ha_bohr; std::array<double,3> diagonal_stress_gpa; };',
 'inline constexpr double energy_tolerance_ha=5e-6;',
 'inline constexpr double force_tolerance_ha_bohr=5e-5;',
 'inline constexpr double diagonal_stress_tolerance_ha_bohr3=5e-5;',
 'inline constexpr std::array<Pin,4> pins={{']
for c in d['cases']:
 assert c['accepted_external_output'] and c['exit_code']==0 and c['converged'] and c['scf_converged_runs']==20
 assert c['initial_grid_points']==[36,32,30] and c['n_alpha']==2 and c['n_beta']==1
 counts=c['all_printed_spin_electron_counts'];assert counts and counts==[2,1]*(len(counts)//2)
 for key in ['input','output']:assert sha(a.reference.parent/c[key])==c[key+'_sha256']
 assert all('FUNCTIONAL| '+x+':' in c['printed_xc'] for x in c['xc_sections'])
 array='{'+','.join('{'+','.join(format(v,'.17g') for v in row)+'}' for row in c['forces_ha_bohr'])+'}'
 sigma='{'+','.join(format(c['stress_tensor_gpa'][i][i],'.17g') for i in range(3))+'}'
 lines.append(' {"%s",%d,%.17g,{%s},%s},'%(c['id'],['PBE','PBEsol','revPBE','RPBE'].index(c['xc']),c['energy_ha'],array,sigma))
lines+=['}};','} // namespace pbe_periodic_cp2k_uks','']
a.header.write_text('\n'.join(lines))
print('Exported 4 actual independent UKS pins; reference SHA256',sha(a.reference))
