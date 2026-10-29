#!/usr/bin/env python3
"""Export constexpr pins exclusively from accepted, hash-verified QE outputs."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path)
p.add_argument('header',type=Path)
a=p.parse_args();d=json.loads(a.reference.read_text())
expected={system+'-'+xc for system in ['si2-rks','h3-uks'] for xc in ['pbesol','revpbe','rpbe']}
assert d['state']=='captured' and len(d['cases'])==6 and {c['id'] for c in d['cases']}==expected
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
lines=['// Generated only from actual independent QE output. Never from Atomiverse.','#pragma once','#include <array>','namespace pbe_periodic_qe {',
       'struct Pin { const char* id; int functional; bool spin; int nat; double energy_ha; std::array<std::array<double, 3>, 3> forces_ha_bohr; std::array<double, 6> stress_ha_bohr3; };',
       '// functional: 0=PBEsol, 1=revPBE, 2=RPBE. Physical Voigt xx,yy,zz,xy,xz,yz.',
       'inline constexpr double energy_tolerance_ha = 1e-5;',
       'inline constexpr double force_tolerance_ha_bohr = 1e-4;',
       'inline constexpr double stress_tolerance_ev_ang3 = 2e-3;',
       'inline constexpr std::array<Pin, 6> pins = {{']
for c in d['cases']:
 assert c['accepted_external_output']
 assert sha(a.reference.parent/c['input'])==c['input_sha256']
 assert sha(a.reference.parent/c['output'])==c['output_sha256']
 assert sha(a.reference.parent/c['id']/c['potential'])==c['potential_sha256']
 forces=c['forces_ha_bohr']+([[0.,0.,0.]] if c['nat']==2 else [])
 array='{'+','.join('{'+','.join(format(v,'.17g') for v in row)+'}' for row in forces)+'}'
 sigma='{'+','.join(format(v,'.17g') for v in c['stress_voigt_ha_bohr3'])+'}'
 lines.append('  {"%s", %s, %s, %s, %s, {%s}, %s},' %(c['id'],['PBEsol','revPBE','RPBE'].index(c['xc']),'true' if c['nat']==3 else 'false',c['nat'],format(c['energy_ha'],'.17g'),array,sigma))
lines+=['}};','} // namespace pbe_periodic_qe','']
a.header.write_text('\n'.join(lines))
print('Exported 6 actual QE pins; reference SHA256',sha(a.reference))
