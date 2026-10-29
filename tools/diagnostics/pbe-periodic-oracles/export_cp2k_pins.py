#!/usr/bin/env python3
"""Export only accepted, hash-verified actual CP2K outputs into a new header."""
import argparse
import hashlib
import json
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path)
p.add_argument('header', type=Path)
a = p.parse_args()
d = json.loads(a.reference.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert d['state'] == 'captured'
assert len(d['cases']) == 3
assert {c['id'] for c in d['cases']} == {'gpw-h2-rks-' + x for x in ['pbesol', 'revpbe', 'rpbe']}
lines = ['// Generated exclusively from actual independent CP2K output.', '#pragma once', '#include <array>', 'namespace pbe_periodic_cp2k {',
         'struct Pin { const char* id; int functional; double energy_ha; std::array<std::array<double,3>,2> forces_ha_bohr; std::array<double,3> diagonal_stress_gpa; };',
         'inline constexpr double energy_tolerance_ha = 5e-6;',
         'inline constexpr double force_tolerance_ha_bohr = 5e-5;',
         'inline constexpr double diagonal_stress_tolerance_ha_bohr3 = 5e-5;',
         'inline constexpr std::array<Pin,3> pins = {{']
for c in d['cases']:
    assert c['accepted_external_output']
    for kind in ['input', 'output']:
        assert sha(a.reference.parent / c[kind]) == c[kind + '_sha256']
    assert all('FUNCTIONAL| ' + x + ':' in c['printed_xc'] for x in c['xc_sections'])
    forces = '{' + ','.join('{' + ','.join(format(v, '.17g') for v in row) + '}' for row in c['forces_ha_bohr']) + '}'
    stress = '{' + ','.join(format(c['stress_tensor_gpa'][i][i], '.17g') for i in range(3)) + '}'
    lines.append('  {"%s", %d, %.17g, {%s}, %s},' % (c['id'], ['PBEsol', 'revPBE', 'RPBE'].index(c['xc']), c['energy_ha'], forces, stress))
lines += ['}};', '} // namespace pbe_periodic_cp2k', '']
a.header.write_text('\n'.join(lines))
print('Exported 3 independent CP2K pins; reference SHA256', sha(a.reference))
