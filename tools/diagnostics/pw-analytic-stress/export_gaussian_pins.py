#!/usr/bin/env python3
"""Check independent QE raw references and reproduce Gaussian literal pins."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

a = argparse.ArgumentParser()
a.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
a.add_argument('--raw-root', type=Path)
a.add_argument('--output', type=Path)
a.add_argument('--check', action='store_true')
a = a.parse_args()
rawroot = a.raw_root or a.root/'gaussian-rawrefs'
if not rawroot.exists():
    rawroot = a.root/'rawrefs'
m = json.loads((rawroot/'manifest.json').read_text())
cs = []
for entry in [*m['primary_cases'], *m['controls']]:
    d = rawroot/Path(entry['raw_dir']).name
    for name, digest in entry['raw_files_sha256'].items():
        assert hashlib.sha256((d/name).read_bytes()).hexdigest() == digest, (name, 'hash')
    assert (d/'exit-code.txt').read_text().strip() == '0'
    raw = (d/'qe.out').read_text()
    assert 'JOB DONE.' in raw and 'convergence has been achieved' in raw
    assert 'convergence NOT achieved' not in raw
    inp = (d/'input.in').read_text()
    assert re.search(r"smearing\s*=\s*'gaussian'", inp, re.I)
    assert abs(float(re.search(r'degauss\s*=\s*([0-9.eE+-]+)', inp)[1])-entry['width']) < 1e-16
    label = re.search(r"input_dft\s*=\s*'([^']+)'", inp, re.I)[1].upper()
    assert label == ('SLA+VWN+NOGX+NOGC' if entry['model']=='lsda' else entry['model'].upper())
    o = ET.parse(d/'data-file-schema.xml').getroot().find('output')
    assert (o.findtext('band_structure/lsda')=='true') == entry['spin']
    assert int(o.findtext('band_structure/nks')) == entry['kmesh']**3
    for grid in ['fft_grid', 'fft_smooth']:
        node = o.find('basis_set/'+grid)
        assert [int(node.attrib[k]) for k in ['nr1','nr2','nr3']] == entry['grid']
    f = list(map(float, o.findtext('forces').split()))
    s = list(map(float, o.findtext('stress').split()))
    assert len(f)==3*entry['nat'] and len(s)==9
    c = dict(entry)
    c.update(E=float(o.findtext('total_energy/etot')),
             Ne=float(o.findtext('band_structure/nelec')),
             M=float(o.findtext('magnetization/total') or 0),
             S_Ry=2*float(o.findtext('total_energy/demet')),
             f=[f[i:i+3] for i in range(0,len(f),3)],
             s=[s[i] for i in [0,4,8,1,2,5]])
    assert c['E']==entry['E'] and c['f']==entry['f'] and c['s']==entry['s']
    assert c['Ne']==entry['Ne'] and c['M']==entry['M'] and c['S_Ry']==entry['S_Ry']
    assert (abs(c['S_Ry'])>1e-6)==entry['entropy_qualified']
    assert not entry['spin'] or abs(c['M'])>.05
    cs.append(c)
assert len(cs)==13 and len(m['primary_cases'])==12
lines=['#pragma once','#include "diagnostic_recipe.h"','#include "multi_k_fd_pins.h"','#include <array>','namespace pw_gaussian_qe {','using Pin=pw_multi_k_fd_qe::Pin;using Model=pw_multi_k_fd_qe::Model;','struct Case { Pin pin;pw_analytic_diagnostic::Recipe recipe; };','inline constexpr std::array<Case,12>cases={{']
def row(c):
    f=c['f']+([[0.,0.,0.]]if c['nat']==2 else [])
    ff='{'+','.join('{'+','.join(format(v,'.17g')for v in x)+'}'for x in f)+'}'
    ss='{'+','.join(format(v,'.17g')for v in c['s'])+'}'
    qq='{'+','.join(map(str,c['grid']))+'}'
    model={'pbe':'PBE','pbesol':'PBEsol','revpbe':'RevPBE','rpbe':'RPBE','lsda':'LSDA'}[c['model']]
    pin='{"%s",Model::%s,%s,%d,%s,%s,%s,%s,%s,{%s},%s}'%(c['id'],model,'true'if c['spin']else'false',c['nat'],qq,format(c['Ne'],'.17g'),format(c['M'],'.17g'),format(c['S_Ry'],'.17g'),format(c['E'],'.17g'),ff,ss)
    recipe='{ATV::electronic::Smearing::gaussian,%s,%d,%s}'%(format(c['width'],'.17g'),c['kmesh'],'true'if c['isolated']else'false')
    return '{'+pin+','+recipe+'}'
lines += [row(c)+','for c in cs[:12]]
lines += ['}};','inline constexpr Case near_integer_control='+row(cs[12])+';','}']
text='\n'.join(lines)+'\n'
def tokens(s):
    s=re.sub(r'//[^\n]*|/\*.*?\*/','',s,flags=re.S)
    return re.findall(r'"(?:\\.|[^"\\])*"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|::|\S',s)
if a.check:
    assert tokens(text)==tokens((a.root/'gaussian_ensemble_pins.h').read_text()),'literal token mismatch'
if a.output:
    a.output.write_text(text)
print('PASS actualQE Gaussian raw hashes and literal pin tokens: 12 primary, 1 entropy-unqualified control')
