#!/usr/bin/env python3
"""Verify bundled raw hashes and export literal pins from independent QE XML."""
from pathlib import Path
import argparse,hashlib,json,re,xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args();root=a.root
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
manifest=json.loads((root/'rawrefs/manifest.json').read_text());cases=[]
for entry in manifest['primary_cases']:
 d=root/entry['raw_dir']
 for f,digest in entry['raw_files_sha256'].items():assert sha(d/f)==digest,(d/f,'hash')
 assert(d/'exit-code.txt').read_text().strip()=='0';raw=(d/'qe.out').read_text();assert 'JOB DONE.'in raw and 'convergence has been achieved'in raw and 'convergence NOT achieved'not in raw
 tree=ET.parse(d/'data-file-schema.xml').getroot();o=tree.find('output');inp=(d/'input.in').read_text()
 label=re.search(r"input_dft='([^']+)'",inp)[1].upper();expected='SLA+VWN+NOGX+NOGC' if entry['model']=='lsda' else entry['model'].upper();assert label==expected
 assert (o.findtext('band_structure/lsda')=='true')==entry['spin']
 for grid in ['fft_grid','fft_smooth']:
  node=o.find('basis_set/'+grid);assert [int(node.attrib[k])for k in ['nr1','nr2','nr3']]==entry['matched_quad']
 nat=3 if entry['spin']else 2;ff=[float(v)for v in o.findtext('forces').split()];ss=[float(v)for v in o.findtext('stress').split()];assert len(ff)==3*nat and len(ss)==9 and int(o.findtext('band_structure/nks'))==8
 cases.append({'id':entry['id'],'model':entry['model'],'spin':entry['spin'],'nat':nat,'quad':entry['matched_quad'],'Ne':float(o.findtext('band_structure/nelec')),'M':float(o.findtext('magnetization/total')or 0),'entropy':2*float(o.findtext('total_energy/demet')),'E':float(o.findtext('total_energy/etot')),'force':[ff[3*i:3*i+3]for i in range(nat)],'stress':[ss[i]for i in [0,4,8,1,2,5]]})
assert len(cases)==10 and {c['id']for c in cases}=={stem+'-'+model for stem in ['si2-rks','h3-isolated-uks']for model in ['pbe','pbesol','revpbe','rpbe','lsda']}
lines=['// Generated from the bundled actual QE7.6 XML/output; never native goldens.','#pragma once','#include "atomiverse/cpu/xc/xc_functional.h"','#include <array>','namespace pw_multi_k_fd_qe {','using Model=ATV::cpu::xc::XCType;','struct Pin {const char* id;Model model;bool spin;int nat;std::array<int,3> quadrature;double electrons,magnetization,entropy_ry,energy_ha;std::array<std::array<double,3>,3> forces_ha_bohr;std::array<double,6> stress_ha_bohr3;};','inline constexpr double energy_gate_ha=1e-5,force_gate_ha_bohr=1e-4,stress_gate_ev_ang3=2e-3;','inline constexpr double minimum_abs_magnetization=.05,minimum_abs_entropy_ry=1e-6;','inline constexpr std::array<Pin,10> pins={{']
for c in cases:
 f=c['force']+([[0.,0.,0.]]if c['nat']==2 else[]);ff='{'+','.join('{'+','.join(format(v,'.17g')for v in x)+'}'for x in f)+'}';ss='{'+','.join(format(v,'.17g')for v in c['stress'])+'}';qq='{'+','.join(str(v)for v in c['quad'])+'}'
 enum={'pbe':'PBE','pbesol':'PBEsol','revpbe':'RevPBE','rpbe':'RPBE','lsda':'LSDA'}[c['model']]
 lines.append('{"%s",Model::%s,%s,%d,%s,%s,%s,%s,%s,{%s},%s},'%(c['id'],enum,'true'if c['spin']else'false',c['nat'],qq,format(c['Ne'],'.17g'),format(c['M'],'.17g'),format(c['entropy'],'.17g'),format(c['E'],'.17g'),ff,ss))
lines+=['}};','} // namespace pw_multi_k_fd_qe',''];text='\n'.join(lines)
def tokens(s):
 s=re.sub(r'//[^\n]*|/\*.*?\*/','',s,flags=re.S)
 return re.findall(r'"(?:\\.|[^"\\])*"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|::|\S',s)
if a.check:assert tokens(text)==tokens((root/'multi_k_fd_pins.h').read_text()),'exported literal tokens differ from static pins'
if a.output:a.output.write_text(text)
print('PASS actualQE raw hashes and exact pin tokens',len(cases))
