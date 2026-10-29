#!/usr/bin/env python3
"""Check bundled independent QE fixed-band references and exact literal pins."""
import argparse,hashlib,json,re,math,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--raw-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args();rawroot=a.raw_root or a.root/'fixed-k-rawrefs'
if not rawroot.exists():rawroot=a.root/'rawrefs'
m=json.loads((rawroot/'manifest.json').read_text());cs=[]
for c in m['primary_cases']:
 d=rawroot/c['raw_dir']
 for n,h in c['raw_files_sha256'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,(n,'reference hash')
 assert(d/'exit-code.txt').read_text().strip()=='0';raw=(d/'qe.out').read_text();assert'Program PWSCF v.7.6'in raw and'JOB DONE.'in raw and'convergence has been achieved'in raw and'convergence NOT achieved'not in raw
 inp=(d/'input.in').read_text();assert"occupations='fixed'"in inp and'degauss'not in inp and'smearing='not in inp
 if c['spin']:assert'tot_magnetization=1'in inp and'starting_magnetization'not in inp
 assert re.search(r"input_dft='([^']+)'",inp)[1].upper()==('SLA+VWN+NOGX+NOGC'if c['model']=='lsda'else c['model'].upper())
 o=ET.parse(d/'data-file-schema.xml').getroot().find('output');assert(o.findtext('band_structure/lsda')=='true')==c['spin']and int(o.findtext('band_structure/nks'))==8
 for grid in ['fft_grid','fft_smooth']:
  n=o.find('basis_set/'+grid);assert[int(n.attrib[k])for k in['nr1','nr2','nr3']]==c['matched_quad']
 nat=3 if c['spin']else 2;f=list(map(float,o.findtext('forces').split()));s=list(map(float,o.findtext('stress').split()));assert len(f)==3*nat and len(s)==9 and all(math.isfinite(v)for v in f+s)
 E=float(o.findtext('total_energy/etot'));Ne=float(o.findtext('band_structure/nelec'));M=float(o.findtext('magnetization/total')or 0);S=2*float(o.findtext('total_energy/demet')or 0);ff=[f[i:i+3]for i in range(0,len(f),3)];ss=[s[i]for i in[0,4,8,1,2,5]]
 assert E==c['E']and ff==c['f']and ss==c['s']and Ne==c['Ne']and M==c['M']and S==0
 assert abs(Ne-(3 if c['spin']else 8))<1e-8 and abs(M-(1 if c['spin']else 0))<1e-8
 nb=[int(v.text)for v in o.find('band_structure')if 'nbnd'in v.tag];assert len(nb)==(2 if c['spin']else 1)and all(v==(2 if c['spin']else 4)for v in nb)
 cs.append(c)
assert len(cs)==10 and{(c['model'],c['spin'])for c in cs}=={(model,spin)for model in['lsda','pbe','pbesol','revpbe','rpbe']for spin in[False,True]}
lines=['#pragma once','#include "multi_k_fd_pins.h"','namespace pw_fixed_k_qe {','using Pin=pw_multi_k_fd_qe::Pin;using Model=pw_multi_k_fd_qe::Model;','inline constexpr std::array<Pin,10> pins={{']
for c in cs:
 f=c['f']+([[0.,0.,0.]]if not c['spin']else[]);ff='{'+','.join('{'+','.join(format(v,'.17g')for v in row)+'}'for row in f)+'}';ss='{'+','.join(format(v,'.17g')for v in c['s'])+'}';qq='{'+','.join(map(str,c['matched_quad']))+'}';model={'pbe':'PBE','pbesol':'PBEsol','revpbe':'RevPBE','rpbe':'RPBE','lsda':'LSDA'}[c['model']];lines+=['{"%s",Model::%s,%s,%d,%s,%s,%s,0,%s,{%s},%s},'%(c['id'],model,'true'if c['spin']else'false',3 if c['spin']else 2,qq,format(c['Ne'],'.17g'),format(c['M'],'.17g'),format(c['E'],'.17g'),ff,ss)]
lines+=['}};','} // namespace pw_fixed_k_qe'];text='\n'.join(lines)+'\n'
def tokens(s):
 s=re.sub(r'//[^\n]*|/\*.*?\*/','',s,flags=re.S);return re.findall(r'"(?:\\.|[^"\\])*"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|::|\S',s)
if a.check:assert tokens(text)==tokens((a.root/'fixed_k_pins.h').read_text()),'static literal tokens differ'
if a.output:a.output.write_text(text)
print('PASS actualQE fixed-band raw hashes and literal pin tokens:10primary; entropy0; global-insulator not inferred')
