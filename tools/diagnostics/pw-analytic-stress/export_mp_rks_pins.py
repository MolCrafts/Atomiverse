#!/usr/bin/env python3
"""Rebuild the single primary pin from captured actual QE XML, never a native golden."""
from pathlib import Path
import argparse,hashlib,json,re,xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--check',type=Path);a=p.parse_args();d=a.reference_dir;r=json.loads((d/'qe-reference.json').read_text())
for name,want in r['raw_files_sha256'].items():
 assert hashlib.sha256((d/'qe'/name).read_bytes()).hexdigest()==want, name
root=ET.parse(d/'qe/scratch/skew_si2_mp1_pbe.save/data-file-schema.xml').getroot();out=root.find('output');E=float(out.findtext('total_energy/etot'));S=2*float(out.findtext('total_energy/demet'));f=list(map(float,out.findtext('forces').split()));w=list(map(float,out.findtext('stress').split()));W=[w[0],w[4],w[8],w[1],w[2],w[5]]
assert E==r['E_Ha'] and S==r['minusTS_Ry'] and f==sum(r['full6F_HaBohr'],[]) and W==r['sigma_HaBohr3'];fmt=lambda v:format(v,'.17g')
s='#pragma once\n#include "multi_k_fd_pins.h"\nnamespace rks_mp_qe {\ninline constexpr pw_multi_k_fd_qe::Pin pin = {\n"si2-primitive-rks-pbe-mp1-002", pw_multi_k_fd_qe::Model::PBE, false, 2, {48,54,48},8,0,'+fmt(S)+','+fmt(E)+',{{{'+','.join(map(fmt,f[:3]))+'},{'+','.join(map(fmt,f[3:]))+'},{0,0,0}}},{'+','.join(map(fmt,W))+'}};\n}\n'
if a.check:
 tokens=lambda x:re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z_0-9]*|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[^\s]',x)
 assert tokens(s)==tokens(a.check.read_text()),'exported C++ token mismatch';print('PASS MP_RKS_PINS actual_QE_XML raw_hashes exact_tokens')
else:print(s,end='')
