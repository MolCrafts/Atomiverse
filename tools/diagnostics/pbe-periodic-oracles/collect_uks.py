#!/usr/bin/env python3
"""Collect four actual-CP2K H3 UKS comparisons; reuse narrow common checks."""
import argparse
import json
from pathlib import Path
import shutil
import collect_cp2k as common

IDS=tuple('gpw-h3-uks-'+model for model in ('pbe','pbesol','revpbe','rpbe'))
SOURCES=('uks.cpp','uks_fixture.h','uks_pins.h','export_uks_pins.py',
         'capture_cp2k.py','collect_cp2k.py','collect_uks.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--backend',required=True,choices=('CPU','CUDA'))
    parser.add_argument('--job',required=True,type=int)
    parser.add_argument('--exit-code',required=True,type=int)
    parser.add_argument('--log',default='main.log')
    parser.add_argument('--unit-xml',type=Path,action='append',default=[])
    parser.add_argument('--unit-list',type=Path,action='append',default=[])
    args=parser.parse_args()
    errors=[]
    reference=common.HERE/'cp2k-uks-2026-10-28/reference.json'
    cases={}
    try:
        cases=common.external_reference(reference,errors,expected_ids=IDS,
                                        atom_count=3,export_name='export_uks_pins.py',
                                        pins_name='uks_pins.h',require_spin=True)
    except (OSError,ValueError,KeyError,TypeError,AssertionError) as exc:
        errors.append('invalid actual UKS reference: '+str(exc))
    path=args.native/args.log
    try:
        log=path.read_text()
    except OSError as exc:
        errors.append('missing native UKS log: '+str(exc))
        log=''
    records=common.parse_native(log,args.backend,cases,errors,expected_ids=IDS,
                                atom_count=3,protocol='PBE_PERIODIC_CP2K_UKS',
                                force_protocol='GPW_CP2K_UKS_FORCE',
                                stress_protocol='GPW_CP2K_UKS_DIAG_STRESS',require_spin=True)
    if args.exit_code:
        errors.append('native command exited '+str(args.exit_code))
    provenance=common.provenance(args.native,args.backend,args.unit_xml,args.unit_list,errors,
                                 consumer_binary='pbe-periodic-uks-oracles',consumer_sources=SOURCES)
    result={'passed':not errors,'errors':errors,'backend':args.backend,'job_id':args.job,
            'native_directory':str(args.native),'native_log':str(path),
            'native_log_sha256':common.sha(path) if path.is_file() else None,
            'independent_engine':'CP2K 2026.2','external_reference_sha256':common.sha(reference),
            'external_stress_method':'CP2K NUMERICAL; 20 converged SCFs and alpha2/beta1 in every printed spin pair',
            'scope':'Periodic orthorhombic Gamma H3 UKS, Na2/Nb1, one 80Ry grid36x32x30; DZV-GTH-PADE/GTH-PBE-q1; PBE/PBEsol/revPBE/RPBE; absolute E, all9F, only3diagonal physical stresses',
            'original_physical_gates':common.BOUNDS,'records':records,'native_provenance':provenance,
            'formal_source_qualification':False,'formal_build_qualification':False,
            'general_functional_qualification':False,
            'limits':['No shear/full-six-tensor GPW qualification: only xx/yy/zz checked',
                      'No multigrid, k-mesh, smearing, other-system or other-potential qualification',
                      'No empty-beta or negative-raw-spin extension qualification; original CPU pointwise derivative failures retained separately',
                      'CP2K stress oracle is full-SCF numerical strain differentiation',
                      'Mulliken populations are optional; integer spin counts come from actual Tr(Palpha S)/Tr(Pbeta S)']}
    args.output.mkdir(parents=True,exist_ok=True)
    for source in args.native.iterdir():
        if source.is_file() and source.suffix in ('.log','.json','.sha256','.txt','.sh','.xml','.list'):
            destination=args.output/source.name
            if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
    for unit in provenance['unit_suites']:
        for key in ('xml','inventory'):
            source=Path(unit[key]);destination=args.output/source.name
            if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'cases':len(records),'errors':errors}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
