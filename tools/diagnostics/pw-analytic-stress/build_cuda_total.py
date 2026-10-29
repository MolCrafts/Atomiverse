#!/usr/bin/env python3
"""Link this opt-in diagnostic against a selected, already-built static archive.

Reuses an existing CUDA-enabled executable's generated flags/link dependencies;
never configures or mutates that shared build. All outputs go to --output.
"""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path.cwd())
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--template-target', default='gpw-driver-uks-pbe')
parser.add_argument('--family', action='store_true')
parser.add_argument('--fd', action='store_true')
parser.add_argument('--uks', action='store_true')
parser.add_argument('--domain', action='store_true')
parser.add_argument('--legacy-reference', action='store_true')
parser.add_argument('--include-root', type=Path)
parser.add_argument('--baseline-source-manifest', type=Path)
parser.add_argument('--candidate-archive', type=Path)
a = parser.parse_args()
if sum([a.family,a.fd,a.uks,a.domain,a.legacy_reference])>1:parser.error("choose one diagnostic kind")
if a.legacy_reference and not a.include_root:parser.error("legacy reference requires matching archived --include-root")
root = a.root.resolve()
build = (root / a.build).resolve()
archive = (root / a.archive).resolve()
out = (root / a.output).resolve()
baseline_attestation = None
if a.legacy_reference:
    if not a.baseline_source_manifest or not a.candidate_archive:
        parser.error('legacy reference requires --baseline-source-manifest and --candidate-archive')
    from verify_legacy_baseline import verify
    baseline_attestation = verify(archive, (root / a.include_root).resolve(),
        (root / a.baseline_source_manifest).resolve(),
        (root / a.candidate_archive).resolve())
out.mkdir(parents=True, exist_ok=True)
source_dir = Path(__file__).resolve().parent
target = build / 'CMakeFiles' / (a.template_target + '.dir')
flags = dict(line.split(' = ', 1) for line in
             (build / 'CMakeFiles/atomiverse.dir/flags.make').read_text().splitlines()
             if ' = ' in line)
base = shlex.split((target / 'link.txt').read_text())
nvcc = shlex.split((target / 'dlink.txt').read_text())[0]
commands = []
def run(cmd):
    commands.append(cmd)
    subprocess.run(cmd, cwd=build, check=True, close_fds=False)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
source_files = sorted(f for f in source_dir.iterdir() if f.is_file() and f.suffix in ['.cpp','.h','.py','.cmake'])
closure = {str(f):sha(f) for f in source_files}
pins = source_dir / 'qe-2026-10-28/pins.h'
closure[str(pins)] = sha(pins)
for f in (source_dir/'qe-gamma-pbe-fixed-2026-10-28').rglob('*'):
    if f.is_file():closure[str(f)]=sha(f)
for f in (source_dir/'qe-gamma-rks-fd-2026-10-28').rglob('*'):
    if f.is_file():closure[str(f)]=sha(f)
extra_include=[]
if a.include_root:
    include=(root/a.include_root).resolve();extra_include=['-I'+str(include)]
    for f in include.rglob('*'):
        if f.is_file():closure[str(f)]=sha(f)
for f in (source_dir/'qe-gamma-uks-fixed-2026-10-28').rglob('*'):
    if f.is_file():closure[str(f)]=sha(f)
archive_sha = sha(archive)
obj = out / 'cuda_total.o'
dlink = out / 'cuda_total-dlink.o'
binary_name = 'pw-analytic-stress-cuda'
source_name = 'cuda_total.cpp'
if a.family:binary_name,source_name='pw-analytic-stress-cuda-family','cuda_total_family.cpp'
if a.fd:binary_name,source_name='pw-analytic-stress-cuda-fd','cuda_total_fd.cpp'
if a.uks:binary_name,source_name='pw-analytic-stress-cuda-uks','cuda_uks_total.cpp'
if a.domain:binary_name,source_name='pw-analytic-stress-cuda-domain','cuda_uks_domain.cpp'
if a.legacy_reference:binary_name,source_name='pw-analytic-stress-cuda-legacy-reference','cuda_uks_legacy.cpp'
binary=out/binary_name

run([base[0]] + shlex.split(flags['CXX_DEFINES']) +
    extra_include + ['-I' + str(source_dir)] +
    shlex.split(flags['CXX_INCLUDES']) +
    ['-std=c++20', '-O2', '-fopenmp', '-Werror=return-type', '-c',
     str(source_dir / source_name), '-o', str(obj)])
rsp = out / 'device-libs.rsp'
rsp.write_text((target / 'deviceLinkLibs.rsp').read_text().replace(
    'libatomiverse.a', str(archive)))
run([nvcc, '-O3', '-DNDEBUG',
     '--generate-code=arch=compute_90,code=[compute_90,sm_90]',
     '-shared', '-dlink', str(obj), '-o', str(dlink),
     '--options-file', str(rsp)])
link = base.copy()
main_objects = 0
for i, arg in enumerate(link):
    if arg.startswith('-Wl,--dependency-file='):
        link[i] = '-Wl,--dependency-file=' + str(out / 'link.d')
    elif arg.endswith('cmake_device_link.o'):
        link[i] = str(dlink)
    elif arg.endswith('.cpp.o') and ('/' + a.template_target + '.dir/') in arg:
        link[i] = str(obj)
        main_objects += 1
    elif arg == 'libatomiverse.a':
        link[i] = str(archive)
assert main_objects == 1, 'Choose a template executable with one C++ source'
link[link.index('-o') + 1] = str(binary)
wraps = [
    '_ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishERNS2_13DeviceDensityERNS2_16DeviceBandSolverERKNS_3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_IdLm6EEPNS0_5FrameE',
    '_ZN3ATV4cuda2pw8internal16DeviceBandSolver17solve_final_bandsERNS_3cpu2pw8internal11PWWorkspaceENS2_20FinalBandDestinationEi',
    'cudaMemcpyAsync',
    '_ZN3ATV4cuda2pw8internal23project_frozen_nonlocalENS2_28FrozenNonlocalDeviceGeometryENS2_20FrozenBandDeviceViewENS2_27FrozenNonlocalDeviceScratchEP13cublasContextP11CUstream_st',
]
wraps += ['_ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspaceC1ERKNS_3cpu5FrameERNS4_2pw8internal9PWProblemEiNS4_2xc6XCTypeENS_10electronic8SmearingEdSt8optionalINS2_20FixedSpinOccupationsEEP11CUstream_stbb','_ZN3ATV4cuda2pw8internal9FFTPlan3DC1EiiiP11CUstream_st']
if a.domain:
    wraps = [wraps[0],wraps[1],'cudaMemcpyAsync','_ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace15commit_fallbackERNS2_13DeviceDensityERKNS_3cpu2pw8PWResultERKSt6vectorISt5arrayIdLm3EESaISD_EERKSC_IdLm6EEPNS0_5FrameE']
if a.legacy_reference:wraps=[]
link[1:1] = ['-Wl,--wrap=' + name for name in wraps]
run(link)
assert sha(archive) == archive_sha, 'Archive changed while linking'
if baseline_attestation:
    from verify_legacy_baseline import verify
    assert verify(archive, (root / a.include_root).resolve(),
        (root / a.baseline_source_manifest).resolve(),
        (root / a.candidate_archive).resolve()) == baseline_attestation, 'Baseline inputs changed during compilation'
for path, digest in closure.items():
    assert sha(Path(path)) == digest, 'Diagnostic source changed while linking'
(out / 'build-receipt.json').write_text(json.dumps({
    'source_closure_sha256': closure,
    'archive': str(archive), 'archive_sha256': archive_sha,
    'binary_sha256': sha(binary), 'commands': commands,
    'independent_baseline_attestation': baseline_attestation,
    'scope': 'Diagnostic-only GNU wrappers and oracle readbacks; no production hooks.',
}, indent=2) + '\n')
print(binary)
