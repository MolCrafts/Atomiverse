import hashlib,json,re,shlex,subprocess
from pathlib import Path
root=Path.cwd()
base=root/'.claude/campaigns/dft-gap-wave2-20261026/cpu-pw'
build=base/'build'
out=base/'dfpt-guard';out.mkdir(exist_ok=True)
flags=(build/'CMakeFiles/atomiverse.dir/flags.make').read_text()
args=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
 args+=shlex.split(re.search(r'^'+key+r' = (.*)$',flags,re.M).group(1))
gtest_include=next(p for p in [root/'build/_deps/googletest-src/googletest/include',root/'build-hf-auto/_deps/googletest-src/googletest/include'] if p.exists())
compiler='/software/sse2/el9_epyc9005/manual/GCC/14.3.0/hpc1/bin/c++'
sources=[root/'src/cpu/planewave/dfpt_phonon.cpp',root/'tests/cpu/planewave/test_dfpt_admission.cpp',root/'src/cpu/vibration/supercell.cpp',root/'src/cpu/vibration/phonon_dispersion.cpp']
commands=[];objects=[]
for i,p in enumerate(sources):
 obj=out/f'guard-{i}.o';objects.append(str(obj))
 cmd=[compiler,*args,'-I'+str(gtest_include),'-c',str(p),'-o',str(obj)]
 commands.append(cmd);subprocess.run(cmd,cwd=build,check=True)
original=shlex.split((build/'CMakeFiles/pw-spin-smear-stress.dir/link.txt').read_text())
link=[original[0],*objects,str(root/'build-hf-auto/lib/libgtest_main.a')]
i=1
while i<len(original):
 if original[i]=='-o':i+=2;continue
 if not original[i].endswith('.o'):link.append(original[i])
 i+=1
exe=out/'test-dfpt-mp-admission'
link+=[str(root/'build-hf-auto/lib/libgtest.a'),'-o',str(exe)]
commands.append(link);subprocess.run(link,cwd=build,check=True)
with (out/'test.log').open('w') as stream:
 subprocess.run([str(exe)],cwd=build,stdout=stream,stderr=subprocess.STDOUT,check=True)
receipt={'sources_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'commands':commands,'library_sha256':hashlib.sha256((build/'libatomiverse.a').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'scope':'Independent six-case MP DFPT admission guard units; separate object link; no physical stress rerun'}
(out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print((out/'test.log').read_text())
