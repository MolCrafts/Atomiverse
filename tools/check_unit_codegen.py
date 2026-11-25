#!/usr/bin/env python3
"""Compare optimized unit conversions against handwritten scalar arithmetic."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE = r'''
#include "atomiverse/core/units.h"
using namespace ATV::units;
extern "C" double raw_scale(double scale,double x) { return x*scale; }
extern "C" double unit_scale(Conversion scale,double x) { return scale(x); }
extern "C" double raw_identity(double x) { return x; }
extern "C" double unit_identity(double x) {
    constexpr auto u=from_style(Style::metal);
    return u.electric_field(x);
}
extern "C" double raw_static(double x) {
    constexpr auto u=PhysicalUnits::from_style(Style::real);
    return x*(u.electric_field().scale/(u.force().scale/u.charge().scale));
}
extern "C" double unit_static(double x) {
    constexpr auto u=PhysicalUnits::from_style(Style::real);
    return convert<u.electric_field(),u.force()/u.charge()>(x);
}
'''

def instructions(assembly, symbol):
    body=assembly.split('\n'+symbol+':\n',1)[1].split('\n\t.size',1)[0]
    return [line.strip() for line in body.splitlines()
            if line.startswith('\t') and not line.lstrip().startswith(('.', '#'))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',default=os.environ.get('CXX','c++'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='atv-units-codegen-') as directory:
        cpp=Path(directory)/'units.cpp'
        asm=Path(directory)/'units.s'
        cpp.write_text(SOURCE)
        subprocess.run([args.compiler,'-std=c++20','-O3','-S','-I'+str(root/'include'),str(cpp),'-o',str(asm)],check=True)
        assembly=asm.read_text()
        for suffix in ('scale','identity','static'):
            raw=instructions(assembly,'raw_'+suffix)
            wrapped=instructions(assembly,'unit_'+suffix)
            if raw != wrapped:
                raise SystemExit(f'{suffix}: generated instructions differ: {raw!r} != {wrapped!r}')
            print(f'{suffix}: identical optimized instructions')


if __name__ == '__main__':
    main()
