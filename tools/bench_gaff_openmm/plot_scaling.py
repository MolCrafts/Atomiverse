#!/usr/bin/env python3
"""Render accepted Arrhenius GAFF scaling measurements; never invent points."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    cohort = None
    binaries = {}
    devices = {}
    input_representations = {}
    openmm_version = None
    for precision in ('double', 'mixed'):
        seen_counts = set()
        files = sorted(args.results.glob(f'{precision}-r*/summary.json'))
        if not files:
            raise ValueError(f'no {precision} results')
        for path in files:
            result = json.loads(path.read_text())
            if result['N'] in seen_counts:
                raise ValueError(f'duplicate atom count within {precision}: {path}')
            seen_counts.add(result['N'])
            if not result['accepted']:
                raise ValueError(f'unaccepted scientific comparison: {path}')
            if 'resources' not in result:
                raise ValueError(f'no measured memory: {path}')
            protocol = result['protocol']
            signature = {key: protocol[key] for key in
                         ('steps', 'warmup', 'samples', 'dt_fs', 'cutoff_A',
                          'neighbor_skin_A', 'alpha_inv_A', 'grid_spacing_A',
                          'term', 'neighbor_reserve', 'throughput_monitor',
                          'resource_measurement', 'motion_bytes')}
            signature['data_sha256'] = result['inputs_sha256']['data']
            signature['ff_sha256'] = result['inputs_sha256']['ff']
            if cohort is None:
                cohort = signature
                openmm_version = result['openmm']['version']
            if signature != cohort or result['openmm']['version'] != openmm_version:
                raise ValueError(f'mixed benchmark settings, inputs or OpenMM versions: {path}')
            binary = result['inputs_sha256']['atv_binary']
            if precision in binaries and binaries[precision] != binary:
                raise ValueError(f'mixed Atomiverse binaries within {precision}: {path}')
            binaries[precision] = binary
            expected_position = 'float32+residual' if precision == 'mixed' else 'float64'
            if protocol['position_representation'] != expected_position or protocol['motion_bytes'] != 8:
                raise ValueError(f'incompatible integration precision: {path}')
            representation = protocol['initial_position_storage']
            if precision in input_representations and input_representations[precision] != representation:
                raise ValueError(f'mixed input rounding conventions within {precision}: {path}')
            input_representations[precision] = representation
            if protocol['precision'] != precision or result['openmm']['precision'] != precision:
                raise ValueError(f'precision mismatch: {path}')
            if result['openmm']['platform'] != 'CUDA':
                raise ValueError(f'non-CUDA reference: {path}')
            device = result['resources']['atv']['gpu_uuids']
            if device != result['resources']['openmm']['gpu_uuids']:
                raise ValueError(f'engines ran on different GPUs: {path}')
            if precision in devices and devices[precision] != device:
                raise ValueError(f'mixed physical GPUs within {precision}: {path}')
            devices[precision] = device
            for engine in ('atv', 'openmm'):
                ms = result['atv_ms_per_step'] if engine == 'atv' else result['openmm']['ms_per_step']
                memory = result['resources'][engine]
                rows.append(dict(precision=precision, engine=engine, atoms=result['N'],
                                 ms_per_step=ms, steps_per_second=1000/ms,
                                 gpu_peak_mib=memory['gpu_peak_mib'],
                                 host_peak_rss_mib=memory['host_peak_rss_mib'],
                                 gpu_samples=memory['gpu_samples'],
                                 whole_comparison_seconds=result['benchmark_seconds'],
                                 pme_mesh='x'.join(map(str, result['openmm']['pme'])),
                                 gpu_uuid=memory['gpu_uuids'][0],
                                 initial_force_relative_error=result['relative_max_force_error'],
                                 max_final_position_error_A=result['max_final_position_error_A'],
                                 source=str(path)))
    rows.sort(key=lambda row: (row['precision'], row['engine'], row['atoms']))
    counts = {precision: sorted({r['atoms'] for r in rows if r['precision'] == precision})
              for precision in ('double', 'mixed')}
    if counts['double'] != counts['mixed'] or len(counts['double']) < 3:
        raise ValueError('both precisions need at least three matching atom counts')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'gaff_scaling.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, NullLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#CAD3DE', 'text.color': '#172B45',
                         'axes.labelcolor': '#172B45', 'xtick.color': '#4B5D72',
                         'ytick.color': '#4B5D72', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    colors = {'atv': '#187A85', 'openmm': '#C77732'}
    names = {'atv': 'Atomiverse', 'openmm': 'OpenMM '+openmm_version}
    metrics = [('ms_per_step', 1, 'Step time (ms) — lower is faster'),
               ('gpu_peak_mib', 1/1024, 'Observed peak GPU memory (GiB)'),
               ('host_peak_rss_mib', 1/1024, 'Observed peak host RSS (GiB)')]
    for col, precision in enumerate(('double', 'mixed')):
        axes[0, col].set_title(precision.capitalize()+' precision', loc='left',
                               fontsize=15, fontweight='bold', pad=15)
        for row, (metric, factor, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            endpoints = {engine: max((r for r in rows if r['precision'] == precision
                                      and r['engine'] == engine), key=lambda r: r['atoms'])[metric]*factor
                         for engine in ('atv', 'openmm')}
            for engine in ('atv', 'openmm'):
                points = sorted([r for r in rows if r['precision'] == precision
                                 and r['engine'] == engine], key=lambda r: r['atoms'])
                x = [r['atoms'] for r in points]
                y = [r[metric]*factor for r in points]
                ax.plot(x, y, color=colors[engine], marker='o' if engine == 'atv' else 's',
                        linewidth=2.3, markersize=6, label=names[engine])
                above = endpoints[engine] >= endpoints['openmm' if engine == 'atv' else 'atv']
                if endpoints['atv'] == endpoints['openmm']:
                    above = engine == 'atv'
                ax.annotate(f'{y[-1]:.3g}', (x[-1], y[-1]), xytext=(7, 7 if above else -7),
                            textcoords='offset points', va='center', color=colors[engine],
                            fontsize=10)
            ax.set_xscale('log')
            ax.set_xlim(counts[precision][0]*0.82, counts[precision][-1]*1.65)
            ax.set_xticks(counts[precision])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: f'{n/1000:g}k'))
            ax.minorticks_off()
            ax.grid(axis='both', color='#E7EDF3', linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_yscale('log')
                all_times = [r['ms_per_step'] for r in rows]
                ax.set_ylim(min(all_times)*0.75, max(all_times)*1.25)
                ticks = [m*10**e for e in range(-5, 5) for m in (1, 2, 5)
                         if min(all_times)*0.75 <= m*10**e <= max(all_times)*1.25]
                ax.set_yticks(ticks)
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                ax.legend(frameon=False, loc='upper left')
            else:
                ax.set_ylim(0, max(r[metric]*factor for r in rows)*1.15)
                ax.yaxis.set_major_locator(MaxNLocator(5))
            if row == 2:
                ax.set_xlabel('Number of atoms (log scale)')
    fig.suptitle('GAFF PEO scaling · one NVIDIA GH200', fontsize=22,
                 fontweight='bold', x=0.07, ha='left', y=0.98)
    fig.text(0.07, 0.945, f"Fixed density · matched PME and cutoff · {cohort['steps']} steps × {cohort['samples']} samples, median · all physics checks passed",
             fontsize=11, color='#4B5D72')
    fig.text(0.07, 0.025,
             'GPU: PID-filtered NVML samples every 100 ms, including CUDA context; transient peaks may be missed.\n'
             'Host: observed engine-process RSS, including C++ / Python front-end; comparison parent excluded. Normal neighbor capacity.\n'
             'Mixed: FP32 force/FFT, position + residual, FP64 velocity/integration in both engines. Exact common initial geometry.\n'
             'Points are measurements; connecting lines are guides. Throughput is unmonitored; memory is sampled in separate engine runs.',
             fontsize=9, color='#4B5D72', linespacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.96, top=0.9, bottom=0.12, hspace=0.25, wspace=0.3)
    fig.savefig(args.out/'gaff_scaling.png', dpi=180, facecolor='white')
    fig.savefig(args.out/'gaff_scaling.svg', facecolor='white')
    svg = args.out/'gaff_scaling.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(points=len(rows), atoms=counts['double'],
                          longest_comparison_seconds=max(r['whole_comparison_seconds'] for r in rows),
                          output=str(args.out))))


if __name__ == '__main__':
    main()
