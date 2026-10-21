"""Short engine execution with optional PID-filtered GPU/host memory sampling."""
from pathlib import Path
import os
import select
import subprocess
import threading
import time


def run_engine(command, budget, measure_memory=False, memory_command=None):
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=budget)
    wall = time.perf_counter()-start
    if result.returncode:
        raise RuntimeError(f"engine failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    resources = None
    if measure_memory:
        if memory_command is None:
            raise ValueError('memory measurement needs a separate command/output destination')
        _, probe_wall, resources = _run_memory_probe(memory_command, budget)
        resources['probe_process_seconds'] = probe_wall
        resources['separate_from_throughput'] = True
    return result, wall, resources


def _run_memory_probe(command, budget):
    start = time.perf_counter()

    # nvidia-smi uses NVML, with a documented 100-ms polling period. Filter
    # strictly by child PID: other allocations on this node are not our memory.
    monitor = subprocess.Popen(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
         '--format=csv,noheader,nounits', '--loop-ms=100'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    resources = dict(gpu_peak_mib=0, host_peak_rss_mib=0, gpu_samples=0,
                     gpu_uuids=[], gpu_poll_interval_ms=100,
                     gpu_metric='observed peak process GPU memory, including CUDA context; 100-ms sampling',
                     host_metric='maximum observed /proc/PID/status VmHWM')
    errors = []
    stop = threading.Event()
    child = None
    pending = ''

    def consume(chunk):
        nonlocal pending
        pending += chunk.decode()
        lines = pending.split('\n')
        pending = lines.pop()
        for line in lines:
            fields = [v.strip() for v in line.split(',')]
            if len(fields) == 3 and fields[0] == str(child.pid):
                memory = float(fields[1])
                resources['gpu_peak_mib'] = max(resources['gpu_peak_mib'], memory)
                resources['gpu_samples'] += 1
                if fields[2] not in resources['gpu_uuids']:
                    resources['gpu_uuids'].append(fields[2])

    def sample():
        try:
            while not stop.is_set():
                try:
                    for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                        if line.startswith('VmHWM:'):
                            resources['host_peak_rss_mib'] = max(
                                resources['host_peak_rss_mib'], int(line.split()[1])/1024)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                if select.select([monitor.stdout], [], [], 0.025)[0]:
                    chunk = os.read(monitor.stdout.fileno(), 65536)
                    if not chunk:
                        if monitor.poll() is not None:
                            raise RuntimeError('nvidia-smi memory monitor exited early')
                        continue
                    consume(chunk)
        except Exception as exc:
            errors.append(str(exc))

    sampler = None
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            stdout, stderr = child.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise
        wall = time.perf_counter()-start
        result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=2)
            if sampler.is_alive():
                errors.append('memory sampler did not stop')
        monitor.terminate()
        tail, _ = monitor.communicate(timeout=2)
        if child is not None:
            try:
                consume(tail)
            except Exception as exc:
                errors.append(str(exc))
    if result.returncode:
        raise RuntimeError(f"engine failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    if errors or resources['gpu_samples'] == 0 or len(resources['gpu_uuids']) != 1 or resources['host_peak_rss_mib'] <= 0:
        raise RuntimeError(f'invalid process memory measurement: {errors}, {resources}')
    return result, wall, resources
