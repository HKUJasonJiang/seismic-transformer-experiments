"""Processed ground-motion text records: first line = sample count, then 'time acceleration' rows (s, m/s^2)."""
import hashlib
import re
from pathlib import Path
import numpy as np


def number(path):
    match = re.search(r'_(\d+)$', Path(path).stem)
    if not match:
        raise ValueError(f'Record file name lacks a numeric suffix: {path}')
    return int(match.group(1))


def read_record(path, sample_rate_hz=50.0, points=3000):
    path = Path(path)
    raw = path.read_bytes()
    tokens = raw.decode('ascii').split()
    count = int(float(tokens[0]))
    values = np.asarray(tokens[1:], dtype=np.float64)
    if len(values) != 2*count or count != points:
        raise ValueError(f'{path.name}: expected {points} two-column samples, header says {count}, found {len(values)//2}')
    values = values.reshape(count, 2)
    time, acceleration = values[:, 0], values[:, 1]
    dt = 1/sample_rate_hz
    if abs(time[0]) > 1e-9 or not np.allclose(np.diff(time), dt, rtol=0, atol=1e-6):
        raise ValueError(f'{path.name}: time must start at 0 with uniform {dt} s spacing')
    if not np.isfinite(acceleration).all() or not np.abs(acceleration).max() > 0:
        raise ValueError(f'{path.name}: acceleration must be finite and nonzero')
    return {'id': path.stem, 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'time': np.arange(count)*dt, 'acceleration': acceleration}


def read_library(folder, prefix):
    files = sorted(Path(folder).glob(f'{prefix}_*.txt'), key=number)
    if not files:
        raise ValueError(f'No {prefix}_*.txt records in {folder}')
    return [read_record(f) for f in files]
