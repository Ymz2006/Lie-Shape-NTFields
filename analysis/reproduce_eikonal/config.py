from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = 'GPU-ed74700f-7cd4-63be-4218-89dfd7b69248'
SEEDS = (1, 2, 3, 4, 5)
MODELS = {
    'metric': ('metric_baseline', 'metric', 'original', None),
    'aniso': ('lie_metric', 'lie', 'two_route', None),
    'iso': ('lie_isotropic', 'lie_iso', 'two_route', 'lie'),
    'iso_orig': ('lie_isotropic', 'lie_iso', 'original', 'lie'),
    'iso_chart': ('lie_isotropic', 'lie_iso', 'two_route', 'chart'),
    'l1': ('metric_lie_rhombus', 'rhombus', 'two_route', None),
}
NAMES = {
    'metric': 'Metric baseline', 'aniso': 'Lie anisotropic',
    'iso': 'Lie isotropic', 'iso_orig': 'Lie isotropic / original network',
    'iso_chart': 'Isotropic / chart gradient', 'l1': 'Anisotropic L1 block sum',
}
COLORS = ('#4f6488', '#b8741c', '#13867b', '#347eae', '#7656a8', '#b5452f')
BASE_SETS = (('', 'Standard 1k'), ('_tight_3d', 'Tight 0.005'),
             ('_tight_3d_exp1', 'Non-linear'), ('_tight_3d_exp2', 'Farthest'),
             ('_tight_3d_exp3', 'Hard + far'))
CELLS = {'Tshape3d_env1': 'T-shape', 'Lshape3d_env3': 'L-shape', 'Ashape3d_env4': 'A-shape'}


def sets_for(cell):
    sets = list(BASE_SETS)
    if cell == 'Tshape3d_env1':
        sets.insert(2, ('_tight001_3d', 'Very tight'))
        sets.extend((('_rotheavy', 'Rotation-heavy'), ('_bigangle', 'Big angles'),
                     ('_transonly', 'Translation only')))
    return [(cell + suffix, label) for suffix, label in sets]


def jobs():
    return [(cell, model, seed) for seed in SEEDS for cell in CELLS for model in MODELS]


def tag(cell, model, seed):
    return f'{cell}_{model}_s{seed}'


def ensure_gpu3():
    import subprocess
    import torch
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True)
    visible = query.stdout.strip().splitlines()
    if visible != [GPU_UUID] or torch.cuda.device_count() != 1:
        raise RuntimeError(f'Only physical device 3 may be exposed; visible UUIDs: {visible}')
