"""Training entry point for the 3-D shape path-planning task.

Run ``dataprocessing/preprocess_obj.py`` first to produce the .npy training
data, then start training with:

    python train/train_3dshape.py

The configuration space is SE(3): (x, y, z, rx, ry, rz), so dim = 6.  The three
rotation coordinates (rx, ry, rz) are a rotation vector stored normalized by
2*pi, exactly as produced by ``preprocess_obj.py``.  ``source`` is the goal
configuration used for the periodic travel-time plots written into the
experiment folder.

Experiment tracking (Weights & Biases) is on by default; configure it with the
flags from ``train/wandb_utils.py``, e.g.::

    python train/train_3dshape.py --project ntrl-demo --tags rectangle env1 \
        --notes "narrow margin" --epochs 5000 --batch-size 2000
    python train/train_3dshape.py --no-wandb            # disable tracking
"""

import sys
sys.path.append('.')

import os
import time
import argparse

import importlib
from train.wandb_utils import add_wandb_args, apply_overrides, start_run, finish_run

parser = argparse.ArgumentParser(description='Train the 3-D shape planner.')
parser.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1_yrot',
                    help='Directory holding the .npy training data.')
parser.add_argument('--modelPath', default='./Experiments/3dshape',
                    help='Experiment root the run folder is created under.')
parser.add_argument('--name', default=None,
                    help='Explicit run-folder name under --modelPath. Without it '
                         'the folder is <dataset parent>_<timestamp>, which two '
                         'runs launched in the same minute would collide on -- '
                         'pass it when sweeping several datasets in parallel.')
parser.add_argument('--device', default='cuda:0',
                    help='Torch device to train on, e.g. cuda:0, cuda:2, cpu.')
parser.add_argument('--models', default='shape_metric',
                    choices=['shape_metric', 'lie_metric', 'metric_baseline', 'lie_metric_6s',
                             'metric_lie_rhombus'],
                    help='Model package under models/: shape_metric (blocked '
                         'Eikonal), lie_metric (anisotropic Eikonal, Lie '
                         'gradient), metric_baseline (original metric NTFields), '
                         'lie_metric_6s (six per-dimension speeds; needs speeds_6.npy '
                         'from dataprocessing/preprocess_obj_6s.py) or metric_lie_rhombus '
                         '(lie_metric with the additive cost dt = da/St + db/Sr: rhombus, not ellipse).')
parser.add_argument('--arch', default='two_route', choices=['two_route', 'original'],
                    help='Network for shape_metric / lie_metric / metric_lie_rhombus: two_route (separate '
                         'trans/rot encoders) or original (single-route metric NTFields). '
                         'metric_baseline / lie_metric_6s always use the original network.')
parser.add_argument('--dual-p', type=float, default=None,
                    help='metric_lie_rhombus only: exponent of the Eikonal dual norm '
                         '(default inf = max(St|grad_v|, Sr|grad_w|); 2 = lie_metric).')
parser.add_argument('--rnl', default='none', choices=['none', 'orig', 'lie'],
                    help='Relative normal loss on the band endpoint x0 (models/rnl.py, '
                         'shape_metric / lie_metric only): none, orig (coordinate '
                         'probes, Shape-NTFields eqs. 6-7) or lie (Lie-group probe curves).')
# Unset RNL flags keep the Shape-NTFields values in models/rnl.py::DEFAULTS.
parser.add_argument('--rnl-weight', type=float, default=None,
                    help='Peak RNL weight (ramped in from epoch 1500 over 2000 epochs).')
parser.add_argument('--rnl-eps', type=float, default=None, help='RNL probe size eps.')
parser.add_argument('--rnl-rho-trans', type=float, default=None,
                    help='Target translational speed slope rho_trans.')
parser.add_argument('--rnl-rho-rot', type=float, default=None,
                    help='Target rotational speed slope rho_rot.')
parser.add_argument('--rnl-gmax', type=float, default=None, help='RNL slope clip G_max.')
parser.add_argument('--rnl-cap', type=float, default=None,
                    help='Cap on the RNL contribution as a fraction of the Eikonal one.')
add_wandb_args(parser)
args = parser.parse_args()

md = importlib.import_module('models.{}.model_train_metric'.format(args.models))

modelPath = args.modelPath

# --dataPath / --data set the dataset path; resolve BEFORE building the Model
# since the experiment folder name is derived from it.
dataPath = args.data or args.dataPath

# source / goal configuration (x, y, z, rx, ry, rz) -- rotvec stored normalized by 2*pi
model = md.Model(modelPath, dataPath, 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=args.device)
if args.name:
    model.folder = os.path.join(modelPath, args.name)

apply_overrides(model, args)
if args.models in ('shape_metric', 'lie_metric', 'metric_lie_rhombus'):
    model.Params['Network']['Architecture'] = args.arch
    cfg = model.Params['Training']['RNL']
    cfg['mode'] = args.rnl
    for key, val in (('weight', args.rnl_weight), ('eps', args.rnl_eps),
                     ('rho_trans', args.rnl_rho_trans), ('rho_rot', args.rnl_rho_rot),
                     ('G_max', args.rnl_gmax), ('cap_ratio', args.rnl_cap)):
        if val is not None:
            cfg[key] = val
elif args.rnl != 'none':
    parser.error('--rnl is only implemented for shape_metric / lie_metric / metric_lie_rhombus')
if args.dual_p is not None:
    if args.models != 'metric_lie_rhombus':
        parser.error('--dual-p is only used by metric_lie_rhombus')
    model.Params['Training']['Dual p'] = args.dual_p
start_run(args, model, task='3dshape')

start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))

finish_run(args)
