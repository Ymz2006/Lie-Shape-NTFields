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
                             'metric_lie_rhombus', 'lie_twist', 'lie_metric_rnl', 'lie_isotropic'],
                    help='Model package under models/: shape_metric (blocked '
                         'Eikonal), lie_metric (anisotropic Eikonal, Lie '
                         'gradient), metric_baseline (original metric NTFields), '
                         'lie_metric_6s (six per-dimension speeds; needs speeds_6.npy '
                         'from dataprocessing/preprocess_obj_6s.py) or metric_lie_rhombus '
                         '(lie_metric with the additive cost dt = da/St + db/Sr: rhombus, not ellipse) '
                         'or lie_twist (Riemannian Eikonal from body-twist optimal control, '
                         'with the shape\'s twist metric on se(3)) or lie_metric_rnl (lie_metric, '
                         '(sqrt(N)-1)^2, plus the relative normal loss of models/lie_metric_rnl/rnl_lie.py) '
                         'or lie_isotropic (isotropic Eikonal S_dist |(grad_v, grad_w)| = 1 on the Lie gradient: '
                         'translational speed only).')
parser.add_argument('--arch', default='two_route', choices=['two_route', 'original'],
                    help='Network for shape_metric / lie_metric / metric_lie_rhombus / lie_twist / lie_isotropic: two_route (separate '
                         'trans/rot encoders) or original (single-route metric NTFields). '
                         'metric_baseline / lie_metric_6s always use the original network.')
parser.add_argument('--dual-p', type=float, default=None,
                    help='metric_lie_rhombus only: exponent of the Eikonal dual norm '
                         '(default inf = max(St|grad_v|, Sr|grad_w|); 2 = lie_metric; '
                         '1 = the block sum St|grad_v| + Sr|grad_w|).')
parser.add_argument('--residual', default=None, choices=['N', 'sqrtN'],
                    help='lie_metric / lie_twist / lie_metric_rnl: Eikonal residual mapping, (N-1)^2 or '
                         '(sqrt(N)-1)^2 (defaults: lie_metric N, lie_twist / lie_metric_rnl sqrtN).')
parser.add_argument('--rot-input', default=None,
                    choices=['chart', 'matrix', 'euler6', 'rot6', 'rot6_direct', 'atlas', 'quat', 'quat_sheets'],
                    help='two_route only: rotation-route input, chart (rotvec Fourier features, default); '
                         'lie_twist: matrix (R(rotvec) features: tau on SO(3)); lie_metric: euler6 / rot6 / '
                         'rot6_direct / atlas / quat / quat_sheets (see models/lie_metric/model_network_metric.py ROT_INPUTS).')
parser.add_argument('--rot-freq', type=float, default=None,
                    help='lie_metric --rot-input rot6 / quat / quat_sheets only: std of the random Fourier '
                         'frequencies on the 6D / quaternion features (default 2).')
parser.add_argument('--twist-speed', default=None, choices=['block', 'dist', 'combined'],
                    help='lie_twist only: HJB speed form. block (default) G = D^-1 G0 D^-1 with '
                         'D = diag(S_dist, S_angle); dist / combined: scalar S (S_dist / '
                         'metric_baseline\'s combined speed) with constant G = G0.')
parser.add_argument('--grad', default=None, choices=['lie', 'chart'],
                    help='lie_twist / lie_isotropic only: left-trivialized gradient (lie, default) or the '
                         'coordinate gradient dtau/dchi (chart, the non-Lie control).')
parser.add_argument('--rot-scale', type=float, default=None,
                    help='lie_twist only: extra factor k on the rotation unit, G0 -> '
                         'diag(I,kI) G0 diag(I,kI) (rotation costs k times more per turn).')
parser.add_argument('--twist-metric', default=None, choices=['shape', 'identity', 'shape_scaled', 'shape_max'],
                    help='lie_twist only: Lie-algebra metric G0 (default shape = the shape\'s '
                         'surface twist metric; identity = lie_metric; shape_scaled = shape with '
                         'the rotation unit rescaled to mean rotational eigenvalue 1; shape_max = '
                         'per-axis worst-case point speed).')
parser.add_argument('--rnl', default=None, choices=['none', 'orig', 'lie'],
                    help='Relative normal loss on the band endpoint x0 (models/rnl.py for '
                         'shape_metric / lie_metric / metric_lie_rhombus / lie_twist, default none; '
                         'models/lie_metric_rnl/rnl_lie.py for lie_metric_rnl, default lie): none, '
                         'orig (coordinate probes, Shape-NTFields eqs. 6-7) or lie (Lie-group probe curves).')
# Unset RNL flags keep the package defaults (models/rnl.py::DEFAULTS, the
# Shape-NTFields values, or models/lie_metric_rnl/rnl_lie.py::DEFAULTS).
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
parser.add_argument('--rnl-ramp-start', type=int, default=None, help='Epoch the RNL weight ramp starts.')
parser.add_argument('--rnl-ramp-len', type=int, default=None, help='Epochs the RNL weight ramp lasts.')
parser.add_argument('--rnl-speed', default=None, choices=['aniso', 'block'],
                    help='lie_metric_rnl only: speed the RNL constrains, aniso (S*(x0) / N_x0, the '
                         'Eikonal-consistent speed, default) or block (1/|grad_v tau|, 1/|grad_w tau|).')
parser.add_argument('--rnl-anchor', default=None, choices=['net', 'detach', 'target'],
                    help='lie_metric_rnl only: S(x0) in the slope G = (S(probe) - S(x0)) / eps: '
                         'net (differentiated, Shape-NTFields), detach (default) or target (S*(x0)).')
parser.add_argument('--rnl-rho-mode', default=None, choices=['const', 'feasible', 'analytic'],
                    help='lie_metric_rnl only: slope threshold, const (rho_trans / rho_rot), feasible '
                         '(rho capped at -kappa S(x0) / eps) or analytic (the target field\'s own slope '
                         'along the probe, default).')
parser.add_argument('--rnl-kappa', type=float, default=None, help='lie_metric_rnl --rnl-rho-mode feasible: kappa.')
parser.add_argument('--rnl-balance', default=None, choices=['grad', 'norm', 'value', 'none'],
                    help='lie_metric_rnl only: how the RNL is held against the Eikonal term, value (default, '
                         'Shape-NTFields: loss value capped at --rnl-cap x the Eikonal one; needs the ramp from a '
                         'trained network, stalls if switched on early), grad (conflicting gradient component '
                         'removed, norm capped at --rnl-cap x the Eikonal one; never stalls, keeps the Eikonal fit), '
                         'norm (the norm cap only) or none.')
parser.add_argument('--rnl-backoff', type=float, default=None,
                    help='lie_metric_rnl only: multiply the RNL weight by this on every rejected epoch '
                         '(regrown x1.05 per accepted one); 0 = off (default).')
parser.add_argument('--init-checkpoint', default=None,
                    help='lie_metric_rnl only: continue from this checkpoint (network, optimizer, B '
                         'and epoch), e.g. a no-RNL lie_metric run at the epoch the RNL ramp starts.')
parser.add_argument('--speed-scale', type=float, default=None,
                    help='lie_metric_rnl only: factor on both speed targets (control for a uniformly '
                         'slower field; default 1).')
parser.add_argument('--accept-on', default=None, choices=['eik', 'total'],
                    help='lie_metric_rnl only: accept / reject epochs on the Eikonal-only (default) '
                         'or the total (Eikonal + RNL) loss ratio.')
parser.add_argument('--td-weight', type=float, default=None,
                    help='metric_baseline only: TD loss weight (original metric NTFields 1e-3; default 0, off).')
parser.add_argument('--normal-weight', type=float, default=None,
                    help='metric_baseline only: normal-alignment loss weight (original 1e-3; default 0, off).')
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
if args.models in ('shape_metric', 'lie_metric', 'metric_lie_rhombus', 'lie_twist', 'lie_metric_rnl',
                   'lie_isotropic'):
    model.Params['Network']['Architecture'] = args.arch
    cfg = model.Params['Training']['RNL']
    rnl_keys = (('mode', args.rnl), ('weight', args.rnl_weight), ('eps', args.rnl_eps),
                ('rho_trans', args.rnl_rho_trans), ('rho_rot', args.rnl_rho_rot),
                ('G_max', args.rnl_gmax), ('cap_ratio', args.rnl_cap),
                ('ramp_start', args.rnl_ramp_start), ('ramp_len', args.rnl_ramp_len))
    rnl_lie_keys = (('speed', args.rnl_speed), ('anchor', args.rnl_anchor),
                    ('rho_mode', args.rnl_rho_mode), ('kappa', args.rnl_kappa),
                    ('balance', args.rnl_balance), ('backoff', args.rnl_backoff))
    if args.models == 'lie_metric_rnl':
        rnl_keys = rnl_keys + rnl_lie_keys
        if args.accept_on is not None:
            model.Params['Training']['Accept on'] = args.accept_on
        model.Params['Training']['Init checkpoint'] = args.init_checkpoint
        if args.speed_scale is not None:
            model.Params['Training']['Speed scale'] = args.speed_scale
    else:
        if any(v is not None for _, v in rnl_lie_keys) or args.accept_on is not None \
                or args.init_checkpoint is not None or args.speed_scale is not None:
            parser.error('--rnl-speed / --rnl-anchor / --rnl-rho-mode / --rnl-kappa / --rnl-balance / '
                         '--rnl-backoff / --accept-on / --init-checkpoint / --speed-scale are only used by lie_metric_rnl')
        cfg['mode'] = 'none'       # models/rnl.py: RNL is opt-in for the other packages
    for key, val in rnl_keys:
        if val is not None:
            cfg[key] = val
elif args.rnl not in (None, 'none'):
    parser.error('--rnl is only implemented for shape_metric / lie_metric / metric_lie_rhombus / '
                 'lie_twist / lie_metric_rnl')
if args.dual_p is not None:
    if args.models != 'metric_lie_rhombus':
        parser.error('--dual-p is only used by metric_lie_rhombus')
    model.Params['Training']['Dual p'] = args.dual_p
if args.residual is not None:
    if args.models not in ('lie_metric', 'lie_twist', 'lie_metric_rnl', 'lie_isotropic'):
        parser.error('--residual is only used by lie_metric / lie_twist / lie_metric_rnl / lie_isotropic')
    model.Params['Training']['Residual'] = args.residual
if args.rot_input is not None:
    if args.models not in ('lie_twist', 'lie_metric'):
        parser.error('--rot-input is only used by lie_twist / lie_metric')
    if args.rot_input != 'chart' and (args.rot_input == 'matrix') != (args.models == 'lie_twist'):
        parser.error('--rot-input {} is not implemented for {}'.format(args.rot_input, args.models))
    model.Params['Network']['Rotation input'] = args.rot_input
if args.rot_freq is not None:
    if args.models != 'lie_metric' or args.rot_input not in ('rot6', 'quat', 'quat_sheets'):
        parser.error('--rot-freq is only used by lie_metric --rot-input rot6 / quat / quat_sheets')
    model.Params['Network']['Rotation freq'] = args.rot_freq
for flag, val, section, key in (('--twist-metric', args.twist_metric, 'Training', 'Twist metric'),
                                 ('--twist-speed', args.twist_speed, 'Training', 'Twist speed'),
                                 ('--grad', args.grad, 'Training', 'Gradient'),
                                 ('--rot-scale', args.rot_scale, 'Training', 'Rotation scale')):
    if val is not None:
        if args.models != 'lie_twist' and not (flag == '--grad' and args.models == 'lie_isotropic'):
            parser.error('{} is only used by lie_twist'.format(flag))
        model.Params[section][key] = val
for flag, val, key in (('--td-weight', args.td_weight, 'TD weight'),
                       ('--normal-weight', args.normal_weight, 'Normal weight')):
    if val is not None:
        if args.models != 'metric_baseline':
            parser.error('{} is only used by metric_baseline'.format(flag))
        model.Params['Training'][key] = val
start_run(args, model, task='3dshape')

start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))

finish_run(args)
