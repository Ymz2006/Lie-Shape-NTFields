"""Use the existing trainers unchanged, with explicit seeds and output-only hooks."""
import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.reproduce_eikonal.config import GPU_UUID, MODELS, ROOT, ensure_gpu3, tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4000)
    args = parser.parse_args()
    import numpy as np
    import torch
    ensure_gpu3()
    os.chdir(ROOT)
    package, _, arch, gradient = MODELS[args.model]
    module = importlib.import_module(f'models.{package}.model_train_metric')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    run = args.out.resolve() / tag(args.cell, args.model, args.seed)
    run.mkdir(parents=True, exist_ok=True)
    model = module.Model(str(run), f'datasets/3dshape/{args.cell}/', 6, [0.0] * 6, device='cuda:0')
    model.folder = str(run)
    cfg = model.Params['Training']
    cfg['Number of Epochs'] = args.epochs
    cfg['Save Every * Epoch'] = args.epochs
    if args.model == 'metric':
        cfg.update({'TD weight': 0.0, 'Normal weight': 0.0})
    else:
        model.Params['Network']['Architecture'] = arch
        cfg['RNL']['mode'] = 'none'
    if args.model in ('aniso', 'iso', 'iso_orig', 'iso_chart'):
        cfg['Residual'] = 'sqrtN'
    if gradient:
        cfg['Gradient'] = gradient
    if args.model == 'l1':
        cfg['Dual p'] = 1.0
    # Checkpoint serialization and plots do not influence optimization. Preserve
    # the final checkpoint only to fit the entire sweep in available disk space.
    original_save = model.save

    def save_final(epoch='', val_loss=''):
        if epoch == args.epochs:
            original_save(epoch=epoch, val_loss=val_loss)

    model.save = save_final
    if hasattr(model, 'save_latest'):
        model.save_latest = lambda **kwargs: None
    metadata = {'seed': args.seed, 'cell': args.cell, 'model': args.model,
                'package': package, 'physical_device': 3, 'gpu_uuid': GPU_UUID,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'params': model.Params, 'epochs': args.epochs,
                'notes': 'Final checkpoint only; cosmetic plots off; original optimization unchanged.'}
    (run / 'training.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    start = time.time()
    model.train()
    checkpoints = list(run.glob(f'Model_Epoch_{args.epochs:05d}_*.pt'))
    if len(checkpoints) != 1:
        raise RuntimeError(f'Expected exactly one final checkpoint: {checkpoints}')
    checkpoint = torch.load(checkpoints[0], map_location='cpu')
    if checkpoint['epoch'] != args.epochs:
        raise RuntimeError('Incorrect final epoch')
    metadata.update(checkpoint=str(checkpoints[0]), seconds=time.time() - start,
                    final_loss=float(model.total_train_loss[-1]) if model.total_train_loss else None,
                    completed=True)
    (run / 'training.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(json.dumps({'completed': str(run), 'seconds': metadata['seconds']}), flush=True)


if __name__ == '__main__':
    main()
