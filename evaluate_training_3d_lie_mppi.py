"""Run the standard evaluator with Lie MPPI and a Lie-model default.

All existing evaluator arguments and outputs remain available; for example:
    python evaluate_training_3d_lie_mppi.py --models lie_metric \
        --checkpoint <checkpoint.pt> --dataPath testing_data/3dshape/Tshape3d_env1 \
        --out results/lie_mppi/<run> --seed 0 --no-viser
"""

from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    for option, default in [('--sampler', 'lie'), ('--models', 'lie_metric')]:
        if not any(arg == option or arg.startswith(option + '=')
                   for arg in sys.argv[1:]):
            sys.argv.extend([option, default])
    runpy.run_path(str(Path(__file__).with_name('evaluate_training_3d_batched.py')),
                   run_name='__main__')
