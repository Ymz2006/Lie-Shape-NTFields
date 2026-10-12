"""Integration checks for explicit seeds, score validation, and final reporting."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.reproduce_eikonal.config import CELLS, MODELS, ROOT, SEEDS, sets_for, tag
from analysis.reproduce_eikonal.report import build_report, holm
from analysis.reproduce_eikonal.run import validate_eval


def main():
    import torch
    smoke = ROOT / 'results/eikonal_five_seed_device3_smoke'
    status = json.loads((smoke / 'status.json').read_text())
    assert status['state'] == 'complete' and len(status['jobs']) == 6
    assert all(job['state'] == 'complete' for job in status['jobs'].values())
    assert not json.loads((smoke / 'summary.json').read_text())['complete']
    cell = 'Tshape3d_env1'
    name = tag(cell, 'iso', 1)
    checkpoint = next((smoke / 'checkpoints' / name).glob('Model_Epoch_00002_*.pt'))
    with tempfile.TemporaryDirectory(prefix='eikonal-validation-') as temp:
        root = Path(temp)
        subprocess.run([sys.executable, 'analysis/reproduce_eikonal/train_one.py', '--cell', cell,
                        '--model', 'iso', '--seed', '1', '--out', str(root / 'repeat'), '--epochs', '2'],
                       check=True)
        repeat = next((root / 'repeat' / name).glob('Model_Epoch_00002_*.pt'))
        first = torch.load(checkpoint, map_location='cpu')
        second = torch.load(repeat, map_location='cpu')
        assert torch.equal(first['B_state_dict'], second['B_state_dict'])
        assert all(torch.equal(v, second['model_state_dict'][k]) for k, v in first['model_state_dict'].items())
        print('Explicit seed repeat: Fourier features and all trained weights are bitwise identical.')
        for other in ('aniso', 'iso_chart'):
            p = next((smoke / 'checkpoints' / tag(cell, other, 1)).glob('Model_Epoch_00002_*.pt'))
            assert torch.equal(first['B_state_dict'], torch.load(p, map_location='cpu')['B_state_dict'])
        result_path = smoke / 'plan' / name / (cell + '.json')
        result = validate_eval(result_path, (cell, 'iso', 1), cell, checkpoint, 10)
        corrupted = copy.deepcopy(result)
        corrupted['summary']['seed'] = 9
        bad = root / 'bad.json'
        bad.write_text(json.dumps(corrupted))
        try:
            validate_eval(bad, (cell, 'iso', 1), cell, checkpoint, 10)
        except AssertionError:
            pass
        else:
            raise AssertionError('Wrong evaluation seed was accepted')
        corrupted = copy.deepcopy(result)
        corrupted['summary']['nonconv_rate'] += .01
        bad.write_text(json.dumps(corrupted))
        try:
            validate_eval(bad, (cell, 'iso', 1), cell, checkpoint, 10)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incorrect score was accepted')
        assert holm({'a': .01, 'b': .03, 'c': .5}) == {'a': .03, 'b': .06, 'c': .5}
        fixture = root / 'synthetic-report-only'
        fixture.mkdir()
        manifest = json.loads((smoke / 'manifest.json').read_text())
        manifest['epochs'] = 4000
        (fixture / 'manifest.json').write_text(json.dumps(manifest))
        (fixture / 'status.json').write_text(json.dumps({'state': 'complete', 'updated': 'synthetic validation', 'jobs': {}}))
        for ci, c in enumerate(CELLS):
            for mi, model in enumerate(MODELS):
                for seed in SEEDS:
                    folder = fixture / 'plan' / tag(c, model, seed)
                    folder.mkdir(parents=True)
                    for dataset, _ in sets_for(c):
                        # Deliberately synthetic scores for report logic only.
                        success = .5 + ci * .05 + mi * .02 + seed * .01
                        sm = dict(result['summary'], success_rate=success, collision_rate=.1,
                                  nonconv_rate=1-success-.1, n_cases=10, n_endpoints_ok=10)
                        rows = [{'case': i} for i in range(10)]
                        (folder / (dataset + '.json')).write_text(json.dumps({'summary': sm, 'records': rows, 'reproduction': {}}))
        build_report(fixture)
        summary = json.loads((fixture / 'summary.json').read_text())
        assert summary['complete'] and summary['evaluations'] == 570
        assert abs(summary['suites'][cell]['iso']['mean'] - 57) < 1e-10
        assert abs(summary['sets'][cell]['comparisons']['iso_vs_aniso']['delta_pp'] - 2) < 1e-10
        assert (fixture / 'figures/suite_success.svg').exists()
        assert (fixture / 'figures/failure_modes.svg').exists()
        missing = fixture / 'plan' / tag(cell, 'metric', 5) / (cell + '.json')
        missing.unlink()
        build_report(fixture)
        summary = json.loads((fixture / 'summary.json').read_text())
        assert not summary['complete'] and summary['evaluations'] == 569
        assert summary['sets'][cell]['comparisons']['iso_vs_metric']['welch_p'] is None
        assert summary['suites'][cell]['metric']['n'] == 4
    print('PASS: six model integrations, seed identity, matched Fourier draws, corrupted-score rejection, Holm correction, complete/partial report generation.')


if __name__ == '__main__':
    main()
