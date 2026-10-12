"""Recompute statistics from fresh per-case JSON; never reuse historical scores."""
import csv
import html
import json
import math
import re
from pathlib import Path

from analysis.reproduce_eikonal.config import CELLS, COLORS, MODELS, NAMES, ROOT, SEEDS, sets_for, tag


def stats(values):
    import numpy as np
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'values': values}


def welch(a, b):
    from scipy.stats import ttest_ind
    if len(a) != 5 or len(b) != 5:
        return None
    value = float(ttest_ind(a, b, equal_var=False).pvalue)
    return value if math.isfinite(value) else (1.0 if a == b else 0.0)


def holm(values):
    ordered = sorted((p, key) for key, p in values.items() if p is not None)
    if len(ordered) != len(values):
        return {key: None for key in values}
    result, last = {}, 0.0
    for i, (p, key) in enumerate(ordered):
        last = max(last, min(1.0, (len(ordered) - i) * p))
        result[key] = last
    return result


def format_stat(value):
    if not value['n']:
        return 'pending'
    if value['sd'] is None:
        return f"{value['mean']:.2f} (n={value['n']})"
    return f"{value['mean']:.2f} ± {value['sd']:.2f} (n={value['n']})"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>'
        for row in rows) + '</tbody></table></div>'


def build_report(out):
    out = Path(out)
    if not (out / 'manifest.json').exists():
        return
    manifest = json.loads((out / 'manifest.json').read_text())
    status = json.loads((out / 'status.json').read_text())
    device_label = ', '.join(map(str, manifest.get('physical_devices', [3])))
    device_map = manifest.get('execution_device_map', {'3': manifest['gpu_uuid']})
    original = out / 'original.html'
    source = original.read_text() if original.exists() else (ROOT / 'Isotropic vs Anisotropic Lie Eikonal.html').read_text()
    historical = json.loads(re.search(r'const DATA = (.*?);\nconst NAMES', source, re.S).group(1))
    histories = {e['sets'][0]['id']: e for e in historical['envs']}
    entries, csv_rows, suites, suite_seed_rows = {}, [], {}, []
    count, denominator_issues = 0, []
    for cell in CELLS:
        suites[cell] = {}
        for dataset, label in sets_for(cell):
            entry = entries[dataset] = {'cell': cell, 'label': label, 'models': {}, 'comparisons': {}}
            endpoint_sets = []
            for model in MODELS:
                records = []
                for seed in SEEDS:
                    path = out / 'plan' / tag(cell, model, seed) / (dataset + '.json')
                    if not path.exists():
                        continue
                    result = json.loads(path.read_text())
                    if 'reproduction' not in result:
                        continue  # only validated, provenance-stamped evaluations
                    sm = result['summary']
                    if manifest['epochs'] != 4000:
                        continue  # smoke data never appear as benchmark results
                    records.append((seed, sm))
                    endpoint_sets.append(tuple(row['case'] for row in result['records']))
                    count += 1
                    csv_rows.append({'cell': cell, 'model': model, 'training_seed': seed,
                                     'evaluation_seed': 0, 'dataset': dataset,
                                     'n_cases': sm['n_cases'], 'n_endpoints_ok': sm['n_endpoints_ok'],
                                     'success_pct': 100 * sm['success_rate'],
                                     'collision_pct': 100 * sm['collision_rate'],
                                     'nonconvergence_pct': 100 * sm['nonconv_rate'],
                                     'checkpoint': sm['checkpoint'], 'result': str(path.relative_to(out)),
                                     'median_iters_success': sm['median_iters_success'],
                                     'median_trans_ratio_success': sm['median_trans_ratio_success'],
                                     'median_rotgeo_ratio_success': sm['median_rotgeo_ratio_success']})
                entry['models'][model] = {
                    'seeds': [seed for seed, _ in records],
                    **{key: stats([100 * sm[field] for _, sm in records]) for key, field in
                       [('success', 'success_rate'), ('collision', 'collision_rate'), ('nonconv', 'nonconv_rate')]},
                    'n_endpoints_ok': records[0][1]['n_endpoints_ok'] if records else None,
                }
            if endpoint_sets and any(ids != endpoint_sets[0] for ids in endpoint_sets):
                denominator_issues.append(dataset)
        for model in MODELS:
            values, valid_seeds = [], []
            for seed in SEEDS:
                rows = [row for row in csv_rows if row['cell'] == cell and row['model'] == model
                        and row['training_seed'] == seed]
                if len(rows) == len(sets_for(cell)):
                    valid_seeds.append(seed)
                    value = sum(row['success_pct'] for row in rows) / len(rows)
                    values.append(value)
                    suite_seed_rows.append({'cell': cell, 'model': model, 'training_seed': seed,
                                            'suite_success_pct': value,
                                            'suite_collision_pct': sum(r['collision_pct'] for r in rows) / len(rows),
                                            'suite_nonconvergence_pct': sum(r['nonconvergence_pct'] for r in rows) / len(rows)})
            suites[cell][model] = dict(stats(values), seeds=valid_seeds)
    expected = len(MODELS) * len(SEEDS) * sum(len(sets_for(c)) for c in CELLS)
    complete = count == expected and status['state'] == 'complete' and manifest['epochs'] == 4000
    comparisons = [('iso', 'metric'), ('iso', 'aniso'), ('iso', 'iso_orig'), ('iso', 'iso_chart')]
    raw_primary = {}
    for dataset, entry in entries.items():
        for a, b in comparisons:
            av, bv = entry['models'][a]['success']['values'], entry['models'][b]['success']['values']
            key = f'{a}_vs_{b}'
            p = welch(av, bv)
            entry['comparisons'][key] = {'delta_pp': sum(av) / 5 - sum(bv) / 5 if len(av) == len(bv) == 5 else None,
                                         'welch_p': p}
            if b in ('metric', 'aniso'):
                raw_primary[f'{dataset}:{key}'] = p
    adjusted = holm(raw_primary)
    for key, p in adjusted.items():
        dataset, comparison = key.split(':')
        entries[dataset]['comparisons'][comparison]['holm_p'] = p
    suite_comparisons = []
    for cell in CELLS:
        for a, b in comparisons + [('iso_orig', 'metric'), ('l1', 'aniso')]:
            av, bv = suites[cell][a]['values'], suites[cell][b]['values']
            suite_comparisons.append({'cell': cell, 'comparison': f'{a} − {b}',
                                     'delta_pp': sum(av) / 5 - sum(bv) / 5 if len(av) == len(bv) == 5 else None,
                                     'welch_p': welch(av, bv)})
    payload = {'complete': complete, 'evaluations': count, 'expected_evaluations': expected,
               'suites': suites, 'sets': entries, 'suite_comparisons': suite_comparisons,
               'denominator_issues': denominator_issues, 'status': status}
    (out / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    for name, rows in [('per_seed.csv', csv_rows), ('suite_per_seed.csv', suite_seed_rows)]:
        if rows:
            with (out / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    state = 'COMPLETE' if complete else ('SMOKE VALIDATION ONLY' if manifest['epochs'] != 4000 else 'IN PROGRESS / INCOMPLETE')
    text = [f'# Isotropic vs anisotropic Lie Eikonal: five-seed reproduction\n',
            f'**{state}** — {count}/{expected} validated full-set evaluations. Updated {status["updated"]}.\n',
            'Fresh training seeds 1–5 for six configurations in three environments (90 checkpoints, 570 evaluations). '
            f'GPU work uses authorized physical devices {device_label}. Evaluation seed 0 matches the supplied HTML.\n']
    sections = [f'<p class="status">{state} · {count}/{expected} validated full-set evaluations</p>',
                f'<p>Updated {html.escape(status["updated"])}. Fresh training seeds 1–5; physical GPUs {device_label}; '
                'six configurations, three environments, nineteen frozen test sets.</p>']
    if not complete:
        warning = 'The full reproduction is not complete. Partial numbers describe completed work only; no five-seed conclusions are claimed.'
        text.append(warning + '\n')
        sections.append('<p class="status">' + warning + '</p>')
    findings = []
    if complete:
        wins = sum(entries[d]['comparisons']['iso_vs_aniso']['delta_pp'] > 0 for d in entries)
        findings.append(f'Lie isotropic exceeds Lie anisotropic on {wins}/19 test-set means.')
        for cell in CELLS:
            delta = next(r['delta_pp'] for r in suite_comparisons if r['cell'] == cell and r['comparison'] == 'iso − metric')
            findings.append(f'{CELLS[cell]}: Lie isotropic minus metric baseline is {delta:+.2f} percentage points in suite success.')
        findings.append('Network and gradient controls are reported separately below; differences are quantified without assuming the original narrative reproduced.')
    for finding in findings:
        text.append(finding + '\n')
        sections.append('<p>' + html.escape(finding) + '</p>')
    headers = ['Environment'] + [NAMES[m] for m in MODELS]
    rows = [[CELLS[c]] + [format_stat(suites[c][m]) for m in MODELS] for c in CELLS]
    sections += ['<h2>Suite success (%)</h2>', table(headers, rows)]
    text += ['## Suite success (%)\n', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    text += ['| ' + ' | '.join(row) + ' |' for row in rows]
    sections.append('<p>Each seed is averaged equally across the environment’s test sets. Error bars are sample standard deviations across training seeds, not confidence intervals. L1 is the extra model mentioned in the original report’s notes.</p>')
    overall_rows = []
    for model in MODELS:
        means = [suites[cell][model]['mean'] for cell in CELLS]
        all_five = all(suites[cell][model]['n'] == 5 for cell in CELLS)
        overall_rows.append([NAMES[model], f'{sum(means) / 3:.2f}' if all_five else 'pending'])
    sections += ['<h2>Three-environment average</h2>', table(['Model', 'Success (%)'], overall_rows),
                 '<p>Equal weight for each environment. This is the aggregation used by the original note comparing L1 with Lie anisotropic.</p>']
    text += ['\n## Three-environment average\n', '| Model | Success (%) |', '| --- | --- |']
    text += ['| ' + ' | '.join(row) + ' |' for row in overall_rows]
    if suite_seed_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures = out / 'figures'
        figures.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for axis, cell in zip(axes, CELLS):
            for i, (model, color) in enumerate(zip(MODELS, COLORS)):
                s = suites[cell][model]
                if not s['n']:
                    continue
                axis.scatter([i + (j - (s['n'] - 1) / 2) * .07 for j in range(s['n'])], s['values'], color=color, alpha=.6)
                axis.errorbar(i, s['mean'], yerr=s['sd'] or 0, fmt='_', ms=18, color=color, capsize=4)
            axis.set_title(CELLS[cell])
            axis.set_xticks(range(len(MODELS)), ['Metric', 'Aniso', 'Iso', 'Iso orig', 'Chart', 'L1'], rotation=35)
            axis.set_ylim(0, 100)
            axis.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Suite success (%)')
        fig.tight_layout()
        fig.savefig(figures / 'suite_success.svg')
        fig.savefig(figures / 'suite_success.png', dpi=180)
        plt.close(fig)
        sections.append('<figure><img src="figures/suite_success.svg" alt="Suite success across five training seeds"><figcaption>Dots are completed training-seed suites; mean ± one standard deviation.</figcaption></figure>')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for axis, field, title in zip(axes, ('suite_collision_pct', 'suite_nonconvergence_pct'), ('Collision', 'Nonconvergence')):
            for i, (model, color) in enumerate(zip(MODELS, COLORS)):
                v = [[r[field] for r in suite_seed_rows if r['cell'] == c and r['model'] == model] for c in CELLS]
                heights = [sum(a) / len(a) if a else float('nan') for a in v]
                axis.bar([j + (i - 2.5) * .12 for j in range(3)], heights, width=.12, label=NAMES[model], color=color)
            axis.set_title(title)
            axis.set_xticks(range(3), list(CELLS.values()))
            axis.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Mean rate (%)')
        axes[1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(figures / 'failure_modes.svg')
        plt.close(fig)
        sections += ['<h2>How runs fail</h2>', '<img src="figures/failure_modes.svg" alt="Collision and nonconvergence rates">',
                     '<p>Collision and nonconvergence can overlap. Success requires convergence and no collision.</p>']
    control_rows = [[CELLS[r['cell']], r['comparison'],
                     f"{r['delta_pp']:+.2f}" if r['delta_pp'] is not None else 'pending',
                     f"{r['welch_p']:.5g}" if r['welch_p'] is not None else 'pending'] for r in suite_comparisons]
    sections += ['<h2>Loss, network, and gradient controls</h2>',
                 table(['Environment', 'Comparison', 'Δ success (pp)', 'Welch p'], control_rows),
                 '<p>Iso − iso_orig changes the network and its Fourier initialization; iso_orig − metric changes the speed target and gradient. Iso − iso_chart fixes the two-route network and translational speed while changing the rotational gradient. These controls quantify combined implementation effects rather than a pure architecture-only causal effect.</p>']
    text += ['\n## Suite comparisons\n', '| Environment | Comparison | Δ success (pp) | Welch p |', '| --- | --- | --- | --- |']
    text += ['| ' + ' | '.join(row) + ' |' for row in control_rows]
    for cell in CELLS:
        sections.append('<h2>' + CELLS[cell] + ': every test set</h2>')
        text.append('\n## ' + CELLS[cell] + ': every test set\n')
        headers = ['Test set', 'Valid cases'] + list(NAMES.values()) + ['iso − metric / p / Holm', 'iso − aniso / p / Holm']
        rows = []
        for dataset, label in sets_for(cell):
            e = entries[dataset]
            row = [label, next((m['n_endpoints_ok'] for m in e['models'].values() if m['n_endpoints_ok'] is not None), 'pending')]
            for model in MODELS:
                m = e['models'][model]
                row.append(format_stat(m['success']) + (f"; coll {m['collision']['mean']:.2f}; nonconv {m['nonconv']['mean']:.2f}" if m['success']['n'] else ''))
            for key in ('iso_vs_metric', 'iso_vs_aniso'):
                c = e['comparisons'][key]
                row.append(f"{c['delta_pp']:+.2f} / {c['welch_p']:.5g} / {c['holm_p']:.5g}" if c.get('holm_p') is not None else 'pending')
            rows.append(row)
        sections.append(table(headers, rows))
        text.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        text.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    historical_rows = []
    for cell in CELLS:
        for model in list(MODELS)[:5]:
            old = stats(histories[cell]['suite'][model])
            new = suites[cell][model]
            delta = new['mean'] - old['mean'] if new['n'] == 5 else None
            historical_rows.append([CELLS[cell], NAMES[model], format_stat(old), format_stat(new), f'{delta:+.2f}' if delta is not None else 'pending'])
    sections += ['<h2>Comparison with the supplied HTML</h2>',
                 table(['Environment', 'Model', 'Original suite %', 'Fresh suite %', 'Δ fresh − original (pp)'], historical_rows)]
    text += ['\n## Original versus fresh suite results\n', '| Environment | Model | Original | Fresh | Δ (pp) |', '| --- | --- | --- | --- | --- |']
    text += ['| ' + ' | '.join(row) + ' |' for row in historical_rows]
    notes = [
        'Protocol: 4,000 epochs; original batch size 2,000; original AdamW, accept/reject loop and speed warps; Eikonal only; TD, normal and RNL losses off. Only final checkpoint serialization is retained; cosmetic training plots are disabled.',
        'Seed initialization: Python random, NumPy and PyTorch CPU/CUDA are explicitly seeded before creating each model. Original trainer cuDNN settings are retained; explicit seeding does not promise bitwise identity across different software versions or hardware.',
        f'Runtime: PyTorch {manifest["torch"]}, CUDA {manifest["cuda"]}, Python {manifest["python"].split()[0]}. Authorized GPU UUIDs: {json.dumps(device_map)}. Each worker sees one assigned CUDA device, internally cuda:0. The sweep began on GPU 3 and expanded to GPUs 0–3 after user authorization; existing results were preserved.',
        'Loss implementation audit: N is already the Eikonal norm (the square root of the quadratic form). The sqrtN mapping takes an additional square root: 0.01(√N − 1)², summed over endpoints and weighted by exp(−τ/2). The reproduction keeps this exact nested-root implementation.',
        'Gradient audit: translation is differentiated in world coordinates and rotation is mapped to the body Lie gradient. The isotropic translation-block norm is invariant under rotation, so the world/body choice does not affect this norm.',
        'Network audit: original and two-route networks also draw Fourier features with different standard deviations (1 versus 2, both truncated to [−2,2]). The network control therefore includes the feature initialization difference.',
        'Planner: forward chart MPPI; evaluation seed 0; batch 125; 200 iterations; step 0.015; temperature 50; 50 proposals; horizon 5; chart convergence threshold 0.01; 50,000 environment surface points for collision checks. The later Lie MPPI controller is not used in this reproduction.',
        'Frozen cases are loaded from the nineteen existing test sets. Each evaluation excludes colliding start/goal endpoints, and uses the original waypoint collision check with the appended goal. It does not certify continuous swept-volume collision freedom.',
        'Statistics: sample mean ± sample standard deviation over five training seeds. Suite means average sets equally, so a small challenge set and the standard 1k set have equal weight. Welch tests reproduce the original analysis; Holm correction covers the 38 primary per-set comparisons (iso versus metric and anisotropic). Suite/control p-values are exploratory and unadjusted; p-values appear only when both configurations have all five seeds.',
        'Historical audit: all 330 displayed set/model/seed records in the supplied HTML match the existing evaluation JSON to its displayed precision. The existing training entry point does not set RNG seeds; the historical seed labels do not supply documented RNG states. Historical scores are never used as fresh reproduction results.',
        'The original L1 claim (67.7 versus 66.4) is evaluated again using the six-configuration sweep. L1 is an appendix comparison rather than one of the five plotted configurations.',
        'Input and source SHA-256 hashes, runtime, model settings, exact commands and checkpoint hashes are preserved in manifest.json, source/, checkpoints/*/training.json and plan/*/*.json. Historical inputs are retained as original.html.',
    ]
    if manifest.get('execution_updates'):
        notes.append('GPU expansion is recorded in manifest.json execution_updates. Training uses the frozen optimization code with a device/provenance adapter; collision workers were reduced from eight to two per evaluation to share the CPU across GPUs. Pool.map preserves case ordering and the per-case geometry checks are unchanged. The expansion retains the original output directory name.')
    if denominator_issues:
        notes.append('Endpoint-denominator mismatch detected for: ' + ', '.join(denominator_issues))
    sections.append('<h2>Method and audit</h2><ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in notes) + '</ul>')
    text += ['\n## Method and audit\n'] + ['- ' + note for note in notes]
    job_rows = [[name, value.get('state'), value.get('completed_sets', 0), value.get('current_set', ''), value.get('error', '')]
                for name, value in status['jobs'].items()]
    sections += ['<h2>Execution status</h2>', table(['Job', 'State', 'Completed sets', 'Current set', 'Error'], job_rows),
                 '<p><a href="per_seed.csv">Per-seed CSV</a> · <a href="suite_per_seed.csv">Suite CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="manifest.json">Reproducibility manifest</a> · <a href="REPORT.md">Markdown report</a></p>']
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    if status['state'] == 'running':
        document += '<meta http-equiv="refresh" content="60">'
    document += '<title>Five-seed Lie Eikonal reproduction</title><style>body{font:15px/1.6 system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1b2230}main{max-width:1200px;margin:auto;padding:32px}h1{font:600 34px Georgia,serif}h2{margin-top:36px;font:600 25px Georgia,serif}.status{padding:14px;border-left:4px solid #b8741c;background:white}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:12px}td,th{padding:8px;text-align:left;border:1px solid #d9dee6;white-space:nowrap}th{background:#eef1f5}img{max-width:100%}li{margin-bottom:10px}figcaption{color:#5a6474}a{color:#13867b}</style><main><h1>Isotropic vs anisotropic Lie Eikonal</h1><p>Fresh five-seed reproduction</p>'
    document += '\n'.join(sections) + '</main></html>'
    for name, content in [('REPORT.html', document), ('REPORT.md', '\n'.join(text) + '\n')]:
        temporary = out / (name + '.tmp')
        temporary.write_text(content)
        temporary.replace(out / name)
