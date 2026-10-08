"""Validate research_tree/tree.yaml and render it.

    python research_tree/render.py                  # validate + write tree.html
    python research_tree/render.py --text           # print the tree in the terminal
    python research_tree/render.py --text --root lie_metric
"""
import argparse
import datetime
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STATUSES = ('ref', 'win', 'mixed', 'fail', 'running', 'idea', 'dropped')
FIELDS = ('id', 'parent', 'title', 'status', 'date', 'change', 'fwd', 'coll',
          'result', 'notes', 'runs', 'code')
ANSI = {'ref': '34', 'win': '32', 'mixed': '33', 'fail': '31',
        'running': '36', 'idea': '35', 'dropped': '90'}


def load(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    meta, nodes = doc.get('meta', {}), doc.get('nodes') or []
    errors, ids = [], {}
    for i, n in enumerate(nodes):
        where = 'node #{} ({})'.format(i + 1, n.get('id', '?'))
        for k in ('id', 'title', 'status'):
            if not n.get(k):
                errors.append('{}: missing "{}"'.format(where, k))
        for k in n:
            if k not in FIELDS:
                errors.append('{}: unknown field "{}"'.format(where, k))
        if n.get('status') not in STATUSES:
            errors.append('{}: status "{}" not in {}'.format(where, n.get('status'), STATUSES))
        for k in ('fwd', 'coll'):
            if k in n and not isinstance(n[k], (int, float)):
                errors.append('{}: "{}" must be a number'.format(where, k))
        for k in ('runs', 'code'):
            if isinstance(n.get(k), str):
                n[k] = [n[k]]
        if 'date' in n:
            n['date'] = str(n['date'])
        if n.get('id') in ids:
            errors.append('{}: duplicate id'.format(where))
        ids[n.get('id')] = n
    for n in nodes:
        p = n.get('parent')
        if p is not None and p not in ids:
            errors.append('{}: parent "{}" does not exist'.format(n.get('id'), p))
    for n in nodes:  # cycle check
        seen, cur = set(), n
        while cur is not None and cur.get('parent') in ids:
            if cur['id'] in seen:
                errors.append('{}: parent chain has a cycle'.format(n['id']))
                break
            seen.add(cur['id'])
            cur = ids[cur['parent']]
    if meta.get('reference') and meta['reference'] not in ids:
        errors.append('meta.reference "{}" does not exist'.format(meta['reference']))
    if errors:
        sys.exit('tree.yaml is invalid:\n  ' + '\n  '.join(errors))
    return meta, nodes


def children_of(nodes):
    kids = {n['id']: [] for n in nodes}
    for n in nodes:
        if n.get('parent') is not None:
            kids[n['parent']].append(n)
    return kids


def fmt_metric(n, by_id):
    if 'fwd' not in n:
        return ''
    s = '{:.1f}%'.format(n['fwd'])
    p = by_id.get(n.get('parent'))
    if p and 'fwd' in p:
        s += ' ({:+.1f})'.format(n['fwd'] - p['fwd'])
    return s


def print_text(nodes, root=None, color=True):
    by_id = {n['id']: n for n in nodes}
    kids = children_of(nodes)

    def line(n, prefix, branch):
        st = n['status']
        tag = '[{}]'.format(st)
        if color:
            tag = '\033[{}m{}\033[0m'.format(ANSI[st], tag)
        parts = [prefix + branch + n['title'], tag, fmt_metric(n, by_id)]
        ident = '\033[90m{}\033[0m'.format(n['id']) if color else n['id']
        print(' '.join(p for p in parts if p) + '  ' + ident)

    def walk(n, prefix):
        cs = kids[n['id']]
        for i, c in enumerate(cs):
            last = i == len(cs) - 1
            line(c, prefix, '└── ' if last else '├── ')
            walk(c, prefix + ('    ' if last else '│   '))

    roots = [by_id[root]] if root else [n for n in nodes if n.get('parent') is None]
    for i, r in enumerate(roots):
        if i:
            print()
        line(r, '', '')
        walk(r, '')


def write_html(meta, nodes, out):
    with open(os.path.join(HERE, 'template.html')) as f:
        tpl = f.read()
    data = {'meta': meta, 'nodes': nodes,
            'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
    blob = json.dumps(data, default=str).replace('</', '<\\/')
    with open(out, 'w') as f:
        f.write(tpl.replace('/*__TREE_DATA__*/null', blob))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tree', default=os.path.join(HERE, 'tree.yaml'))
    ap.add_argument('--out', default=os.path.join(HERE, 'tree.html'))
    ap.add_argument('--text', action='store_true', help='print the tree instead of writing HTML')
    ap.add_argument('--root', help='with --text: print only this subtree')
    ap.add_argument('--no-color', action='store_true')
    args = ap.parse_args()

    meta, nodes = load(args.tree)
    if args.text:
        if args.root and args.root not in {n['id'] for n in nodes}:
            sys.exit('no node "{}"'.format(args.root))
        print_text(nodes, args.root, color=not args.no_color and sys.stdout.isatty())
        return
    write_html(meta, nodes, args.out)
    counts = {s: sum(n['status'] == s for n in nodes) for s in STATUSES}
    print('{} nodes ({}) -> {}'.format(
        len(nodes), ', '.join('{} {}'.format(v, k) for k, v in counts.items() if v),
        os.path.relpath(args.out)))


if __name__ == '__main__':
    main()
